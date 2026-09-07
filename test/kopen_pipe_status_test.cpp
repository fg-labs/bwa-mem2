/* SPDX-License-Identifier: MIT */

/* test/kopen_pipe_status_test.cpp — contract test for the `<cmd` pipe input
 * path in src/kopen.cpp (kopen / kclose).
 *
 * The byte-identity gate runs only on plain FASTQ files, so it never exercises
 * the pipe opener. This standalone binary pins the properties that decide
 * whether a failed input producer can masquerade as a successful run, and that
 * shutdown cannot hang on a producer that lingers.
 *
 * A `<cmd` input forks `cmd` and reads its stdout (kopen.cpp's pipe branch,
 * bwa's own literal-leading-`<` syntax, NOT bash `<(...)`). kclose must reap
 * the producer and return its exit status: 0 for a clean exit, the exact
 * WEXITSTATUS for a non-zero exit, and non-zero for a crash by a signal it did
 * not itself receive from kclose. A producer that closes its own stdout but
 * keeps running must be terminated, not waited on forever.
 *
 * Coverage notes:
 *  - Exact statuses are asserted (not just non-zero) so a regression that
 *    flattened every failure to a constant is caught.
 *  - Both "empty output + exit 0" and "non-empty output + non-zero exit" cases
 *    exist, so kclose keying off stream-emptiness instead of exit status would
 *    fail here (the three-case confound the first version had is broken).
 *  - Both the direct-execvp branch (need_shell == 0) and the /bin/sh branch
 *    (need_shell == 1, triggered by a shell metacharacter) are exercised, plus
 *    the cmd2argv rejection of a `<`-only / whitespace-only command.
 *  - drain-then-close-then-kclose mirrors main_mem: the reader owns and closes
 *    the fd before kclose runs (kclose's documented precondition).
 *  - A SIGALRM watchdog turns any kclose hang into a loud test failure rather
 *    than a stuck CI job.
 *  - A `<cmd &` producer backgrounds a descendant and exits itself almost
 *    immediately; kclose must terminate that descendant too (via the shared
 *    process group), not just reap the shell it forked.
 *  - A local, unprivileged loopback HTTP server pins that an oversized/
 *    unterminated response header is rejected, not accepted with the unread
 *    tail delivered as body data (http_open parses a port out of the URL).
 *  - A local, unprivileged loopback FTP mock pins two passive-mode properties:
 *    the data connection is opened before RETR (RFC 959 ordering), and the host
 *    advertised in the PASV reply is ignored in favor of the control-connection
 *    peer (FTP-bounce / SSRF guard, CWE-918). ftp_open parses a port out of the
 *    URL, so this too runs against a loopback server rather than requiring a
 *    privileged port-21 host.
 *
 * No committed fixture: the producers are synthetic shell/exec commands built
 * only from `false`, a missing command, and /bin/sh builtins; the HTTP and FTP
 * cases use same-process loopback servers, no external network access.
 */

#include <cstdio>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <arpa/inet.h>

/* Declared in src/fastmap.h; forward-declared here to keep the test lean and
 * free of the aligner headers. Signatures must match src/kopen.cpp. */
void *kopen(const char *fn, int *_fd);
int kclose(void *a);

enum { ANY_NONZERO = -1 };  /* expected-status sentinel: assert rc != 0 */

/* Read the pipe to EOF so the producer runs to completion (or dies) before we
 * reap it, mirroring how the reader drains the stream. */
static void drain_to_eof(int fd)
{
    char buf[4096];
    while (read(fd, buf, sizeof buf) > 0) { /* discard */ }
}

/* Open `spec`, drain it, close the read end (as the reader does before kclose),
 * then check kclose's status. `expect` is the exact rc, or ANY_NONZERO to
 * require only that rc != 0. Returns 0 on the expected outcome, 1 on mismatch. */
static int check(const char *spec, int expect)
{
    int fd = -1;
    void *ko = kopen(spec, &fd);
    if (ko == 0 || fd < 0) {
        /* The pipe is created synchronously; a producer that will fail only
         * surfaces that at kclose, so kopen must still succeed here. */
        fprintf(stderr, "FAIL: %s: kopen returned no handle (ko=%p fd=%d)\n",
                spec, ko, fd);
        return 1;
    }
    drain_to_eof(fd);
    close(fd);  /* honor kclose's precondition: read end closed before reaping */
    int rc = kclose(ko);
    int ok = (expect == ANY_NONZERO) ? (rc != 0) : (rc == expect);
    if (!ok) {
        fprintf(stderr, "FAIL: %s: kclose returned %d (expected %s%d)\n",
                spec, rc, expect == ANY_NONZERO ? "!= " : "== ",
                expect == ANY_NONZERO ? 0 : expect);
        return 1;
    }
    return 0;
}

/* A `<`-only or whitespace-only command has no argv; kopen must reject it
 * (cmd2argv returns NULL) rather than fork a child with a NULL argv[0]. */
static int check_empty_cmd_rejected(const char *spec)
{
    int fd = -1;
    void *ko = kopen(spec, &fd);
    if (ko != 0) {
        fprintf(stderr, "FAIL: %s: kopen should reject an empty command\n", spec);
        kclose(ko);
        return 1;
    }
    return 0;
}

/* `<sleep 30 &` backgrounds a descendant and the shell that spawned it exits
 * almost immediately with status 0 -- reparenting `sleep` away from us before
 * kclose ever runs. kclose only knows the shell's own pid; without also
 * signalling the shell's process group, `sleep` survives kclose entirely.
 * The shell captures the descendant's pid to a temp file (`echo $!`) before
 * it exits so this test can poll for that pid's actual death afterward --
 * proving the descendant was terminated, not just that kclose returned. */
static int check_process_group_cleanup(void)
{
    char pidfile[] = "/tmp/kopen_pgtest_XXXXXX";
    int pf = mkstemp(pidfile);
    if (pf < 0) {
        fprintf(stderr, "FAIL: process_group_cleanup: mkstemp failed\n");
        return 1;
    }
    close(pf);

    // The backgrounded `sleep` must not inherit the pipe's write end (fd 1):
    // a background job keeps the parent shell's fds by default, so without
    // its own redirect it would hold the pipe open long after the shell
    // itself exits and drain_to_eof below would block on it, not on kclose.
    char spec[256];
    snprintf(spec, sizeof spec, "<sleep 30 >/dev/null 2>&1 & echo $! > '%s'", pidfile);

    int fd = -1;
    void *ko = kopen(spec, &fd);
    if (ko == 0 || fd < 0) {
        fprintf(stderr, "FAIL: process_group_cleanup: kopen returned no handle\n");
        unlink(pidfile);
        return 1;
    }
    drain_to_eof(fd);
    close(fd);
    int rc = kclose(ko);
    if (rc != 0) {
        fprintf(stderr, "FAIL: process_group_cleanup: kclose returned %d, expected 0\n", rc);
        unlink(pidfile);
        return 1;
    }

    FILE *fp = fopen(pidfile, "r");
    long bg_pid = -1;
    if (fp) {
        if (fscanf(fp, "%ld", &bg_pid) != 1) bg_pid = -1;
        fclose(fp);
    }
    unlink(pidfile);
    if (bg_pid <= 0) {
        fprintf(stderr, "FAIL: process_group_cleanup: could not read backgrounded pid\n");
        return 1;
    }

    /* kclose's own SIGKILL escalation has already run by the time it returns;
     * poll briefly to give the kernel a short window to finish reaping under
     * load rather than asserting instantaneously. */
    for (int i = 0; i < 20; ++i) {
        if (kill((pid_t)bg_pid, 0) != 0) return 0;  /* gone: PASS */
        struct timespec ts = {0, 20 * 1000 * 1000};  /* 20ms */
        nanosleep(&ts, NULL);
    }
    fprintf(stderr,
            "FAIL: process_group_cleanup: backgrounded pid %ld still alive after kclose\n",
            bg_pid);
    return 1;
}

/* http_open bounds its header read at bufsz-1 (0x10000) bytes and must reject
 * a response that never sends the terminating "\r\n\r\n" -- otherwise a
 * response that merely fills the buffer is accepted with its unfinished
 * header tail delivered to the caller as body data (corrupted input,
 * silently). http_open parses a port out of the URL (unlike ftp_open's
 * hardcoded port 21 -- see the file header comment), so this can run against
 * an unprivileged loopback server instead of needing a real network. */
static int check_http_oversized_header(void)
{
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        fprintf(stderr, "FAIL: http_oversized_header: socket() failed\n");
        return 1;
    }
    int one = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;  /* ephemeral port */
    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof addr) != 0) {
        fprintf(stderr, "FAIL: http_oversized_header: bind() failed\n");
        close(listen_fd);
        return 1;
    }
    socklen_t alen = sizeof addr;
    if (getsockname(listen_fd, (struct sockaddr *)&addr, &alen) != 0 ||
        listen(listen_fd, 1) != 0) {
        fprintf(stderr, "FAIL: http_oversized_header: getsockname/listen failed\n");
        close(listen_fd);
        return 1;
    }
    unsigned short port = ntohs(addr.sin_port);

    pid_t server = fork();
    if (server < 0) {
        fprintf(stderr, "FAIL: http_oversized_header: fork() failed\n");
        close(listen_fd);
        return 1;
    }
    if (server == 0) {
        /* Server child: accept one connection, send a status line that looks
         * like a real HTTP response, then bufsz+ bytes of header-shaped
         * filler with NO "\r\n\r\n" anywhere -- the exact case hdr_end
         * guards -- then exit. Bound the accept() wait: if the client never
         * connects (e.g. a bug upstream of this test), the parent's own
         * SIGALRM watchdog still catches the hang, but exiting this child
         * promptly avoids leaking an orphaned listener past the test run. */
        struct timeval accept_tv = {5, 0};
        setsockopt(listen_fd, SOL_SOCKET, SO_RCVTIMEO, &accept_tv, sizeof accept_tv);
        int c = accept(listen_fd, NULL, NULL);
        if (c >= 0) {
            struct timeval tv = {0, 200 * 1000};
            setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
            char discard[4096];
            ssize_t drained = read(c, discard, sizeof discard);  /* best-effort */
            (void)drained;

            static const char status[] = "HTTP/1.1 200 OK\r\n";
            ssize_t written = write(c, status, sizeof(status) - 1);
            (void)written;
            char filler[4096];
            memset(filler, 'a', sizeof filler);  /* no \r or \n anywhere */
            size_t total = 0;
            while (total < 0x10000 + sizeof filler) {
                ssize_t n = write(c, filler, sizeof filler);
                if (n <= 0) break;
                total += (size_t)n;
            }
            close(c);
        }
        close(listen_fd);
        _exit(0);
    }
    close(listen_fd);  /* parent doesn't accept */

    char url[64];
    snprintf(url, sizeof url, "http://127.0.0.1:%u/x", (unsigned)port);
    int fd = -1;
    void *ko = kopen(url, &fd);
    int fail = 0;
    if (ko != 0) {
        fprintf(stderr,
                "FAIL: http_oversized_header: kopen accepted an oversized, "
                "unterminated header (must fail)\n");
        kclose(ko);
        fail = 1;
    }
    int status;
    waitpid(server, &status, 0);
    return fail;
}

/* Read one line (up to and including '\n') from fd into buf, NUL-terminated.
 * The FTP mock uses this to consume one command per handshake step without
 * parsing it; the socket carries a receive timeout so a stalled client cannot
 * wedge the server child past the parent's watchdog. */
static void ftp_mock_read_line(int fd, char *buf, size_t sz)
{
    size_t n = 0;
    while (n + 1 < sz) {
        char c;
        if (read(fd, &c, 1) <= 0) break;
        buf[n++] = c;
        if (c == '\n') break;
    }
    buf[n] = 0;
}

static void ftp_mock_send(int fd, const char *s)
{
    ssize_t n = write(fd, s, strlen(s));
    (void)n;  /* best-effort: a write failure just fails the handshake below */
}

/* ftp_open must open the PASV data connection BEFORE issuing RETR (RFC 959
 * passive mode), and must NOT trust the host advertised in the PASV reply --
 * it reuses the control connection's peer address for the data connection
 * (FTP-bounce / SSRF guard, CWE-918). This same-process loopback mock is an
 * independent oracle for BOTH properties:
 *
 *   - It advertises a DIFFERENT, non-routable host (203.0.113.1, RFC 5737
 *     TEST-NET-3) in the 227 reply while its real data listener is on
 *     127.0.0.1. A client that honored the advertised host would never reach
 *     the listener, so a successful open proves the advertised host was
 *     ignored in favor of the control peer.
 *   - After the 227 reply it watches the data listener and the control channel
 *     together with select(): a correct client connects the data socket before
 *     writing RETR, so the pending data connection is observed before any RETR
 *     byte. If a RETR byte arrives on the control channel with no data
 *     connection pending, the mock reports reversed ordering (child exits 1).
 *
 * ftp_open takes the control port from the URL, so this runs unprivileged with
 * no real network. */
static int check_ftp_pasv_before_retr(void)
{
    int ctrl_listen = socket(AF_INET, SOCK_STREAM, 0);
    if (ctrl_listen < 0) {
        fprintf(stderr, "FAIL: ftp_pasv_before_retr: socket() failed\n");
        return 1;
    }
    int one = 1;
    setsockopt(ctrl_listen, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;  /* ephemeral */
    if (bind(ctrl_listen, (struct sockaddr *)&addr, sizeof addr) != 0) {
        fprintf(stderr, "FAIL: ftp_pasv_before_retr: bind() failed\n");
        close(ctrl_listen);
        return 1;
    }
    socklen_t alen = sizeof addr;
    if (getsockname(ctrl_listen, (struct sockaddr *)&addr, &alen) != 0 ||
        listen(ctrl_listen, 1) != 0) {
        fprintf(stderr, "FAIL: ftp_pasv_before_retr: getsockname/listen failed\n");
        close(ctrl_listen);
        return 1;
    }
    unsigned short ctrl_port = ntohs(addr.sin_port);

    pid_t server = fork();
    if (server < 0) {
        fprintf(stderr, "FAIL: ftp_pasv_before_retr: fork() failed\n");
        close(ctrl_listen);
        return 1;
    }
    if (server == 0) {
        struct timeval to = {5, 0};
        setsockopt(ctrl_listen, SOL_SOCKET, SO_RCVTIMEO, &to, sizeof to);
        int c = accept(ctrl_listen, NULL, NULL);
        close(ctrl_listen);
        if (c < 0) _exit(2);
        setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &to, sizeof to);

        /* Real data listener on loopback; its port is what the client must use. */
        int data_listen = socket(AF_INET, SOCK_STREAM, 0);
        if (data_listen < 0) { close(c); _exit(2); }
        setsockopt(data_listen, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        struct sockaddr_in da;
        memset(&da, 0, sizeof da);
        da.sin_family = AF_INET;
        da.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        da.sin_port = 0;
        socklen_t dal = sizeof da;
        if (bind(data_listen, (struct sockaddr *)&da, sizeof da) != 0 ||
            getsockname(data_listen, (struct sockaddr *)&da, &dal) != 0 ||
            listen(data_listen, 1) != 0) {
            close(c);
            close(data_listen);
            _exit(2);
        }
        unsigned short data_port = ntohs(da.sin_port);

        char line[256];
        ftp_mock_send(c, "220 mock ready\r\n");
        ftp_mock_read_line(c, line, sizeof line);  /* USER */
        ftp_mock_send(c, "331 need password\r\n");
        ftp_mock_read_line(c, line, sizeof line);  /* PASS */
        ftp_mock_send(c, "230 logged in\r\n");
        ftp_mock_read_line(c, line, sizeof line);  /* TYPE */
        ftp_mock_send(c, "200 type ok\r\n");
        ftp_mock_read_line(c, line, sizeof line);  /* PASV */
        char pasv[128];
        /* Advertise a bogus (non-loopback, non-routable) host; the client must
         * ignore it and reuse the control peer (127.0.0.1) instead. */
        snprintf(pasv, sizeof pasv,
                 "227 Entering Passive Mode (203,0,113,1,%u,%u)\r\n",
                 (unsigned)(data_port >> 8), (unsigned)(data_port & 0xff));
        ftp_mock_send(c, pasv);

        int ordered_ok = 0, data_fd = -1;
        for (;;) {
            fd_set rfds;
            FD_ZERO(&rfds);
            FD_SET(c, &rfds);
            FD_SET(data_listen, &rfds);
            int mx = c > data_listen ? c : data_listen;
            struct timeval tv = {5, 0};
            int s = select(mx + 1, &rfds, NULL, NULL, &tv);
            if (s <= 0) break;  /* timeout/error: ordered_ok stays 0 */
            /* Check the data listener first: if the connection is pending it
             * arrived before (or with) any RETR byte -- correct ordering. */
            if (FD_ISSET(data_listen, &rfds)) {
                data_fd = accept(data_listen, NULL, NULL);
                ordered_ok = 1;
                break;
            }
            if (FD_ISSET(c, &rfds)) break;  /* RETR before any data connection */
        }
        if (ordered_ok) {
            ftp_mock_read_line(c, line, sizeof line);  /* RETR */
            ftp_mock_send(c, "150 opening data\r\n");
            if (data_fd >= 0) {
                ftp_mock_send(data_fd, "X");  /* one byte so the client sees a stream */
                close(data_fd);
            }
            ftp_mock_send(c, "226 transfer complete\r\n");
        }
        if (data_fd >= 0) close(data_fd);
        close(data_listen);
        close(c);
        _exit(ordered_ok ? 0 : 1);
    }
    close(ctrl_listen);  /* parent doesn't accept */

    char url[64];
    snprintf(url, sizeof url, "ftp://127.0.0.1:%u/mock", (unsigned)ctrl_port);
    int fd = -1;
    void *ko = kopen(url, &fd);
    int fail = 0;
    if (ko == 0 || fd < 0) {
        fprintf(stderr,
                "FAIL: ftp_pasv_before_retr: kopen failed -- PASV-before-RETR "
                "ordering or the PASV-host (SSRF) guard regressed\n");
        fail = 1;
    } else {
        drain_to_eof(fd);
        close(fd);
        kclose(ko);
    }
    int status = 0;
    waitpid(server, &status, 0);
    if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
        fprintf(stderr,
                "FAIL: ftp_pasv_before_retr: mock reported RETR before the data "
                "connection (reversed ordering)\n");
        fail = 1;
    }
    return fail;
}

/* A control reply that is truncated -- the peer closes after the status digits
 * but before the terminating CRLF -- must FAIL the handshake, not be parsed as a
 * valid reply. kftp_get_response reads a byte at a time; on EOF before a
 * complete line it must return failure rather than running strtol() over the
 * partial buffer (which would recover the leading status code from a truncated
 * reply). The final RETR response is the one gate where a parsed-but-truncated
 * code would otherwise be accepted (ftp_open keeps the data fd iff it reads
 * 150), so this mock drives a full handshake, opens the data connection, then
 * sends a truncated "150 opening" with NO CRLF and closes. A correct client
 * rejects it (kopen == NULL); the pre-fix code accepted 150 and returned the fd.
 * This runs unprivileged on loopback with no real network. */
static int check_ftp_truncated_reply(void)
{
    int ctrl_listen = socket(AF_INET, SOCK_STREAM, 0);
    if (ctrl_listen < 0) {
        fprintf(stderr, "FAIL: ftp_truncated_reply: socket() failed\n");
        return 1;
    }
    int one = 1;
    setsockopt(ctrl_listen, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;  /* ephemeral */
    if (bind(ctrl_listen, (struct sockaddr *)&addr, sizeof addr) != 0 ||
        listen(ctrl_listen, 1) != 0) {
        fprintf(stderr, "FAIL: ftp_truncated_reply: bind/listen failed\n");
        close(ctrl_listen);
        return 1;
    }
    socklen_t alen = sizeof addr;
    if (getsockname(ctrl_listen, (struct sockaddr *)&addr, &alen) != 0) {
        fprintf(stderr, "FAIL: ftp_truncated_reply: getsockname failed\n");
        close(ctrl_listen);
        return 1;
    }
    unsigned short ctrl_port = ntohs(addr.sin_port);

    pid_t server = fork();
    if (server < 0) {
        fprintf(stderr, "FAIL: ftp_truncated_reply: fork() failed\n");
        close(ctrl_listen);
        return 1;
    }
    if (server == 0) {
        struct timeval to = {5, 0};
        setsockopt(ctrl_listen, SOL_SOCKET, SO_RCVTIMEO, &to, sizeof to);
        int c = accept(ctrl_listen, NULL, NULL);
        close(ctrl_listen);
        if (c < 0) _exit(2);
        setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &to, sizeof to);

        /* Real data listener on loopback (advertise its port in the 227). */
        int data_listen = socket(AF_INET, SOCK_STREAM, 0);
        if (data_listen < 0) { close(c); _exit(2); }
        setsockopt(data_listen, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        struct sockaddr_in da;
        memset(&da, 0, sizeof da);
        da.sin_family = AF_INET;
        da.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        da.sin_port = 0;
        socklen_t dal = sizeof da;
        if (bind(data_listen, (struct sockaddr *)&da, sizeof da) != 0 ||
            getsockname(data_listen, (struct sockaddr *)&da, &dal) != 0 ||
            listen(data_listen, 1) != 0) {
            close(c);
            close(data_listen);
            _exit(2);
        }
        unsigned short data_port = ntohs(da.sin_port);
        /* Bound the data accept() too, matching every other wait in this TU:
         * a client that opened control but never the data socket must hit the
         * 5s budget, not wedge until the 30s watchdog. */
        setsockopt(data_listen, SOL_SOCKET, SO_RCVTIMEO, &to, sizeof to);

        char line[256];
        ftp_mock_send(c, "220 mock ready\r\n");
        ftp_mock_read_line(c, line, sizeof line);  /* USER */
        ftp_mock_send(c, "331 need password\r\n");
        ftp_mock_read_line(c, line, sizeof line);  /* PASS */
        ftp_mock_send(c, "230 logged in\r\n");
        ftp_mock_read_line(c, line, sizeof line);  /* TYPE */
        ftp_mock_send(c, "200 type ok\r\n");
        ftp_mock_read_line(c, line, sizeof line);  /* PASV */
        char pasv[128];
        snprintf(pasv, sizeof pasv,
                 "227 Entering Passive Mode (127,0,0,1,%u,%u)\r\n",
                 (unsigned)(data_port >> 8), (unsigned)(data_port & 0xff));
        ftp_mock_send(c, pasv);
        /* ftp_open connects the data socket before sending RETR, so accept the
         * data connection first, then consume RETR. */
        int data_fd = accept(data_listen, NULL, NULL);
        ftp_mock_read_line(c, line, sizeof line);  /* RETR */
        /* Truncated final reply: status digits but no CRLF, then close. */
        ftp_mock_send(c, "150 opening");
        if (data_fd >= 0) close(data_fd);
        close(data_listen);
        close(c);
        _exit(0);
    }
    close(ctrl_listen);  /* parent doesn't accept */

    char url[64];
    snprintf(url, sizeof url, "ftp://127.0.0.1:%u/mock", (unsigned)ctrl_port);
    int fd = -1;
    void *ko = kopen(url, &fd);
    int fail = 0;
    if (ko != 0) {
        fprintf(stderr,
                "FAIL: ftp_truncated_reply: kopen accepted a truncated (no-CRLF) "
                "control reply instead of failing the handshake\n");
        drain_to_eof(fd);
        close(fd);
        kclose(ko);
        fail = 1;
    }
    int status = 0;
    waitpid(server, &status, 0);
    return fail;
}

/* Watchdog: a kclose hang (the regression this test guards against) would
 * otherwise wedge CI. Fire well above the ~100ms the real cases take. */
static void on_alarm(int sig)
{
    (void)sig;
    static const char m[] = "FAIL: kopen_pipe_status_test timed out (kclose hang?)\n";
    ssize_t n = write(STDERR_FILENO, m, sizeof m - 1);
    (void)n;
    _exit(1);
}

int main(void)
{
    signal(SIGALRM, on_alarm);
    alarm(30);  /* the process-group and HTTP checks below add real I/O time */

    int fails = 0;

    /* Direct-execvp branch (need_shell == 0). */
    fails += check("<false", 1);                                  /* exits 1 */
    fails += check("<no_such_command_kopen_test_4242", 127);      /* exec fails -> _exit(127) */

    /* /bin/sh branch (need_shell == 1, triggered by ';'). Exact statuses, and
     * both output-emptiness polarities so stream-emptiness can't stand in for
     * exit status. */
    fails += check("<echo hi;", 0);        /* non-empty output, exit 0 */
    fails += check("<exit 0;", 0);         /* empty output, exit 0 */
    fails += check("<exit 3;", 3);         /* exact non-zero status propagated */
    fails += check("<kill -9 $$;", ANY_NONZERO);  /* dies by a signal we didn't send */

    /* Empty / whitespace-only command rejected before forking. */
    fails += check_empty_cmd_rejected("<");
    fails += check_empty_cmd_rejected("<   ");

    /* Producer closes its own stdout (reader sees EOF) then lingers: kclose must
     * terminate it and return promptly (rc 0 -- all output was delivered), not
     * block until the sleep ends. The watchdog catches a regression to a hang. */
    fails += check("<echo hi; exec 1>&-; sleep 30", 0);

    /* A backgrounded descendant must not outlive kclose (process-group
     * cleanup), and an oversized/unterminated HTTP header must be rejected
     * outright, not accepted with a truncated body. */
    fails += check_process_group_cleanup();
    fails += check_http_oversized_header();

    /* FTP passive mode: the data connection is opened before RETR, and the
     * host advertised in the PASV reply is ignored in favor of the control
     * peer (FTP-bounce / SSRF guard). */
    fails += check_ftp_pasv_before_retr();

    /* A control reply truncated by an early peer close (status digits, no CRLF)
     * must fail the handshake, not be parsed as a valid reply. */
    fails += check_ftp_truncated_reply();

    alarm(0);
    if (fails) {
        fprintf(stderr, "kopen_pipe_status_test: %d failure(s)\n", fails);
        return 1;
    }
    printf("PASS: kopen_pipe_status_test\n");
    return 0;
}
