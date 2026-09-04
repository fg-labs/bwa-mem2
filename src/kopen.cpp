/* The MIT License

   Copyright (c) by Attractive Chaos <attractivechaos@aol.co.uk>

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

   Modified Copyright (C) 2020 Intel Corporation, Heng Li.
   Contacts: Vasimuddin Md <vasimuddin.md@intel.com>; Sanchit Misra <sanchit.misra@intel.com>;
   Heng Li <hli@jimmy.harvard.edu>
*/

#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <ctype.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>
#include <time.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#include "utils.h"  /* xassert */

#ifdef USE_MALLOC_WRAPPERS
#  include "malloc_wrap.h"
#endif

static int socket_wait(int fd, int is_read)
{
	fd_set fds, *fdr = 0, *fdw = 0;
	struct timeval tv;
	int ret;
	tv.tv_sec = 5; tv.tv_usec = 0; // 5 seconds time out
	FD_ZERO(&fds);
	FD_SET(fd, &fds);
	if (is_read) fdr = &fds;
	else fdw = &fds;
	ret = select(fd+1, fdr, fdw, 0, &tv);
	if (ret == -1) perror("select");
	return ret;
}

static int socket_connect(const char *host, const char *port)
{
#define __err_connect(func) do { perror(func); freeaddrinfo(res); return -1; } while (0)
#define __err_connect2(func) do { perror(func); freeaddrinfo(res); close(fd); return -1; } while (0)

	int on = 1, fd, gai;
	struct linger lng = { 0, 0 };
	struct addrinfo hints, *res = 0;
	memset(&hints, 0, sizeof(struct addrinfo));
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	// getaddrinfo returns its own error code and does NOT set errno, so report
	// it with gai_strerror rather than perror (which would print a stale errno).
	if ((gai = getaddrinfo(host, port, &hints, &res)) != 0) {
		fprintf(stderr, "ERROR: getaddrinfo: %s\n", gai_strerror(gai));
		if (res) freeaddrinfo(res);
		return -1;
	}
	if ((fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol)) == -1) __err_connect("socket");
	if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) == -1) __err_connect2("setsockopt");
	if (setsockopt(fd, SOL_SOCKET, SO_LINGER, &lng, sizeof(lng)) == -1) __err_connect2("setsockopt");
	if (connect(fd, res->ai_addr, res->ai_addrlen) != 0) __err_connect2("connect");
	freeaddrinfo(res);
	return fd;
#undef __err_connect
#undef __err_connect2
}

static int write_bytes(int fd, const char *buf, size_t len)
{
	ssize_t bytes;
	// Advance buf as well as len on every successful write: a short write
	// (sockets can and do short-write under back-pressure) must resume from the
	// unsent tail, not re-send the first `len` bytes and drop the rest.
	while (len > 0) {
		bytes = write(fd, buf, len);
		if (bytes >= 0) {
			len -= bytes;
			buf += bytes;
		} else if (errno == EAGAIN || errno == EWOULDBLOCK) {
			// Wait for writability instead of busy-spinning. Treat a timeout
			// (socket_wait == 0) as fatal too, matching kftp_get_response's
			// read-side check, so a wedged socket cannot retry forever.
			if (socket_wait(fd, 0) <= 0) return -1;
		} else if (errno != EINTR) {
			return -1;
		}
	}
	return 0;
}

static int http_open(const char *fn)
{
	char *p, *proxy, *q, *http_host = 0, *host = 0, *port = 0, *path = 0, *buf = 0;
	int fd = -1, ret, l, n;
	int hdr_end = 0;  // set once the terminating "\r\n\r\n" is actually seen
	ssize_t bytes = 0, bufsz = 0x10000;

	/* parse URL; adapted from khttp_parse_url() in knetfile.c */
	if (strstr(fn, "http://") != fn) return -1;
	// set ->http_host
	for (p = (char*)fn + 7; *p && *p != '/'; ++p);
	l = p - fn - 7;
	http_host = (char*) calloc(l + 1, 1);
	xassert(http_host != NULL, "http_open: out of memory");
	strncpy(http_host, fn + 7, l);
	http_host[l] = 0;
	for (q = http_host; *q && *q != ':'; ++q);
	if (*q == ':') *q++ = 0;
	// get http_proxy
	proxy = getenv("http_proxy");
	// set host, port and path
	if (proxy == 0) {
		host = strdup(http_host); // when there is no proxy, server name is identical to http_host name.
		port = strdup(*q? q : "80");
		path = strdup(*p? p : "/");
	} else {
		host = (strstr(proxy, "http://") == proxy)? strdup(proxy + 7) : strdup(proxy);
		for (q = host; q && *q && *q != ':'; ++q);
		if (q && *q == ':') *q++ = 0;
		port = strdup((q && *q)? q : "80");
		path = strdup(fn);
	}
	// An allocation failure here is OOM, not a bad URL: abort with a message
	// (the repo's allocation-failure idiom) rather than return a misleading
	// "failed to open file" to the caller.
	xassert(host != NULL && port != NULL && path != NULL, "http_open: out of memory");

	/* connect; adapted from khttp_connect() in knetfile.c */
	l = 0;
	fd = socket_connect(host, port);
	if (fd < 0) goto out;  // report the connect failure directly, don't fall through to write()
	buf = (char*) calloc(bufsz, 1); // FIXME: I am lazy... But in principle, 64KB should be large enough.
	xassert(buf != NULL, "http_open: out of memory");
	// snprintf returns the length it WOULD have written; guard it against bufsz
	// so an over-long path/host cannot make write_bytes read past the buffer.
	n = snprintf(buf, bufsz, "GET %s HTTP/1.0\r\nHost: %s\r\n\r\n", path, http_host);
	if (n < 0 || n >= bufsz) { close(fd); fd = -1; goto out; }
	if (write_bytes(fd, buf, n) != 0) {
		close(fd);
		fd = -1;
		goto out;
	}
	l = 0;
 retry:
	// Bound the loop at bufsz-1 so the terminating buf[l]=0 below stays inside
	// the allocation even when the server sends a header with no blank line.
	while (l < bufsz - 1 && (bytes = read(fd, buf + l, 1)) > 0) { // read HTTP header; FIXME: bad efficiency
		if (buf[l] == '\n' && l >= 3)
			if (strncmp(buf + l - 3, "\r\n\r\n", 4) == 0) { hdr_end = 1; break; }
		++l;
	}
	if (bytes < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) goto retry;

	buf[l] = 0;
	// Require the full status line, the HTTP/ signature, AND the terminator
	// actually being seen (not just the buffer filling up) before parsing the
	// code at buf+8. Without hdr_end, an oversized header that never sends a
	// blank line exits the loop only because l reached bufsz-1 -- bytes > 0 and
	// the prefix can still look like "HTTP/...", so this would otherwise be
	// accepted with the header's unread tail then delivered to the caller as
	// body data instead of the failure this is meant to catch.
	if (bytes < 0 || !hdr_end || l < 14 || strncmp(buf, "HTTP/", 5) != 0) { // prematured / non-HTTP / oversized header
		close(fd);
		fd = -1;
		goto out;
	}
	ret = strtol(buf + 8, &p, 0); // HTTP return code
	if (ret != 200) {
		close(fd);
		fd = -1;
	}
 out:
	free(buf); free(http_host); free(host); free(port); free(path);
	return fd;
}

typedef struct {
	int max_response, ctrl_fd;
	char *response;
} ftpaux_t;

static int kftp_get_response(ftpaux_t *aux)
{
	unsigned char c;
	int n = 0;
	char *p;
	ssize_t r;
	// Read a byte at a time until a complete reply line. Treating -1 as truthy
	// (the old `while (read(...))`) spins forever on any read error, doubling
	// response each pass until the allocator fails. EINTR is retried (a signal
	// is not a protocol error); any other read error, or EOF before a complete
	// reply line, is a hard failure so a truncated response is never parsed.
	// The wait is inside the loop so EVERY read is bounded, including the one
	// that follows a continuation line (n = 0; continue) below, which used to
	// re-enter a blocking read() with no timeout.
	while (1) {
		if (socket_wait(aux->ctrl_fd, 1) <= 0) return n? -1 : 0;
		r = read(aux->ctrl_fd, &c, 1);
		if (r < 0 && errno == EINTR) continue;  // a signal is not a protocol error; the write side retries EINTR too
		if (r <= 0) return -1;                  // read error, or EOF before a complete reply line: fail, don't parse a truncated response
		if (n >= aux->max_response) {
			char *tmp;
			int next = aux->max_response? aux->max_response<<1 : 256;
			// realloc into a temporary so the original buffer isn't leaked (and
			// aux->response left dangling) when realloc fails.
			tmp = (char*) realloc(aux->response, next);
			if (tmp == NULL) return -1;  // aux->response still valid; freed by ftp_open cleanup
			aux->response = tmp;
			aux->max_response = next;
		}
		aux->response[n++] = c;
		if (c == '\n') {
			if (n >= 4 && isdigit(aux->response[0]) && isdigit(aux->response[1]) && isdigit(aux->response[2])
				&& aux->response[3] != '-') break;
			n = 0;
			continue;
		}
	}
	// The loop exits only via the valid-reply break above (r > 0), since every
	// read error and EOF returns -1 inline; n < 2 guards a stray short line.
	if (n < 2) return -1;
	aux->response[n-2] = 0;
	return strtol(aux->response, &p, 0);
}

static int kftp_send_cmd(ftpaux_t *aux, const char *cmd, int is_get)
{
	if (socket_wait(aux->ctrl_fd, 0) <= 0) return -1; // socket is not ready for writing
	if (write_bytes(aux->ctrl_fd, cmd, strlen(cmd)) != 0) return -1;
	return is_get? kftp_get_response(aux) : 0;
}

static int ftp_open(const char *fn)
{
	char *p, *host = 0, *port = 0, *retr = 0;
	char host2[80], port2[10];
	int v[6], l, fd = -1, ret, pasv_port, i, np;
	ftpaux_t aux;

	// Initialize aux BEFORE any `goto ftp_open_end`: the cleanup block reads
	// aux.ctrl_fd and frees aux.response, and the allocation-failure gotos below
	// can be reached before the control connection is made.
	memset(&aux, 0, sizeof(ftpaux_t));
	aux.ctrl_fd = -1;  // so ftp_open_end can close it unconditionally when >= 0

	/* parse URL: ftp://host[:port]/path (RFC 1738). The authority is everything
	 * between "ftp://" and the first '/'; an explicit ":port" overrides the
	 * default 21. Parsing the port also lets the tests point ftp_open at an
	 * unprivileged loopback mock instead of requiring a real port-21 server. */
	if (strstr(fn, "ftp://") != fn) return -1;
	for (p = (char*)fn + 6; *p && *p != '/'; ++p);
	if (*p != '/') return -1;
	{
		char *authority = (char*)fn + 6, *colon = 0, *q;
		for (q = authority; q < p; ++q) if (*q == ':') { colon = q; break; }
		if (colon) {
			l = colon - authority;
			size_t plen = p - (colon + 1);
			port = (char*) calloc(plen + 1, 1);
			// OOM here is not a bad URL: abort with a message (the repo idiom).
			xassert(port != NULL, "ftp_open: out of memory");
			memcpy(port, colon + 1, plen);
		} else {
			l = p - authority;
			port = strdup("21");
			xassert(port != NULL, "ftp_open: out of memory");
		}
		host = (char*) calloc(l + 1, 1);
		xassert(host != NULL, "ftp_open: out of memory");
		memcpy(host, authority, l);
	}
	retr = (char*) calloc(strlen(p) + 8, 1);
	xassert(retr != NULL, "ftp_open: out of memory");
	snprintf(retr, strlen(p) + 8, "RETR %s\r\n", p);

	/* connect to ctrl */
	aux.ctrl_fd = socket_connect(host, port);
	if (aux.ctrl_fd == -1) goto ftp_open_end; /* fail to connect ctrl */

	/* connect to the data stream — validate every handshake reply */
	if (kftp_get_response(&aux) != 220) goto ftp_open_end;              /* greeting */
	// USER: RFC 959 permits a server to log the user in directly (230) without a
	// password, or to request one (331). Accept both; only send PASS after 331.
	ret = kftp_send_cmd(&aux, "USER anonymous\r\n", 1);
	if (ret == 331) {
		if (kftp_send_cmd(&aux, "PASS kopen@\r\n", 1) != 230) goto ftp_open_end;
	} else if (ret != 230) {
		goto ftp_open_end;
	}
	if (kftp_send_cmd(&aux, "TYPE I\r\n", 1) != 200) goto ftp_open_end;
	if (kftp_send_cmd(&aux, "PASV\r\n", 1) != 227) goto ftp_open_end;
	if (aux.response == NULL) goto ftp_open_end;
	for (p = aux.response; *p && *p != '('; ++p);
	if (*p != '(') goto ftp_open_end;
	++p;
	// Require all six fields AND each in 0..255 before trusting the PASV reply;
	// a malformed reply otherwise reads uninitialized v[] and can overflow the
	// fixed port2 buffer below.
	for (i = 0; i < 6; ++i) v[i] = 0;
	if (sscanf(p, "%d,%d,%d,%d,%d,%d", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) != 6) goto ftp_open_end;
	for (i = 0; i < 6; ++i) if (v[i] < 0 || v[i] > 255) goto ftp_open_end;
	pasv_port = (v[4]<<8&0xff00) + v[5];
	// Do NOT trust the host advertised in the PASV reply (FTP bounce / SSRF,
	// CWE-918): a hostile or MITM'd server can name an arbitrary address (v[0..3])
	// there to make us open a data connection to an internal host we would never
	// reach otherwise. Reuse the control connection's own peer address for the
	// data connection and take only the port from the PASV reply -- the same
	// mitigation curl applies with --ftp-skip-pasv-ip.
	{
		struct sockaddr_storage peer;
		socklen_t peerlen = sizeof(peer);
		if (getpeername(aux.ctrl_fd, (struct sockaddr*)&peer, &peerlen) != 0) goto ftp_open_end;
		if (getnameinfo((struct sockaddr*)&peer, peerlen, host2, sizeof(host2),
						NULL, 0, NI_NUMERICHOST) != 0) goto ftp_open_end;
	}
	np = snprintf(port2, sizeof(port2), "%d", pasv_port);
	if (np < 0 || (size_t)np >= sizeof(port2)) goto ftp_open_end;
	// Open the PASV data connection BEFORE sending RETR. RFC 959's passive-mode
	// handshake expects the client to be connecting (or already connected) on
	// the announced data port before it asks the server to start sending; some
	// servers reject an early RETR with 425 ("can't open data connection")
	// when the client hasn't connected yet.
	fd = socket_connect(host2, port2);
	if (fd == -1) goto ftp_open_end;
	if (kftp_send_cmd(&aux, retr, 0) != 0) {
		close(fd);
		fd = -1;
		goto ftp_open_end;
	}
	ret = kftp_get_response(&aux);
	if (ret != 150) {
		close(fd);
		fd = -1;
	}

ftp_open_end:
	if (aux.ctrl_fd >= 0) close(aux.ctrl_fd);
	free(host); free(port); free(retr); free(aux.response);
	return fd;
}

static char **cmd2argv(const char *cmd)
{
	int i, beg, end, argc;
	char **argv, *str;
	end = strlen(cmd);
	for (i = end - 1; i >= 0; --i)
		if (!isspace(cmd[i])) break;
	end = i + 1;
	for (beg = 0; beg < end; ++beg)
		if (!isspace(cmd[beg])) break;
	if (beg == end) return 0;  // empty / whitespace-only command (not OOM)
	for (i = beg + 1, argc = 0; i < end; ++i)
		if (isspace(cmd[i]) && !isspace(cmd[i-1]))
			++argc;
	argv = (char**)calloc(argc + 2, sizeof(void*));
	xassert(argv != NULL, "cmd2argv: out of memory");
	argv[0] = str = (char*)calloc(end - beg + 1, 1);
	xassert(str != NULL, "cmd2argv: out of memory");
	strncpy(argv[0], cmd + beg, end - beg);
	for (i = argc = 1; i < end - beg; ++i)
		if (isspace(str[i])) str[i] = 0;
		else if (str[i] && str[i-1] == 0) argv[argc++] = &str[i];
	return argv;
}

#define KO_STDIN    1
#define KO_FILE     2
#define KO_PIPE     3
#define KO_HTTP     4
#define KO_FTP      5

typedef struct {
	int type, fd;
	pid_t pid;
} koaux_t;

void *kopen(const char *fn, int *_fd)
{
	koaux_t *aux = 0;
	*_fd = -1;
	if (strstr(fn, "http://") == fn) {
		int hfd = http_open(fn);
		// http_open returns -1 on any failure; return NULL like the plain-file
		// path so the caller's `ko == 0` check fires (it never inspects fd).
		if (hfd < 0) return 0;
		aux = (koaux_t*) calloc(1, sizeof(koaux_t));
		xassert(aux != NULL, "kopen: calloc failed");
		aux->type = KO_HTTP;
		aux->fd = hfd;
	} else if (strstr(fn, "ftp://") == fn) {
		int ffd = ftp_open(fn);
		if (ffd < 0) return 0;
		aux = (koaux_t*) calloc(1, sizeof(koaux_t));
		xassert(aux != NULL, "kopen: calloc failed");
		aux->type = KO_FTP;
		aux->fd = ffd;
	} else if (strcmp(fn, "-") == 0) {
		aux = (koaux_t*) calloc(1, sizeof(koaux_t));
		xassert(aux != NULL, "kopen: calloc failed");
		aux->type = KO_STDIN;
		aux->fd = STDIN_FILENO;
	} else {
		const char *p, *q;
		for (p = fn; *p; ++p)
			if (!isspace(*p)) break;
		if (*p == '<') { // pipe open
			int need_shell, pfd[2];
			pid_t pid;
			char **argv = 0;
			// a simple check to see if we need to invoke a shell; not always working
			for (q = p + 1; *q; ++q)
				if (ispunct(*q) && *q != '.' && *q != '_' && *q != '-' && *q != ':')
					break;
			need_shell = (*q != 0);
			// Build the argument vector in the PARENT, before forking: cmd2argv
			// allocates, and no allocation may happen in the child of a fork
			// between fork and exec here. It also lets us reject a `<`-only /
			// whitespace-only command (cmd2argv returns NULL) up front instead
			// of dereferencing argv[0] in the child.
			if (!need_shell) {
				argv = cmd2argv(p + 1);
				if (argv == NULL) return 0;
			}
			if (pipe(pfd) != 0) { if (argv) { free(argv[0]); free(argv); } return 0; }
			pid = fork();
			if (pid == -1) { /* fork() error */
				close(pfd[0]); close(pfd[1]);
				if (argv) { free(argv[0]); free(argv); }
				return 0;
			}
			if (pid == 0) { /* the child process: only exec + _exit, no heap work */
				close(pfd[0]);
				// Put the producer in its own process group (led by itself) so
				// kclose can signal the whole group, not just this direct
				// child. A `<cmd` producer can background a descendant and
				// exit itself (e.g. `<cmd &`); that descendant inherits this
				// group and would otherwise survive kclose entirely, since
				// kclose never learns its pid. setpgid (not setsid) keeps the
				// group in the parent's session, but it is NOT the foreground
				// process group, so terminal-generated SIGINT/SIGQUIT do not
				// reach the producer: a silent producer (e.g. `<sleep 30`) can
				// outlive a Ctrl-C that kills the parent before kclose runs.
				// A producer that keeps writing still dies via EPIPE/SIGPIPE.
				setpgid(0, 0);
				dup2(pfd[1], STDOUT_FILENO);
				close(pfd[1]);
				if (!need_shell) execvp(argv[0], argv);
				// execl is variadic, so the trailing sentinel must be a real
				// pointer type: C++'s NULL can be an integer constant, which a
				// variadic callee reads back as a pointer of undefined
				// provenance on the stack rather than a null pointer.
				else execl("/bin/sh", "sh", "-c", p + 1, (char *)NULL);
				// exec failed: _exit (not exit) so the parent's atexit handlers
				// and stdio buffers are not run/flushed from the child.
				_exit(127);
			} else { /* parent process */
				close(pfd[1]);
				if (argv) { free(argv[0]); free(argv); }  // copied into the child's exec image
				// Mirror the child's setpgid(0,0) here: whichever of the two
				// runs first, the group exists before either side can act on
				// it, closing the fork/exec race where a signal sent (or a
				// waitpid done) before the child's own call would otherwise
				// see the old (parent's) group. Failure here (e.g. ESRCH if
				// the child already exited) is harmless -- the child's own
				// call already established the group in that case.
				setpgid(pid, pid);
				aux = (koaux_t*) calloc(1, sizeof(koaux_t));
				xassert(aux != NULL, "kopen: calloc failed");
				aux->type = KO_PIPE;
				aux->fd = pfd[0];
				aux->pid = pid;
			}
		} else {
			*_fd = open(fn, O_RDONLY);
			if (*_fd >= 0) {
				aux = (koaux_t*) calloc(1, sizeof(koaux_t));
				xassert(aux != NULL, "kopen: calloc failed");
				aux->type = KO_FILE;
				aux->fd = *_fd;
			}
		}
	}
	if (aux) *_fd = aux->fd;
	return aux;
}

/* Poll for `pid` to EXIT (up to grace_us microseconds, 10 ms steps, retrying
 * EINTR) WITHOUT reaping it. waitid(..., WNOWAIT) observes the exit but leaves
 * the zombie waitable, so the group leader stays a member of the producer's
 * process group and its PGID cannot be reused before kclose's killpg cleanup
 * runs (POSIX: a group lives while any member, zombie included, remains). The
 * caller reaps the leader with waitpid only after that cleanup. Returns the pid
 * once it has exited (still reapable), 0 if still running when the grace
 * expires, or -1 on error. A short poll rather than a blocking wait so kclose
 * can bound how long it waits on a producer that stopped writing but not exited. */
static pid_t observe_exit_within(pid_t pid, int grace_us)
{
	siginfo_t si;
	int waited = 0;
	for (;;) {
		int r;
		si.si_pid = 0;  // WNOHANG leaves si_pid untouched when no child is ready
		while ((r = waitid(P_PID, pid, &si, WEXITED | WNOWAIT | WNOHANG)) == -1 && errno == EINTR)
			;
		if (r == -1) return -1;
		if (si.si_pid == pid) return pid;   // exited; left reapable by WNOWAIT
		if (waited >= grace_us) return 0;   // still running
		struct timespec ts = {0, 10 * 1000 * 1000};  // 10 ms
		nanosleep(&ts, NULL);
		waited += 10 * 1000;
	}
}

int kclose(void *a)
{
	koaux_t *aux = (koaux_t*)a;
	int rc = 0;
	if (aux->type == KO_PIPE) {
		int status, killed = 0;
		pid_t pid;
		// Reap the pipe producer and surface a failure (a non-zero exit, or a
		// crash by signal), so a `<cmd` that dies mid-stream or never execs
		// cannot be seen by the caller as a clean EOF and let the run exit 0.
		//
		// This must stay bounded: the caller has closed the read end, but a
		// producer can close its own stdout (the reader sees EOF) and keep
		// running -- it never writes again, so it never takes EPIPE, and a plain
		// blocking waitpid would hang shutdown. The old code (WNOHANG + one
		// SIGTERM) never blocked; preserve that no-hang guarantee. So: poll-reap
		// briefly (long enough for a producer that is merely finishing to exit on
		// its own, so its real status is observed), then force-terminate --
		// SIGTERM, then SIGKILL if it ignores SIGTERM -- so shutdown is always
		// bounded.
		const int grace_us = 100 * 1000;  // 100 ms
		// Observe the producer's exit WITHOUT reaping it (see observe_exit_within):
		// the leader must stay an unreaped zombie so its process group -- and thus
		// its PGID -- survives until the killpg cleanup below. Reaping it here (as
		// a plain waitpid would) could empty the group and free the PGID for reuse
		// before killpg runs, so killpg(aux->pid, ...) might then signal an
		// unrelated, same-UID group that happened to inherit the number.
		pid = observe_exit_within(aux->pid, grace_us);
		if (pid == 0) {
			kill(aux->pid, SIGTERM);
			killed = 1;
			pid = observe_exit_within(aux->pid, grace_us);
			if (pid == 0) {  // ignored SIGTERM: force-kill (uncatchable), then observe
				kill(aux->pid, SIGKILL);
				siginfo_t si;
				si.si_pid = 0;
				while (waitid(P_PID, aux->pid, &si, WEXITED | WNOWAIT) == -1 && errno == EINTR)
					;
				pid = (si.si_pid == aux->pid) ? aux->pid : -1;
			}
		}
		// Terminate any surviving members of the producer's process group (see
		// the setpgid(0,0)/setpgid(pid,pid) pair at fork time). aux->pid has
		// exited but is NOT yet reaped -- it remains a zombie member of the
		// group, holding the PGID -- so killpg here cannot race with reuse. A
		// producer can background a descendant and exit itself before we ever
		// escalate to it directly -- e.g. `<cmd &`, where the shell exits almost
		// immediately with status 0 while `cmd` keeps running, reparented away
		// from us the moment the shell is gone. killpg reaches it via the shared
		// group regardless of who its new parent is. Note the zombie leader is
		// still a group member here, so killpg cannot report ESRCH and its return
		// value says nothing about whether any descendant is actually alive.
		// Escalate unconditionally without an extra grace sleep: the 100 ms
		// observe window above already served as the grace period, so gating on
		// killpg's (always-zero) return and then sleeping 50 ms would just pay a
		// fixed cost on every close, including the common clean-exit case.
		killpg(aux->pid, SIGTERM);
		killpg(aux->pid, SIGKILL);
		// The group is cleaned up; now reap the zombie leader to collect its real
		// status and release the PGID. It has already exited, so this cannot hang.
		if (pid == aux->pid) {
			while (waitpid(aux->pid, &status, 0) == -1 && errno == EINTR)
				;
		}
		if (pid != aux->pid) rc = 1;                        // observe/reap failed (e.g. ECHILD)
		else if (WIFEXITED(status)) {
			if (WEXITSTATUS(status) != 0) rc = WEXITSTATUS(status);
		} else if (WIFSIGNALED(status) && !killed) {
			// Died by a signal we did not send -> a real producer crash (e.g. a
			// decompressor SIGSEGV on truncated input). A producer we terminated
			// ourselves for lingering already delivered all its output, so that
			// is not a data failure and rc stays 0.
			rc = 1;
		}
	}
	free(aux);
	return rc;
}

#ifdef _KO_MAIN
#define BUF_SIZE 0x10000
int main(int argc, char *argv[])
{
	void *x;
	int l, fd;
	unsigned char buf[BUF_SIZE];
	FILE *fp;
	if (argc == 1) {
		fprintf(stderr, "Usage: kopen <file>\n");
		return 1;
	}
	x = kopen(argv[1], &fd);
	fp = fdopen(fd, "r");
	if (fp == 0) {
		fprintf(stderr, "ERROR: fail to open the input\n");
		return 1;
	}
	do {
		if ((l = fread(buf, 1, BUF_SIZE, fp)) != 0)
			fwrite(buf, 1, l, stdout);
	} while (l == BUF_SIZE);
	fclose(fp);
	kclose(x);
	return 0;
}
#endif
