/* test/err_fgets_eof_test.cpp -- err_fgets must report end-of-file as
 * "Unexpected end of file", not through a stale strerror(errno).
 *
 * fgets() returns NULL for both a read error AND clean EOF, and errno is not set
 * on EOF -- so the pre-fix err_fgets aborts with whatever errno was left over,
 * printing "[fgets] Success" (glibc) or "[fgets] Undefined error: 0" (macOS) and
 * pointing the operator at the wrong failure. The sibling err_fread_noeof already
 * distinguishes ferror() from EOF; this pins that err_fgets does the same.
 *
 * err_fgets aborts via _err_fatal_simple (exit), so the check runs in a forked
 * child whose stderr is captured through a pipe.
 *
 * Build: links libbwa for err_fgets / _err_fatal_simple (see Makefile).
 */
#include "utils.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>

int main(void)
{
    int pipefd[2];
    if (pipe(pipefd) != 0) { perror("pipe"); return 1; }

    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }
    if (pid == 0) {                       /* child: trigger the EOF abort */
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[0]); close(pipefd[1]);
        FILE *fp = tmpfile();             /* empty file -> fgets sees EOF at once */
        if (!fp) _exit(3);
        char buf[64];
        err_fgets(buf, sizeof buf, fp);   /* NULL at EOF -> _err_fatal_simple -> exit */
        _exit(0);                         /* not reached */
    }

    close(pipefd[1]);
    char out[512] = {0};
    size_t n = 0; ssize_t r;
    while (n < sizeof out - 1 && (r = read(pipefd[0], out + n, sizeof out - 1 - n)) > 0)
        n += (size_t)r;
    close(pipefd[0]);
    int st = 0; waitpid(pid, &st, 0);

    /* tmpfile() unavailable in a restricted sandbox is a setup failure, not a
     * diagnostic mismatch -- report it as SKIP rather than a spurious FAIL. */
    if (WIFEXITED(st) && WEXITSTATUS(st) == 3) {
        printf("SKIP: tmpfile() unavailable in this environment\n");
        return 0;
    }

    /* The positive check is the whole oracle: if err_fgets still reports EOF
     * through a stale errno, this string is absent and the test fails. */
    if (strstr(out, "Unexpected end of file") == NULL) {
        fprintf(stderr, "FAIL: err_fgets EOF diagnostic was: %s\n", out);
        return 1;
    }
    printf("PASS: err_fgets reports EOF as 'Unexpected end of file'\n");
    return 0;
}
