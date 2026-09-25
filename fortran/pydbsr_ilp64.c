/* DSYEVD with 64-bit integers, loaded at run time (added by pydbsr).
 *
 * LAPACK with 32-bit integers cannot run DSYEVD for n > ~32000: its workspace
 * (1 + 6n + 2n^2) overflows Integer(4).  dbsr_hd3 calls pydbsr_dsyevd64() for
 * such matrices; it dlopen()s an ILP64 LAPACK given by $PYDBSR_ILP64_LAPACK
 * (e.g. libscipy_openblas64_.so of the scipy-openblas64 wheel, or conda's
 * libopenblas64_.so) and calls its DSYEVD.
 *
 * Returns the LAPACK info (0 = success), or -9999 if no ILP64 library is
 * available (the caller then uses another solver), -9998 if the workspace
 * cannot be allocated.
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef void (*dsyevd64_t)(const char *jobz, const char *uplo, const int64_t *n, double *a,
                           const int64_t *lda, double *w, double *work, const int64_t *lwork,
                           int64_t *iwork, const int64_t *liwork, int64_t *info,
                           size_t ljobz, size_t luplo);

static dsyevd64_t find_dsyevd64(void)
{
    const char *path = getenv("PYDBSR_ILP64_LAPACK");
    if (path == NULL || *path == '\0')
        return NULL;
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (h == NULL) {
        fprintf(stderr, "pydbsr_dsyevd64: cannot load %s: %s\n", path, dlerror());
        return NULL;
    }
    const char *names[] = {"scipy_dsyevd_64_", "dsyevd_64_", "dsyevd_64", NULL};
    for (int i = 0; names[i]; i++) {
        void *f = dlsym(h, names[i]);
        if (f)
            return (dsyevd64_t)f;
    }
    fprintf(stderr, "pydbsr_dsyevd64: no ILP64 dsyevd in %s\n", path);
    return NULL;
}

int pydbsr_dsyevd64(int ijobz, int iuplo, int n32, double *a, double *w)
{
    char jobz = (char)ijobz, uplo = (char)iuplo;     /* passed as ichar() from Fortran */
    dsyevd64_t f = find_dsyevd64();
    if (f == NULL)
        return -9999;
    int64_t n = n32, lda = n32, lwork = -1, liwork = -1, info = 0, iwq = 0;
    double wq = 0.0;
    f(&jobz, &uplo, &n, a, &lda, w, &wq, &lwork, &iwq, &liwork, &info, 1, 1);
    if (info != 0)
        return (int)info;
    lwork = (int64_t)wq;
    liwork = iwq;
    double *work = malloc((size_t)lwork * sizeof(double));
    int64_t *iwork = malloc((size_t)liwork * sizeof(int64_t));
    if (work == NULL || iwork == NULL) {
        free(work);
        free(iwork);
        return -9998;
    }
    f(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, &info, 1, 1);
    free(work);
    free(iwork);
    return (int)info;
}
