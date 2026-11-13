/* Self-check for cb_block_lower_sound (2026-11-13): (1) on random rows it is never above the exact-arithmetic bound
 * computed in long double; (2) it differs from the old margin-based bound by far less than the old 1e-6 margin;
 * (3) report decisions where old and sound disagree. Build: cc -O1 -I engine -I firmware/gateway/main
 * engine/test_sound.c -lm -o /tmp/ts && /tmp/ts */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "certb.h"
#include "model_transfer.h"

/* the same IBP in 80-bit long double (64-bit mantissa): a near-exact reference the sound bound must not exceed */
static long double ld_lse(const long double *v, int n) {
  long double mx = v[0], s = 0;
  for (int i = 1; i < n; i++) if (v[i] > mx) mx = v[i];
  for (int i = 0; i < n; i++) s += expl(v[i] - mx);
  return mx + logl(s);
}
static long double ld_lower(const CertB *c, const double *x, int wide) {
  double lo[CB_MAXD], hi[CB_MAXD];
  cb_dbox(c, x, lo, hi);
  for (int g = 0; g < 4; g++)
    if (wide >> (g & 1) & 1)
      for (int j = c->F + g * 5 + 2; j <= c->F + g * 5 + 3; j++) {
        lo[j] = fmin(lo[j], (log1p(0.0) - c->mu[j]) / c->sd[j]); hi[j] = fmax(hi[j], (log1p(1.0) - c->mu[j]) / c->sd[j]);
      }
  long double lp[CB_MAXM];
  for (int m = 0; m < c->M; m++) {
    long double mu[CB_MAXC], rad[CB_MAXC], mu2[CB_MAXW], rad2[CB_MAXW], d[CB_MAXK];
    for (int i = 0; i < c->D; i++) {
      long double l = fminl(fmaxl(lo[i], c->clo[i]), c->chi[i]), h = fminl(fmaxl(hi[i], c->clo[i]), c->chi[i]);
      l = fmaxl(fminl(l, 1e30L), -1e30L); h = fmaxl(fminl(h, 1e30L), -1e30L);
      long double ri = c->M > 1 ? c->r[m][i] : 1;
      mu[i] = (l + h) / 2 * ri; rad[i] = (h - l) / 2 * fabsl(ri);
    }
    for (int l = 0; l < c->L; l++) {
      for (int o = 0; o < c->rows[l]; o++) {
        long double zm = c->b[l][o], zr = 0; const int8_t *w = c->w[l] + o * c->cols[l];
        for (int i = 0; i < c->cols[l]; i++) { long double wi = (long double)w[i] * c->ws[l]; zm += wi * mu[i]; zr += fabsl(wi) * rad[i]; }
        long double a = fmaxl(zm - zr, 0), b = fmaxl(zm + zr, 0); mu2[o] = (a + b) / 2; rad2[o] = (b - a) / 2;
      }
      for (int o = 0; o < c->rows[l]; o++) { mu[o] = mu2[o]; rad[o] = rad2[o]; }
    }
    for (int k = 0; k < c->K; k++) {
      long double dm = (long double)c->hb[m][k] - c->hb[m][c->BI], dr = 0;
      for (int j = 0; j < c->W; j++) {
        long double dk = (long double)(c->hw[m][j * c->K + k] - c->hw[m][j * c->K + c->BI]) * c->hws;
        dm += dk * mu[j]; dr += fabsl(dk) * rad[j];
      }
      d[k] = k == c->BI ? 0 : dm - dr;
    }
    lp[m] = -ld_lse(d, c->K);
  }
  return -(ld_lse(lp, c->M) - logl((long double)c->M));
}

int main(void) {
  const CertB *c = &CB_MODEL; double x[CB_MAXD]; srand(7);
  double worst = 0; int flips = 0, n = 20000;
  for (int r = 0; r < n; r++) {
    for (int i = 0; i < c->D; i++) x[i] = (rand() / (double)RAND_MAX - 0.5) * 6;
    int wide = rand() % 4;
    double old = cb_block_lower(c, x, wide) + 1e-6, snd = cb_block_lower_sound(c, x, wide);
    long double ref = ld_lower(c, x, wide);
    if (isfinite(ref)) assert(snd <= ref);           /* sound: never above the near-exact bound */
    assert(snd <= old + 1e-9);                        /* the sound bound is never meaningfully looser on the up side */
    if (old - snd > worst) worst = old - snd;
    if ((old - 1e-6 > c->tau) != (snd > c->tau)) flips++;
  }
  printf("test_sound: %d rows, max(unmargined old - sound) = %.3g (old margin 1e-6), decisions changed vs old: %d\n",
         n, worst, flips);
  assert(worst < 1e-8);
  return 0;
}
