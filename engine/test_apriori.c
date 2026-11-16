/* engine/test_apriori.c: build with cc -I engine -I firmware/gateway/main engine/test_apriori.c -lm; exits 0 iff the bound is below the 1e-6 margin. */
/* A priori bound on |computed - exact| of cb_score_lower (plain float64 IBP, margin build) over every box the engine can
 * be given: inputs are clipped (context columns) or are signed log1p of a 64-bit raw value (flow columns). Error model:
 * each float op errs by <= u relative; k-term dot products by gamma_k * sum|terms| (Higham); exp/log/log1p by CB_LIBM_REL. */
#include <math.h>
#include <stdio.h>
#include "model_transfer.h"  /* the only model any live build selects (every sdkconfig: CONFIG_GW_MODEL_TRANSFER) */
#define U 1.1102230246251565e-16
#define LIBM 1e-12
/* raw field limits on the gateway (ids.c Flow): uint16 port, uint8 proto, duration <= ACTIVE_MS, uint32 counters */
static const double RAWMAX[7] = {65535, 255, 120000, 4294967295.0, 4294967295.0, 4294967295.0, 4294967295.0};
static double gam(int k) { return k * U / (1 - k * U); }
int main(void) {
  const CertB *c = &CB_MODEL; double worst = 0, amax_in = 0, dmax = 0;
  for (int m = 0; m < c->M; m++) {
    double A[CB_MAXW + CB_MAXC], E[CB_MAXW + CB_MAXC];  /* |value| bound and endpoint error bound per input */
    for (int i = 0; i < c->D; i++) {
      /* magnitude: clipped range for context columns, field width for flow columns */
      double raw = i < c->F ? RAWMAX[i] : 16777216.0;  /* context counts < 2^24 (Lemma (iv)); shares <= 1 */
      double base = c->zero[i] ? 0 : (log1p(raw) + fabs(c->mu[i])) / c->sd[i];  /* rounding base of (log1p(v) - mu) / sd */
      double a = c->zero[i] ? 0 : (c->clo[i] > -1e29 ? fmax(fabs(c->clo[i]), fabs(c->chi[i])) : base);
      double ri = c->M > 1 ? fabs(c->r[m][i]) : 1;
      /* endpoint error, counted per op (cb_dbox): v = x*sd + mu (2 ops), expm1 (LIBM), rint or the share's mul/sub/div
       * (<= 4 ops), log1p (LIBM), - mu, / sd (2 ops); then the member scale * r (1 op). Each op errs by u relative to
       * at most base; each libm call by LIBM; error entering a libm call is passed on with factor <= 1 in log space.
       * Total <= base * (2 LIBM + 9u); we use 4 LIBM + 16u, twice that. */
      A[i] = a * ri; E[i] = base * ri * (4 * LIBM + 16 * U); if (A[i] > amax_in) amax_in = A[i];
    }
    int n = c->D;
    for (int l = 0; l < c->L; l++) {
      double A2[CB_MAXW], E2[CB_MAXW];
      for (int o = 0; o < c->rows[l]; o++) {
        double mag = fabs(c->b[l][o]), prop = 0; const int8_t *w = c->w[l] + o * c->cols[l];
        for (int i = 0; i < c->cols[l]; i++) { double aw = fabs(w[i] * c->ws[l]); mag += aw * A[i]; prop += aw * E[i]; }
        /* lo/hi = zm -/+ zr: inherited error from mu and rad (each <= E), own rounding of zm, zr, the w*ws product and the +/- */
        A2[o] = mag * (1 + 1e-9);
        E2[o] = 2 * prop + 2 * gam(c->cols[l] + 2) * mag + 4 * U * mag;  /* ReLU is 1-Lipschitz */
      }
      for (int o = 0; o < c->rows[l]; o++) { A[o] = A2[o]; E[o] = E2[o]; }
      n = c->rows[l];
    }
    for (int k = 0; k < c->K; k++) {
      if (k == c->BI) continue;
      double hb = fabs(c->hb[m][k] - c->hb[m][c->BI]), mag = hb * (1 + 2 * U), prop = 0;
      for (int j = 0; j < c->W; j++) { double ak = fabs((c->hw[m][j * c->K + k] - c->hw[m][j * c->K + c->BI]) * c->hws); mag += ak * A[j]; prop += ak * E[j]; }
      double ed = 2 * prop + 2 * gam(c->W + 2) * mag + 4 * U * mag;
      /* lp_m = -lse(d): lse is 1-Lipschitz in max-norm; own error: max subtraction, K exps (LIBM), K-sum, one log, one add */
      double el = ed + (LIBM + gam(c->K + 1)) * 2 + LIBM * log(c->K) + 2 * U * mag;
      if (el > worst) worst = el; if (mag > dmax) dmax = mag;
    }
  }
  /* s = -(lse(lp) - log M): again 1-Lipschitz plus own rounding; |lp| bounded by worst-case margin magnitude, absorbed above */
  double es = worst + (LIBM + gam(c->M + 1)) * 2 + LIBM * log(c->M) + 4 * U * (dmax + log(c->K) + log(c->M));
  printf("max |margin| %.3g; max |input| %.3g;", dmax, amax_in); printf(" a priori bound on |s_lower computed - exact|: %.3g (margin 1e-6 %s)\n", es, es < 1e-6 ? "covers it" : "does NOT cover it");
  return es < 1e-6 ? 0 : 1;
}
