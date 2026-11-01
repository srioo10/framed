/* certb.h: champion B (TabM, int8) score, Theorem 1' deletion box and float64 IBP lower bound (PREREG §9 CERT,
 * DEPLOY). Header-only, no stdio or malloc: the same code runs in the Pi engine (certids.c) and on the ESP32-S3.
 *
 * Model: x -> clip to [clo, chi] -> per member m: x * r_m -> [Linear -> ReLU] * depth -> logits h . hw_m + hb_m.
 * Score s = -log(mean_m softmax(z_m)[bi]) (nlog head). Weights are int8 x per-tensor scale; every bound below is
 * computed on exactly these dequantized weights in double, so the certificate is about the deployed model.
 * Block rule (Theorem 1'): block only when the IBP lower bound of s over the deletion box of the observed row at m
 * injected flows exceeds tau; then the flow's clean (pre-injection) row also alarms. */
#ifndef CERTB_H
#define CERTB_H
#include <math.h>
#include <stdint.h>

#define CB_MAXD 32
#define CB_MAXW 64
#define CB_MAXK 16
#define CB_MAXM 8
#define CB_MAXL 3
#define CB_MAXC 64 /* max Linear input width (D or W) */
#define CB_HC 20 /* hostctx.COLS: window (10 s, 120 s) x side (src, dst) x stat (n, fail, newport, newpeer, bytes) */

typedef struct {
  int D, F, BI, K, M, W, L;               /* inputs (F flow + 20 host), benign class, classes, members, width, depth */
  double mu[CB_MAXD], sd[CB_MAXD];        /* Prep: signed log1p, then (v - mu) / sd */
  double clo[CB_MAXD], chi[CB_MAXD];      /* input clip in Prep space (+-1e30 = none) */
  int zero[CB_MAXD];                      /* columns fed as 0 (host-context bytes, dropped in training) */
  double r[CB_MAXM][CB_MAXD];             /* TabM member input scales */
  int rows[CB_MAXL], cols[CB_MAXL];
  int8_t w[CB_MAXL][CB_MAXW * CB_MAXC];   /* body Linear weights, row-major (rows x cols) */
  double ws[CB_MAXL], b[CB_MAXL][CB_MAXW];
  int8_t hw[CB_MAXM][CB_MAXW * CB_MAXK];  /* heads, (w x k) per member */
  double hws, hb[CB_MAXM][CB_MAXK];
  double tau;                             /* block / alarm threshold on s */
  int m_budget;                           /* m injected flows per key per 240 s */
} CertB;

static inline double cb_lse(const double *v, int n) {
  double mx = v[0], s = 0;
  for (int i = 1; i < n; i++) if (v[i] > mx) mx = v[i];
  if (isinf(mx)) return mx;
  for (int i = 0; i < n; i++) s += exp(v[i] - mx);
  return mx + log(s);
}

/* raw flow + host features -> model input (Prep space, zeroed columns applied) */
static inline void cb_prep(const CertB *c, const double *raw, double *x) {
  for (int i = 0; i < c->D; i++) {
    double v = raw[i];
    x[i] = c->zero[i] ? 0 : ((v < 0 ? -1 : 1) * log1p(fabs(v)) - c->mu[i]) / c->sd[i];
  }
}

/* centre / radius through one member; lower bounds of z_i - z_bi into d[K] (lo == hi gives the exact margins) */
static inline void cb_margins(const CertB *c, int m, const double *lo, const double *hi, double *d) {
  double mu[CB_MAXC], rad[CB_MAXC];
  double mu2[CB_MAXW], rad2[CB_MAXW];
  for (int i = 0; i < c->D; i++) {
    double l = fmin(fmax(lo[i], c->clo[i]), c->chi[i]), h = fmin(fmax(hi[i], c->clo[i]), c->chi[i]);
    l = fmax(fmin(l, 1e30), -1e30); h = fmax(fmin(h, 1e30), -1e30);
    double ri = c->M > 1 ? c->r[m][i] : 1;
    mu[i] = (l + h) / 2 * ri; rad[i] = (h - l) / 2 * fabs(ri);
  }
  for (int l = 0; l < c->L; l++) {
    for (int o = 0; o < c->rows[l]; o++) {
      double zm = 0, zr = 0; const int8_t *w = c->w[l] + o * c->cols[l];
      for (int i = 0; i < c->cols[l]; i++) { double wi = w[i] * c->ws[l]; zm += wi * mu[i]; zr += fabs(wi) * rad[i]; }
      zm += c->b[l][o];
      double a = fmax(zm - zr, 0), bb = fmax(zm + zr, 0); /* ReLU is monotone */
      mu2[o] = (a + bb) / 2; rad2[o] = (bb - a) / 2;
    }
    for (int o = 0; o < c->rows[l]; o++) { mu[o] = mu2[o]; rad[o] = rad2[o]; }
  }
  for (int k = 0; k < c->K; k++) {
    double dm = c->hb[m][k] - c->hb[m][c->BI], dr = 0;
    for (int j = 0; j < c->W; j++) {
      double dk = (c->hw[m][j * c->K + k] - c->hw[m][j * c->K + c->BI]) * c->hws;
      dm += dk * mu[j]; dr += fabs(dk) * rad[j];
    }
    d[k] = k == c->BI ? 0 : dm - dr;
  }
}

/* lower bound of s over the box [lo, hi] (s itself when lo == hi): member log P(benign) <= -lse_i lower(z_i - z_bi) */
static inline double cb_score_lower(const CertB *c, const double *lo, const double *hi) {
  double lp[CB_MAXM] = {0}, d[CB_MAXK];
  for (int m = 0; m < c->M; m++) { cb_margins(c, m, lo, hi, d); lp[m] = -cb_lse(d, c->K); }
  double s = -(cb_lse(lp, c->M) - log((double)c->M));
  return isnan(s) ? -INFINITY : s;
}

static inline double cb_score(const CertB *c, const double *x) { return cb_score_lower(c, x, x); }

/* Theorem 1' deletion box of an observed row x (sensitivity.deletion_box, windows <= 120 s so a = b = m) */
static inline void cb_dbox(const CertB *c, const double *x, double *lo, double *hi) {
  const double CB_EPS = 1e-4;
  for (int i = 0; i < c->D; i++) lo[i] = hi[i] = x[i];
  int a = c->m_budget;
  for (int g = 0; g < 4; g++) {
    int o = c->F + g * 5;  /* n, fail, newport, newpeer, bytes */
    double rn = expm1(fmax(x[o] * c->sd[o] + c->mu[o], 0)), n = rint(rn);
    double r = fmax(n - a, 0), dr = fmax(r, 1);
    int empty = r < 1;
    lo[o] = (log1p(r) - c->mu[o]) / c->sd[o];
    for (int s = 1; s <= 3; s++) {
      int j = o + s;
      double cc = expm1(fmax(x[j] * c->sd[j] + c->mu[j], 0)) * fmax(n, 1);
      double slo = empty ? 0 : fmax(cc - a, 0) / dr;
      double shi = empty ? 1 : fmin((s == 1 ? cc : cc + a) / dr, 1);
      lo[j] = (log1p(slo) - c->mu[j]) / c->sd[j]; hi[j] = (log1p(shi) - c->mu[j]) / c->sd[j];
    }
    lo[o + 4] = (log1p(0.0) - c->mu[o + 4]) / c->sd[o + 4]; hi[o + 4] = INFINITY;
  }
  for (int i = c->F; i < c->D; i++) { lo[i] = fmin(lo[i], x[i]) - CB_EPS; hi[i] = fmax(hi[i], x[i]) + CB_EPS; }
}

/* certified block: lower bound over the deletion box > tau (float64, outward margin 1e-6 as fighters.score_lower) */
static inline double cb_block_lower(const CertB *c, const double *x) {
  double lo[CB_MAXD], hi[CB_MAXD];
  cb_dbox(c, x, lo, hi);
  return cb_score_lower(c, lo, hi) - 1e-6;
}
#endif
