/* certids: streaming certified-block IDS (PREREG §9 DEPLOY). Host context as engine/onepass.c (bounded memory,
 * 1 s / 10 s buckets for the 10 s / 120 s windows, per-host LRU novelty tables), champion B scored by certb.h.
 *
 * Input  (stdin, CSV, time ordered): t_ms,src,dst,dst_port,out_pkts,in_bytes,f_1,...,f_F   (f_* in model order)
 * Output (stdout, per flow):         score,alarm,lower,block
 *   alarm = score > tau; block = lower > tau, lower = IBP lower bound of the score over the Theorem 1' deletion box
 *   at m injected flows (so a blocked flow's clean row alarms too). --on-block 'cmd %s' runs once per source host per
 *   --ttl seconds of stream time (e.g. an nftables add-element with a timeout).
 * --prepped: stdin rows are D Prep-space values (parity test against Python); prints score,lower.
 * The bucketed windows span at most w + one bucket (<= 130 s < 240 s), so the injection budget per window is still m,
 * and an LRU eviction caused by an injection at tau can only make a real flow novel within [tau, tau + w): the
 * Theorem 1' box (a = b = m) covers the engine's own features. Bounded memory: an evicted host entry loses real history,
 * which Theorem 1' does not cover (PREREG §9 K11), so a flow is blocked only when its source and destination entries
 * have been tracked continuously for >= AGE_MS (the longest bucketed window span); younger entries can alarm but
 * abstain from blocking (--no-age disables this, for the K11 comparison only). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "certb.h"
#include "hostctx.h"

#define AGE_MS 130000

static HcTable T_;
static CertB M_;

/* ---- model file (scripts/engine_export_b.py) ---- */
static void need(int ok, const char *what) { if (!ok) { fprintf(stderr, "model: bad %s\n", what); exit(2); } }
static void readv(FILE *f, const char *name, double *v, int n) {
  char w[32]; need(fscanf(f, "%31s", w) == 1 && !strcmp(w, name), name);
  for (int i = 0; i < n; i++) need(fscanf(f, "%lf", &v[i]) == 1, name);
}
static void load(const char *path) {
  CertB *c = &M_; FILE *f = fopen(path, "r"); need(f != NULL, "path");
  need(fscanf(f, "certb %d %d %d %d %d %d %d %d %lf", &c->D, &c->F, &c->BI, &c->K, &c->M, &c->W, &c->L, &c->m_budget,
              &c->tau) == 9, "header");
  need(c->D == c->F + CB_HC && c->D <= CB_MAXD && c->K <= CB_MAXK && c->M <= CB_MAXM && c->W <= CB_MAXW &&
       c->L <= CB_MAXL, "dims");
  double z[CB_MAXD];
  readv(f, "mu", c->mu, c->D); readv(f, "sd", c->sd, c->D); readv(f, "clo", c->clo, c->D); readv(f, "chi", c->chi, c->D);
  readv(f, "zero", z, c->D); for (int i = 0; i < c->D; i++) c->zero[i] = z[i] != 0;
  for (int m = 0; m < c->M; m++) { char w[8]; need(fscanf(f, "%7s", w) == 1 && !strcmp(w, "r"), "r");
    for (int i = 0; i < c->D; i++) need(fscanf(f, "%lf", &c->r[m][i]) == 1, "r"); }
  for (int l = 0; l < c->L; l++) {
    need(fscanf(f, " layer %d %d %lf", &c->rows[l], &c->cols[l], &c->ws[l]) == 3 && c->rows[l] <= CB_MAXW &&
         c->cols[l] <= CB_MAXC, "layer");
    for (int i = 0; i < c->rows[l] * c->cols[l]; i++) { int x; need(fscanf(f, "%d", &x) == 1, "weights"); c->w[l][i] = x; }
    for (int i = 0; i < c->rows[l]; i++) need(fscanf(f, "%lf", &c->b[l][i]) == 1, "bias");
  }
  need(fscanf(f, " head %lf", &c->hws) == 1, "head");
  for (int m = 0; m < c->M; m++) for (int i = 0; i < c->W * c->K; i++) { int x; need(fscanf(f, "%d", &x) == 1, "hw"); c->hw[m][i] = x; }
  for (int m = 0; m < c->M; m++) for (int k = 0; k < c->K; k++) need(fscanf(f, "%lf", &c->hb[m][k]) == 1, "hb");
  fclose(f);
}

int main(int argc, char **argv) {
  const char *model = "model_b.txt", *hook = NULL; long hosts = 4096; int prepped = 0, age = 1; double ttl = 300;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
    else if (!strcmp(argv[i], "--hosts") && i + 1 < argc) hosts = atol(argv[++i]);
    else if (!strcmp(argv[i], "--on-block") && i + 1 < argc) hook = argv[++i];
    else if (!strcmp(argv[i], "--ttl") && i + 1 < argc) ttl = atof(argv[++i]);
    else if (!strcmp(argv[i], "--prepped")) prepped = 1;
    else if (!strcmp(argv[i], "--no-age")) age = 0;
    else { fprintf(stderr, "usage: certids [--model F] [--hosts N] [--on-block 'cmd %%s'] [--ttl S] [--prepped] [--no-age] < flows.csv\n"); return 2; }
  }
  load(model);
  const CertB *c = &M_;
  char line[4096], src[256], dst[256]; long n = 0, nal = 0, nbl = 0, nab = 0; clock_t c0 = clock();
  double x[CB_MAXD], raw[CB_MAXD];
  if (prepped) {
    while (fgets(line, sizeof line, stdin)) {
      char *p = line;
      for (int i = 0; i < c->D; i++) { x[i] = strtod(p, &p); if (*p == ',') p++; }
      printf("%.17g,%.17g\n", cb_score(c, x), cb_block_lower(c, x)); n++;
    }
    return 0;
  }
  for (T_.cap = 1; T_.cap < (size_t)hosts; T_.cap <<= 1) {}
  T_.tab = calloc(T_.cap, sizeof(Host)); need(T_.tab != NULL, "memory");
  fprintf(stderr, "certids: %zu host slots x %zu B = %.1f MB state; B: %d members, width %d, m = %d, tau = %.6g\n",
          T_.cap, sizeof(Host), T_.cap * sizeof(Host) / 1e6, c->M, c->W, c->m_budget, c->tau);
  while (fgets(line, sizeof line, stdin)) {
    long long t; double portf, outp, inb; int off;
    if (sscanf(line, "%lld,%255[^,],%255[^,],%lf,%lf,%lf%n", &t, src, dst, &portf, &outp, &inb, &off) != 6) continue;
    int port = (int)portf; char *p = line + off;
    for (int i = 0; i < c->F; i++) { if (*p == ',') p++; raw[i] = strtod(p, &p); }  /* live flows have no missing fields */
    uint64_t ks = hc_fnv(src), kd = hc_fnv(dst);
    Host *hs, *hd;
    hc_flow(&T_, ks, kd, t, port, outp, inb, raw + c->F, &hs, &hd);
    cb_prep(c, raw, x);
    double s = cb_score(c, x), lower = -INFINITY;
    int alarm = s > c->tau, block = 0;
    if (alarm) {
      lower = cb_block_lower(c, x); block = lower > c->tau; nal++;
      if (block && age && (t - hs->created < AGE_MS || t - hd->created < AGE_MS)) { block = 0; nab++; }  /* abstain */
    }
    if (block) {
      nbl++;
      if (hook && t >= hs->blocked_until) {
        hs->blocked_until = t + (int64_t)(ttl * 1000);
        char cmd[512]; snprintf(cmd, sizeof cmd, hook, src);
        if (system(cmd) != 0) fprintf(stderr, "hook failed: %s\n", cmd);
      }
    }
    printf("%.7g,%d,%.7g,%d\n", s, alarm, lower, block); n++;
  }
  double sec = (double)(clock() - c0) / CLOCKS_PER_SEC;
  fprintf(stderr, "certids: %ld flows in %.2f s (%.2f us/flow), %ld alarms, %ld certified blocks, %ld abstentions "
          "(entry < %d s old), %ld evictions\n", n, sec, 1e6 * sec / (n ? n : 1), nal, nbl, nab, AGE_MS / 1000, T_.evictions);
  return 0;
}
