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
 * Theorem 1' box (a = b = m) covers the engine's own features. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "certb.h"

#define NS 5
#define B10 10
#define B120 12
#define TAB 16
#define PROBE 8

typedef struct { int64_t id[B10]; float v[B10][NS]; int64_t id2[B120]; float v2[B120][NS];
                 int32_t port[TAB]; uint64_t peer[TAB]; int64_t tport[TAB], tpeer[TAB]; } Side;
typedef struct { uint64_t key; int64_t last, blocked_until; Side s[2]; } Host;

static Host *tab; static size_t cap; static long evictions;
static CertB M_;

static uint64_t fnv(const char *p) { uint64_t h = 1469598103934665603ULL; while (*p) h = (h ^ (uint8_t)*p++) * 1099511628211ULL; return h ? h : 1; }

static Host *get(uint64_t key, int64_t t, const Host *keep) {
  size_t i = key & (cap - 1), stale = SIZE_MAX;
  for (int k = 0; k < PROBE; k++, i = (i + 1) & (cap - 1)) {
    if (tab[i].key == key) return &tab[i];
    if (tab[i].key == 0) { stale = i; goto fresh; }
    if (&tab[i] != keep && (stale == SIZE_MAX || tab[i].last < tab[stale].last)) stale = i;
  }
  evictions++;
fresh:
  memset(&tab[stale], 0, sizeof(Host));
  tab[stale].key = key; tab[stale].last = t;
  for (int b = 0; b < B10; b++) tab[stale].s[0].id[b] = tab[stale].s[1].id[b] = -1;
  for (int b = 0; b < B120; b++) tab[stale].s[0].id2[b] = tab[stale].s[1].id2[b] = -1;
  return &tab[stale];
}

static void wsum(const Side *s, int64_t t, int big, float *out) {
  memset(out, 0, NS * sizeof(float));
  int64_t now = big ? t / 10000 : t / 1000, nb = big ? B120 : B10;
  for (int b = 0; b < nb; b++) {
    int64_t id = big ? s->id2[b] : s->id[b];
    if (id >= 0 && now - id < nb)
      for (int j = 0; j < NS; j++) out[j] += big ? s->v2[b][j] : s->v[b][j];
  }
}

static int64_t seen(int64_t *ts, void *keys, int is64, uint64_t k, int64_t t) {
  int old = 0;
  for (int i = 0; i < TAB; i++) {
    uint64_t ki = is64 ? ((uint64_t *)keys)[i] : (uint64_t)(uint32_t)((int32_t *)keys)[i];
    if (ts[i] && ki == k) { int64_t p = ts[i]; ts[i] = t; return p; }
    if (ts[i] < ts[old]) old = i;
  }
  if (is64) ((uint64_t *)keys)[old] = k; else ((int32_t *)keys)[old] = (int32_t)k;
  ts[old] = t;
  return INT64_MIN;
}

static void add(Side *s, int64_t t, const float *v) {
  int64_t id = t / 1000, id2 = t / 10000; int b = id % B10, b2 = id2 % B120;
  if (s->id[b] != id) { s->id[b] = id; memset(s->v[b], 0, sizeof s->v[b]); }
  if (s->id2[b2] != id2) { s->id2[b2] = id2; memset(s->v2[b2], 0, sizeof s->v2[b2]); }
  for (int j = 0; j < NS; j++) { s->v[b][j] += v[j]; s->v2[b2][j] += v[j + NS]; }
}

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
  const char *model = "model_b.txt", *hook = NULL; long hosts = 4096; int prepped = 0; double ttl = 300;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
    else if (!strcmp(argv[i], "--hosts") && i + 1 < argc) hosts = atol(argv[++i]);
    else if (!strcmp(argv[i], "--on-block") && i + 1 < argc) hook = argv[++i];
    else if (!strcmp(argv[i], "--ttl") && i + 1 < argc) ttl = atof(argv[++i]);
    else if (!strcmp(argv[i], "--prepped")) prepped = 1;
    else { fprintf(stderr, "usage: certids [--model F] [--hosts N] [--on-block 'cmd %%s'] [--ttl S] [--prepped] < flows.csv\n"); return 2; }
  }
  load(model);
  const CertB *c = &M_;
  char line[4096], src[256], dst[256]; long n = 0, nal = 0, nbl = 0; clock_t c0 = clock();
  double x[CB_MAXD], raw[CB_MAXD];
  if (prepped) {
    while (fgets(line, sizeof line, stdin)) {
      char *p = line;
      for (int i = 0; i < c->D; i++) { x[i] = strtod(p, &p); if (*p == ',') p++; }
      printf("%.17g,%.17g\n", cb_score(c, x), cb_block_lower(c, x)); n++;
    }
    return 0;
  }
  for (cap = 1; cap < (size_t)hosts; cap <<= 1) {}
  tab = calloc(cap, sizeof(Host)); need(tab != NULL, "memory");
  fprintf(stderr, "certids: %zu host slots x %zu B = %.1f MB state; B: %d members, width %d, m = %d, tau = %.6g\n",
          cap, sizeof(Host), cap * sizeof(Host) / 1e6, c->M, c->W, c->m_budget, c->tau);
  while (fgets(line, sizeof line, stdin)) {
    long long t; double portf, outp, inb; int off;
    if (sscanf(line, "%lld,%255[^,],%255[^,],%lf,%lf,%lf%n", &t, src, dst, &portf, &outp, &inb, &off) != 6) continue;
    int port = (int)portf; char *p = line + off;
    for (int i = 0; i < c->F; i++) { if (*p == ',') p++; raw[i] = strtod(p, &p); }  /* live flows have no missing fields */
    uint64_t ks = fnv(src), kd = fnv(dst);
    Host *hs = get(ks, t, NULL); hs->last = t;
    Host *hd = ks == kd ? hs : get(kd, t, hs); hd->last = t;
    float w[NS];
    Side *side[2] = {&hs->s[0], &hd->s[1]};
    for (int big = 0, o = c->F; big < 2; big++)
      for (int k = 0; k < 2; k++, o += NS) {
        wsum(side[k], t, big, w);
        double d = w[0] > 1 ? w[0] : 1;
        raw[o] = w[0]; for (int j = 1; j < NS; j++) raw[o + j] = w[j] / d;
      }
    float v[2][2 * NS];
    for (int k = 0; k < 2; k++) {
      int64_t tp = seen(side[k]->tport, side[k]->port, 0, (uint32_t)port, t);
      int64_t tq = seen(side[k]->tpeer, side[k]->peer, 1, k ? ks : kd, t);
      for (int big = 0; big < 2; big++) {
        int64_t wms = big ? 120000 : 10000;
        float *u = v[k] + big * NS;
        u[0] = 1; u[1] = outp == 0; u[2] = tp == INT64_MIN || t - tp > wms; u[3] = tq == INT64_MIN || t - tq > wms; u[4] = inb;
      }
      add(side[k], t, v[k]);
    }
    cb_prep(c, raw, x);
    double s = cb_score(c, x), lower = -INFINITY;
    int alarm = s > c->tau, block = 0;
    if (alarm) { lower = cb_block_lower(c, x); block = lower > c->tau; nal++; }
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
  fprintf(stderr, "certids: %ld flows in %.2f s (%.2f us/flow), %ld alarms, %ld certified blocks, %ld evictions\n",
          n, sec, 1e6 * sec / (n ? n : 1), nal, nbl, evictions);
  return 0;
}
