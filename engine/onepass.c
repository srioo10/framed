/* OnePass streaming engine: bounded-memory host context + int8 MLP + conformal CUSUM per source host.
 *
 * Input  (stdin, one flow per line, CSV, time ordered):
 *   t_ms,src,dst,dst_port,out_pkts,in_bytes,f_1,...,f_F      (f_* = the model's per-flow features, model order)
 * Output (stdout, one line per flow): score,u,cusum,alarm_P,alarm_A   (--hc also prints the 20 host features)
 * Alarms: the first A (CUSUM) alarm of a host on a day runs --on-alarm CMD with %s = source host (mitigation hook).
 *
 * Host context approximates scripts' exact windows with buckets: 10 s window = 10 x 1 s, 120 s window = 12 x 10 s
 * (edge error <= one bucket). Novelty of (host, port) and (host, peer) pairs comes from small per-host LRU tables
 * (TAB slots), so a host with more distinct recent pairs than TAB over-counts novelty (the safe direction).
 * Memory is fixed at start: --hosts N entries; a full table evicts the stalest of PROBE probed entries.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NS 5                /* n, fail, newport, newpeer, bytes */
#define B10 10              /* 1 s buckets */
#define B120 12             /* 10 s buckets */
#define TAB 16
#define PROBE 8
#define MAXF 64
#define MAXW 128
#define DAY_MS 86400000LL

typedef struct { int64_t id[B10]; float v[B10][NS]; int64_t id2[B120]; float v2[B120][NS];
                 int32_t port[TAB]; uint64_t peer[TAB]; int64_t tport[TAB], tpeer[TAB]; } Side;
typedef struct { uint64_t key; int64_t last; Side s[2]; double S, Smin; int64_t day; int alarmed, palarm; } Host;

static Host *tab; static size_t cap; static long evictions;

static uint64_t fnv(const char *p) { uint64_t h = 1469598103934665603ULL; while (*p) h = (h ^ (uint8_t)*p++) * 1099511628211ULL; return h ? h : 1; }

static Host *get(uint64_t key, int64_t t, const Host *keep) { /* keep: an entry this flow already holds */
  size_t i = key & (cap - 1), stale = SIZE_MAX;
  for (int k = 0; k < PROBE; k++, i = (i + 1) & (cap - 1)) {
    if (tab[i].key == key) return &tab[i];
    if (tab[i].key == 0) { stale = i; goto fresh; }
    if (&tab[i] != keep && (stale == SIZE_MAX || tab[i].last < tab[stale].last)) stale = i;
  }
  evictions++;
fresh:
  memset(&tab[stale], 0, sizeof(Host));
  tab[stale].key = key; tab[stale].last = t; tab[stale].day = -1;
  for (int b = 0; b < B10; b++) tab[stale].s[0].id[b] = tab[stale].s[1].id[b] = -1;
  for (int b = 0; b < B120; b++) tab[stale].s[0].id2[b] = tab[stale].s[1].id2[b] = -1;
  return &tab[stale];
}

/* sums over buckets inside [t - w, t]: out[NS] */
static void wsum(const Side *s, int64_t t, int big, float *out) {
  memset(out, 0, NS * sizeof(float));
  int64_t now = big ? t / 10000 : t / 1000, nb = big ? B120 : B10;
  for (int b = 0; b < nb; b++) {
    int64_t id = big ? s->id2[b] : s->id[b];
    if (id >= 0 && now - id < nb)
      for (int j = 0; j < NS; j++) out[j] += big ? s->v2[b][j] : s->v[b][j];
  }
}

/* last time this pair was seen (or -inf); records t, evicting the oldest slot */
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

/* ---- model ---- */
static int D, F, HC, BI, K, L, NREF; static double mean[MAXF], mu[MAXF], sd[MAXF], kap[4], thP, thA;
static int rows_[3], cols_[3]; static int8_t *W[3]; static double scale[3], *bias[3], *ref;

static void need(int ok, const char *what) { if (!ok) { fprintf(stderr, "model.txt: bad %s\n", what); exit(2); } }
static void readv(FILE *f, const char *name, double *v, int n) {
  char w[32]; need(fscanf(f, "%31s", w) == 1 && !strcmp(w, name), name);
  for (int i = 0; i < n; i++) need(fscanf(f, "%lf", &v[i]) == 1, name);
}
static void load(const char *path) {
  FILE *f = fopen(path, "r"); need(f != NULL, "path");
  need(fscanf(f, "dims %d %d %d benign %d classes %d", &D, &F, &HC, &BI, &K) == 5, "dims");
  need(D == F + HC && D <= MAXF && HC == 20, "dims");
  readv(f, "mean", mean, D); readv(f, "mu", mu, D); readv(f, "sd", sd, D);
  for (L = 0; L < 3; L++) {
    need(fscanf(f, " layer %d %d %lf", &rows_[L], &cols_[L], &scale[L]) == 3 && rows_[L] <= MAXW, "layer");
    W[L] = malloc(rows_[L] * cols_[L]); bias[L] = malloc(rows_[L] * sizeof(double));
    for (int i = 0; i < rows_[L] * cols_[L]; i++) { int x; need(fscanf(f, "%d", &x) == 1, "weights"); W[L][i] = x; }
    for (int i = 0; i < rows_[L]; i++) need(fscanf(f, "%lf", &bias[L][i]) == 1, "bias");
  }
  need(fscanf(f, " ref %d", &NREF) == 1, "ref"); ref = malloc(NREF * sizeof(double));
  for (int i = 0; i < NREF; i++) need(fscanf(f, "%lf", &ref[i]) == 1, "ref");
  readv(f, "kappas", kap, 4);
  need(fscanf(f, " thresholds %lf %lf", &thP, &thA) == 2, "thresholds");
  fclose(f);
}

static double forward(const double *x) {
  double a[MAXW], b[MAXW]; memcpy(a, x, D * sizeof(double));
  for (int l = 0; l < 3; l++) {
    for (int r = 0; r < rows_[l]; r++) {
      double z = 0; const int8_t *w = W[l] + (size_t)r * cols_[l];
      for (int c = 0; c < cols_[l]; c++) z += w[c] * a[c];
      z = z * scale[l] + bias[l][r];
      b[r] = (l < 2 && z < 0) ? 0 : z;
    }
    memcpy(a, b, rows_[l] * sizeof(double));
  }
  double m = a[0], s = 0;
  for (int k = 1; k < K; k++) if (a[k] > m) m = a[k];
  for (int k = 0; k < K; k++) s += exp(a[k] - m);
  return 1 - exp(a[BI] - m) / s;
}

static double pval(double s) { /* (1 + #{ref >= s}) / (N + 1), ref sorted ascending */
  int lo = 0, hi = NREF;
  while (lo < hi) { int m = (lo + hi) / 2; if (ref[m] < s) lo = m + 1; else hi = m; }
  return (1.0 + NREF - lo) / (NREF + 1.0);
}

int main(int argc, char **argv) {
  const char *model = "model.txt", *hook = NULL; long hosts = 4096; int dump = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
    else if (!strcmp(argv[i], "--hosts") && i + 1 < argc) hosts = atol(argv[++i]);
    else if (!strcmp(argv[i], "--on-alarm") && i + 1 < argc) hook = argv[++i];
    else if (!strcmp(argv[i], "--hc")) dump = 1;
    else { fprintf(stderr, "usage: onepass [--model F] [--hosts N] [--on-alarm 'cmd %%s'] [--hc] < flows.csv\n"); return 2; }
  }
  load(model);
  for (cap = 1; cap < (size_t)hosts; cap <<= 1) {}
  tab = calloc(cap, sizeof(Host)); need(tab != NULL, "memory");
  fprintf(stderr, "onepass: %zu host slots x %zu B = %.1f MB state, model %d int8 weights\n", cap, sizeof(Host),
          cap * sizeof(Host) / 1e6, rows_[0] * cols_[0] + rows_[1] * cols_[1] + rows_[2] * cols_[2]);
  char line[4096], src[256], dst[256]; long n = 0; clock_t c0 = clock();
  while (fgets(line, sizeof line, stdin)) {
    long long t; double portf, outp, inb, f[MAXF], x[MAXF]; int off;
    if (sscanf(line, "%lld,%255[^,],%255[^,],%lf,%lf,%lf%n", &t, src, dst, &portf, &outp, &inb, &off) != 6) continue;
    int port = (int)portf; char *p = line + off;
    for (int i = 0; i < F; i++) {  /* empty field = missing (NaN -> train mean) */
      if (*p == ',') p++;
      if (*p == ',' || *p == '\n' || *p == 0) f[i] = NAN; else f[i] = strtod(p, &p);
    }
    uint64_t ks = fnv(src), kd = fnv(dst);
    Host *hs = get(ks, t, NULL); hs->last = t;
    Host *hd = ks == kd ? hs : get(kd, t, hs); hd->last = t;
    float hc[20], w[NS];
    Side *side[2] = {&hs->s[0], &hd->s[1]};
    for (int big = 0, o = 0; big < 2; big++)
      for (int k = 0; k < 2; k++, o += NS) {
        wsum(side[k], t, big, w);
        double d = w[0] > 1 ? w[0] : 1;
        hc[o] = w[0]; for (int j = 1; j < NS; j++) hc[o + j] = w[j] / d;
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
    for (int i = 0; i < F; i++) x[i] = f[i];
    for (int i = 0; i < HC; i++) x[F + i] = hc[i];
    for (int i = 0; i < D; i++) {
      double r = isfinite(x[i]) ? x[i] : mean[i];
      x[i] = ((r < 0 ? -1 : 1) * log1p(fabs(r)) - mu[i]) / sd[i];
    }
    double s = forward(x), u = pval(s), g = 0;
    for (int k = 0; k < 4; k++) g += kap[k] * pow(u, kap[k] - 1);
    int64_t day = t / DAY_MS;
    if (hs->day != day) { hs->day = day; hs->S = hs->Smin = 0; hs->alarmed = hs->palarm = 0; }
    hs->S += log(g / 4); if (hs->S < hs->Smin) hs->Smin = hs->S;
    double A = hs->S - (hs->Smin < 0 ? hs->Smin : 0);
    int aP = s > thP, aA = A > thA;
    if (aA && !hs->alarmed) {
      hs->alarmed = 1;
      if (hook) { char cmd[512]; snprintf(cmd, sizeof cmd, hook, src); if (system(cmd) != 0) fprintf(stderr, "hook failed: %s\n", cmd); }
    }
    printf("%.7g,%.7g,%.7g,%d,%d", s, u, A, aP, aA);
    if (dump) for (int i = 0; i < 20; i++) printf(",%.7g", hc[i]);
    putchar('\n'); n++;
  }
  double sec = (double)(clock() - c0) / CLOCKS_PER_SEC;
  fprintf(stderr, "onepass: %ld flows in %.2f s (%.2f us/flow), %ld evictions\n", n, sec, 1e6 * sec / (n ? n : 1), evictions);
  return 0;
}
