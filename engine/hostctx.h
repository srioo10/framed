/* hostctx.h: streaming host context for certb.h (PREREG §9 DEPLOY), shared by engine/certids.c and the ESP32 gateway.
 * Header-only, no stdio or malloc: the caller owns the Host table. Bounded memory: 1 s / 10 s buckets for the
 * 10 s / 120 s windows and per-host LRU novelty tables, as engine/onepass.c. Feature order = nids.hostctx.COLS:
 * window (10 s, 120 s) x side (src, dst) x stat (n, fail, newport, newpeer, bytes); shares are over max(n, 1).
 * The bucketed windows span at most w + one bucket (<= 130 s < 240 s), so the injection budget per window stays m. */
#ifndef HOSTCTX_H
#define HOSTCTX_H
#include <stdint.h>
#include <string.h>

#define NS 5
#define B10 10
#define B120 12
#define TAB 16
#define PROBE 8

typedef struct { int64_t id[B10]; float v[B10][NS]; int64_t id2[B120]; float v2[B120][NS];
                 int32_t port[TAB]; uint64_t peer[TAB]; int64_t tport[TAB], tpeer[TAB]; } Side;
typedef struct { uint64_t key; int64_t last, created, blocked_until; Side s[2]; } Host;
typedef struct { Host *tab; size_t cap; long evictions; } HcTable;  /* cap = power of two */

static inline uint64_t hc_fnv(const char *p) {
  uint64_t h = 1469598103934665603ULL;
  while (*p) h = (h ^ (uint8_t)*p++) * 1099511628211ULL;
  return h ? h : 1;
}

static inline Host *hc_get(HcTable *T, uint64_t key, int64_t t, const Host *keep) {
  size_t i = key & (T->cap - 1), stale = SIZE_MAX;
  for (int k = 0; k < PROBE; k++, i = (i + 1) & (T->cap - 1)) {
    if (T->tab[i].key == key) return &T->tab[i];
    if (T->tab[i].key == 0) { stale = i; goto fresh; }
    if (&T->tab[i] != keep && (stale == SIZE_MAX || T->tab[i].last < T->tab[stale].last)) stale = i;
  }
  T->evictions++;
fresh:
  memset(&T->tab[stale], 0, sizeof(Host));
  T->tab[stale].key = key; T->tab[stale].last = T->tab[stale].created = t;
  for (int b = 0; b < B10; b++) T->tab[stale].s[0].id[b] = T->tab[stale].s[1].id[b] = -1;
  for (int b = 0; b < B120; b++) T->tab[stale].s[0].id2[b] = T->tab[stale].s[1].id2[b] = -1;
  return &T->tab[stale];
}

static inline void hc_wsum(const Side *s, int64_t t, int big, float *out) {
  memset(out, 0, NS * sizeof(float));
  int64_t now = big ? t / 10000 : t / 1000, nb = big ? B120 : B10;
  for (int b = 0; b < nb; b++) {
    int64_t id = big ? s->id2[b] : s->id[b];
    if (id >= 0 && now - id < nb)
      for (int j = 0; j < NS; j++) out[j] += big ? s->v2[b][j] : s->v[b][j];
  }
}

static inline int64_t hc_seen(int64_t *ts, void *keys, int is64, uint64_t k, int64_t t) {
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

static inline void hc_add(Side *s, int64_t t, const float *v) {
  int64_t id = t / 1000, id2 = t / 10000; int b = id % B10, b2 = id2 % B120;
  if (s->id[b] != id) { s->id[b] = id; memset(s->v[b], 0, sizeof s->v[b]); }
  if (s->id2[b2] != id2) { s->id2[b2] = id2; memset(s->v2[b2], 0, sizeof s->v2[b2]); }
  for (int j = 0; j < NS; j++) { s->v[b][j] += v[j]; s->v2[b2][j] += v[j + NS]; }
}

/* One flow (start t ms, hashed hosts ks -> kd): writes the 20 context features seen BEFORE this flow into hc[20],
 * then adds the flow to both hosts. Returns the source host (for blocking) via *hs, the destination via *hd. */
static inline void hc_flow(HcTable *T, uint64_t ks, uint64_t kd, int64_t t, int port, double out_pkts, double in_bytes,
                           double *hc, Host **hs_out, Host **hd_out) {
  Host *hs = hc_get(T, ks, t, NULL); hs->last = t;
  Host *hd = ks == kd ? hs : hc_get(T, kd, t, hs); hd->last = t;
  float w[NS];
  Side *side[2] = {&hs->s[0], &hd->s[1]};
  for (int big = 0, o = 0; big < 2; big++)
    for (int k = 0; k < 2; k++, o += NS) {
      hc_wsum(side[k], t, big, w);
      double d = w[0] > 1 ? w[0] : 1;
      hc[o] = w[0]; for (int j = 1; j < NS; j++) hc[o + j] = w[j] / d;
    }
  float v[2][2 * NS];
  for (int k = 0; k < 2; k++) {
    int64_t tp = hc_seen(side[k]->tport, side[k]->port, 0, (uint32_t)port, t);
    int64_t tq = hc_seen(side[k]->tpeer, side[k]->peer, 1, k ? ks : kd, t);
    for (int big = 0; big < 2; big++) {
      int64_t wms = big ? 120000 : 10000;
      float *u = v[k] + big * NS;
      u[0] = 1; u[1] = out_pkts == 0; u[2] = tp == INT64_MIN || t - tp > wms; u[3] = tq == INT64_MIN || t - tq > wms;
      u[4] = in_bytes;
    }
    hc_add(side[k], t, v[k]);
  }
  *hs_out = hs; *hd_out = hd;
}
#endif
