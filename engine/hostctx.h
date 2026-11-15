/* hostctx.h: streaming host context for certb.h (PREREG §9 DEPLOY), shared by engine/certids.c and the ESP32 gateway.
 * Header-only, no stdio or malloc: the caller owns the Host table. Bounded memory: 1 s / 10 s buckets for the
 * 10 s / 120 s windows and per-host LRU novelty tables, as engine/onepass.c. Feature order = nids.hostctx.COLS:
 * window (10 s, 120 s) x side (src, dst) x stat (n, fail, newport, newpeer, bytes); shares are over max(n, 1).
 * The bucketed windows span w - one bucket to w (110-120 s; engine/test_horizon.c), so with the 120 s novelty lookback an injection reaches < 240 s, so the injection budget per window stays m. */
#ifndef HOSTCTX_H
#define HOSTCTX_H
#include <stdint.h>
#include <string.h>

#define NS 5
#define B10 10
#define B120 12
#define TAB 16
#define PROBE 8

#ifdef HC_BLOOM
/* evicted-live memory (LIVE-SCAN, 15 Nov): per novelty table, a Bloom filter of keys evicted while seen < 120 s ago,
 * two generations of >= 120 s, so a miss on a key it does not hold is new in the clean stream too (no false
 * negatives); u counts the window's misses it does hold, the only flows eviction can have made look new. */
#ifndef BW
#define BW 16  /* 32-bit words per generation: 512 bits (ESP32 DRAM; 1024 bits left 21 KB) */
#endif
#define HC_BLOOM_FIELDS uint32_t bl[2][2][BW]; int64_t bgen; float uq[B10][2], uq2[B120][2];
#else
#define HC_BLOOM_FIELDS
#endif
typedef struct { int64_t id[B10]; float v[B10][NS]; int64_t id2[B120]; float v2[B120][NS];
                 int32_t port[TAB]; uint64_t peer[TAB]; int64_t tport[TAB], tpeer[TAB];
                 int64_t tev; HC_BLOOM_FIELDS } Side;  /* tev: last time a port/peer entry still inside 120 s was evicted */
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
  T->tab[stale].s[0].tev = T->tab[stale].s[1].tev = INT64_MIN / 2;
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

/* LRU novelty table. Evicting an entry seen < 120 s ago makes a later real flow look new: injected flows can then
 * inflate novelty beyond m, so the eviction time goes to *tev and the certificate widens that side (cb_block_lower). */
#ifdef HC_BLOOM
static inline uint32_t hc_bh(uint64_t k, int i) { k = (k ^ (i ? 0x9e3779b97f4a7c15ULL : 0)) * 0xff51afd7ed558ccdULL; return (uint32_t)(k >> 40) % (BW * 32); }
static inline void hc_bset(uint32_t *b, uint64_t k) { for (int i = 0; i < 2; i++) { uint32_t h = hc_bh(k, i); b[h / 32] |= 1u << h % 32; } }
static inline int hc_bget(const uint32_t *b, uint64_t k) {
  for (int i = 0; i < 2; i++) { uint32_t h = hc_bh(k, i); if (!(b[h / 32] >> h % 32 & 1)) return 0; }
  return 1;
}
static inline void hc_brot(Side *s, int64_t t) {  /* generation 0 = current; each generation spans >= 120 s */
  if (t - s->bgen < 120000) return;
  if (t - s->bgen < 240000) memcpy(s->bl[0][1], s->bl[0][0], sizeof s->bl[0][0]), memcpy(s->bl[1][1], s->bl[1][0], sizeof s->bl[1][0]);
  else memset(s->bl, 0, sizeof s->bl);
  memset(s->bl[0][0], 0, sizeof s->bl[0][0]); memset(s->bl[1][0], 0, sizeof s->bl[1][0]);
  s->bgen = t;
}
#define HC_BL_ARG , uint32_t (*bl)[BW], int *unc
#else
#define HC_BL_ARG
#endif
static inline int64_t hc_seen(int64_t *ts, void *keys, int is64, uint64_t k, int64_t t, int64_t *tev HC_BL_ARG) {
  int old = 0;
  for (int i = 0; i < TAB; i++) {
    uint64_t ki = is64 ? ((uint64_t *)keys)[i] : (uint64_t)(uint32_t)((int32_t *)keys)[i];
    if (ts[i] && ki == k) { int64_t p = ts[i]; ts[i] = t; return p; }
    if (ts[i] < ts[old]) old = i;
  }
#ifdef HC_BLOOM
  *unc = hc_bget(bl[0], k) || hc_bget(bl[1], k);
  if (ts[old] && t - ts[old] <= 120000)
    hc_bset(bl[0], is64 ? ((uint64_t *)keys)[old] : (uint64_t)(uint32_t)((int32_t *)keys)[old]);
#endif
  if (ts[old] && t - ts[old] <= 120000) *tev = t;
  if (is64) ((uint64_t *)keys)[old] = k; else ((int32_t *)keys)[old] = (int32_t)k;
  ts[old] = t;
  return INT64_MIN;
}

static inline void hc_add(Side *s, int64_t t, const float *v, const float *uq) {
  int64_t id = t / 1000, id2 = t / 10000; int b = id % B10, b2 = id2 % B120;
#ifdef HC_BLOOM
  if (s->id[b] != id) memset(s->uq[b], 0, sizeof s->uq[b]);
  if (s->id2[b2] != id2) memset(s->uq2[b2], 0, sizeof s->uq2[b2]);
  for (int j = 0; j < 2; j++) { s->uq[b][j] += uq[j]; s->uq2[b2][j] += uq[j]; }
#else
  (void)uq;
#endif
  if (s->id[b] != id) { s->id[b] = id; memset(s->v[b], 0, sizeof s->v[b]); }
  if (s->id2[b2] != id2) { s->id2[b2] = id2; memset(s->v2[b2], 0, sizeof s->v2[b2]); }
  for (int j = 0; j < NS; j++) { s->v[b][j] += v[j]; s->v2[b2][j] += v[j + NS]; }
}

/* One flow (t ms: the caller's clock, close time on the gateway; must not go backwards, or a live bucket is reused;
 * hashed hosts ks -> kd): writes the 20 context features seen BEFORE this flow into hc[20],
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
  float v[2][2 * NS], uq[2] = {0, 0};
  for (int k = 0; k < 2; k++) {
#ifdef HC_BLOOM
    int up = 0, uqp = 0;
    hc_brot(side[k], t);
    int64_t tp = hc_seen(side[k]->tport, side[k]->port, 0, (uint32_t)port, t, &side[k]->tev, side[k]->bl[0], &up);
    int64_t tq = hc_seen(side[k]->tpeer, side[k]->peer, 1, k ? ks : kd, t, &side[k]->tev, side[k]->bl[1], &uqp);
    uq[0] = up; uq[1] = uqp;
#else
    int64_t tp = hc_seen(side[k]->tport, side[k]->port, 0, (uint32_t)port, t, &side[k]->tev);
    int64_t tq = hc_seen(side[k]->tpeer, side[k]->peer, 1, k ? ks : kd, t, &side[k]->tev);
#endif
    for (int big = 0; big < 2; big++) {
      int64_t wms = big ? 120000 : 10000;
      float *u = v[k] + big * NS;
      u[0] = 1; u[1] = out_pkts == 0; u[2] = tp == INT64_MIN || t - tp > wms; u[3] = tq == INT64_MIN || t - tq > wms;
      u[4] = in_bytes;
    }
    hc_add(side[k], t, v[k], uq);
  }
  *hs_out = hs; *hd_out = hd;
}

/* certificate widening mask for the flow just added at t: bit k = side k (0 src, 1 dst) had a recent eviction */
#ifndef HC_FLAG_MS
#define HC_FLAG_MS 130000  /* deployed builds; 250000 makes Lemma 2(iii)'s converse provable (eviction < 240 s back) */
#endif
static inline int hc_wide(const Host *hs, const Host *hd, int64_t t) {
  return (t - hs->s[0].tev < HC_FLAG_MS) | (t - hd->s[1].tev < HC_FLAG_MS) << 1;
}
#ifdef HC_BLOOM
/* u[g][s]: flows in group g's window (g = window * 2 + side, as hostctx.COLS) whose new-port (s = 0) / new-peer (s = 1)
 * lookup missed on a key the evicted-live filter holds. Read after hc_flow, so it may include the current flow: u can
 * only be over-counted, which only lowers the box (safe). */
static inline void hc_unc(const Host *hs, const Host *hd, int64_t t, double u[4][2]) {
  const Side *side[2] = {&hs->s[0], &hd->s[1]};
  for (int big = 0; big < 2; big++)
    for (int k = 0; k < 2; k++) {
      int64_t now = big ? t / 10000 : t / 1000, nb = big ? B120 : B10;
      u[big * 2 + k][0] = u[big * 2 + k][1] = 0;
      for (int b = 0; b < nb; b++) {
        int64_t id = big ? side[k]->id2[b] : side[k]->id[b];
        if (id >= 0 && now - id < nb)
          for (int j = 0; j < 2; j++) u[big * 2 + k][j] += big ? side[k]->uq2[b][j] : side[k]->uq[b][j];
      }
    }
}
#endif
#endif
