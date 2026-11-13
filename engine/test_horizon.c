/* Self-check (2026-11-13): can removing in-budget injections raise a key's 120 s new-port count by more than m?
 * Concern (review, Nov 13): if the bucketed window spanned 130 s, novelty (120 s lookback) would reach 250 s back
 * and 2m in-budget injections could matter. Result: the window spans 110-120 s, so the reach stays < 240 s.
 * Builds an observed stream (real + injected) and its clean twin (real only) and compares the src side's 120 s
 * window at the decision flow. Prints the new-port count increase; the box allows +m.
 * Build: cc -O1 -I engine -I firmware/gateway/main engine/test_horizon.c -lm -o /tmp/th && /tmp/th */
#include <assert.h>
#include <stdio.h>
#include "certb.h"
#include "hostctx.h"

#define M 10
#define S 1000LL

/* Worst case found by hand: batch 1 (m flows) sits 119.5 s before real flows A at T-x, so A look old; batch 2
 * (m flows, T-1 s) re-touches ports B last seen by real flows at T-120.9 s, so the real B flows at T-0.5 s look old.
 * Batch 1 and batch 2 are x+118.5 s > 240 s apart, so the attacker stays in budget. If the bucketed window still
 * holds A (x up to ~130 s), removing both batches turns 2m real flows new, but the box allows only +m. */
static void run(int inject, int64_t T, int64_t x, double *n, double *np) {
  static Host tab[64]; HcTable H = {tab, 64, 0};
  for (int i = 0; i < 64; i++) tab[i] = (Host){0};
  double hc[CB_HC]; Host *hs, *hd;
  if (inject) for (int i = 0; i < M; i++) hc_flow(&H, 11, 22, T - x - 119500 + i, 5000 + i, 1, 100, hc, &hs, &hd);
  for (int i = 0; i < M; i++) hc_flow(&H, 11, 22, T - x + i, 5000 + i, 1, 100, hc, &hs, &hd);         /* real A */
  for (int i = 0; i < M; i++) hc_flow(&H, 11, 22, T - 120900 + i, 6000 + i, 1, 100, hc, &hs, &hd);    /* real B early */
  if (inject) for (int i = 0; i < M; i++) hc_flow(&H, 11, 22, T - 1000 + i, 6000 + i, 1, 100, hc, &hs, &hd);
  for (int i = 0; i < M; i++) hc_flow(&H, 11, 22, T - 500 + i, 6000 + i, 1, 100, hc, &hs, &hd);      /* real B late */
  hc_flow(&H, 11, 22, T, 7000, 1, 100, hc, &hs, &hd);                                                 /* decision */
  *n = hc[2 * NS]; *np = hc[2 * NS + 2] * (*n > 1 ? *n : 1);
}

/* invariant behind a = m: a flow older than 120 s is never in the 120 s window (12 x 10 s buckets cover 110-120 s) */
static int in_window(int64_t T, int64_t x) {
  static Host tab[8]; HcTable H = {tab, 8, 0};
  for (int i = 0; i < 8; i++) tab[i] = (Host){0};
  double hc[CB_HC]; Host *hs, *hd;
  hc_flow(&H, 11, 22, T - x, 5000, 1, 100, hc, &hs, &hd);
  hc_flow(&H, 11, 22, T, 7000, 1, 100, hc, &hs, &hd);
  return hc[2 * NS] > 1.5;
}

int main(void) {
  for (int64_t T = 400 * S; T < 420 * S; T += 37)
    for (int64_t x = 120001; x <= 131000; x += 250) assert(!in_window(T, x));
  int worst = 0;
  for (int64_t T = 400 * S; T < 410 * S; T += 700)
    for (int64_t w = 122000; w <= 130000; w += 500) {
      double n1, p1, n0, p0;
      run(1, T, w, &n1, &p1); run(0, T, w, &n0, &p0);
      int rise = (int)(p0 - p1 + 0.5);
      if (rise > worst) worst = rise;
    }
  assert(worst <= M);
  printf("test_horizon ok: 120 s window never holds a flow > 120 s old; worst in-budget new-port rise %d <= m=%d\n", worst, M);
  return 0;
}
