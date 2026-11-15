/* test_bloom (15 Nov, LIVE-SCAN): test_lru for the evicted-live filter build (-DHC_BLOOM). Low side, always (no flag
 * escape): clean new-port count >= observed - m - u, u = hc_unc's count; high side: clean <= observed + m unless the
 * 250 s flag is set. Build: cc -O2 -DHC_BLOOM -DHC_FLAG_MS=250000 -I engine -I firmware/gateway/main engine/test_bloom.c -lm
 * Original header: Brute-force check of Lemma 2(iii) both directions (review Nov 13): with bounded LRU novelty tables, can removing
 * in-budget injections move a decision's source-side new-port count outside [c'-m, c'+m] while the eviction flag is
 * clear? Random streams on one source key, small port universes (forcing evictions), adversarial-random injections
 * (<= M per 240 s). Compares the observed row (real + injected) with the clean twin (real only) at every real flow.
 * Build: cc -O2 -I engine -I firmware/gateway/main engine/test_lru.c -lm -o /tmp/tl && /tmp/tl */
#include <stdio.h>
#include <stdlib.h>
#include "certb.h"
#include "hostctx.h"

#define M 3
#define NR 200
typedef struct { int64_t t; int port; int inj; } Ev;
static int cmp(const void *a, const void *b) { int64_t d = ((Ev *)a)->t - ((Ev *)b)->t; return d < 0 ? -1 : d > 0; }

int main(void) {
  static Host tabA[64], tabB[64]; srand(7);
  long checked = 0, viol = 0, flagged = 0; double usum = 0;
  for (int trial = 0; trial < 100000; trial++) {
    Ev ev[NR + 64]; int n = 0, U = 17 + rand() % 50;         /* port universe just above TAB = 16 */
    int64_t T = 0;
    for (int i = 0; i < NR; i++) { T += 200 + rand() % (1 + rand() % 20000); ev[n++] = (Ev){T, rand() % U, 0}; }
    for (int64_t s = rand() % 200000; s < T; s += 240001 + rand() % 200000)  /* bursts of M, >= 240 s apart: in budget */
      if (rand() % 3) for (int j = 0; j < M; j++) {
        int port = rand() % U, mode = rand() % 3;            /* 0: random; 1: pre-touch a real flow in the next 120 s; */
        if (mode == 1) for (int q = 0; q < NR; q++)          /* 2: a fresh port, to force an eviction */
          if (ev[q].t > s && ev[q].t - s < 120000 && rand() % 2) { port = ev[q].port; break; }
        if (mode == 2) port = U + rand() % 1000;
        ev[n++] = (Ev){s + j, port, 1};
      }
    qsort(ev, n, sizeof(Ev), cmp);
    HcTable A = {tabA, 64, 0}, B = {tabB, 64, 0};
    for (int i = 0; i < 64; i++) tabA[i] = tabB[i] = (Host){0};
    double ho[CB_HC], hcl[CB_HC]; Host *hs, *hd, *cs, *cd;
    for (int i = 0; i < n; i++) {
      hc_flow(&A, 11, 22, ev[i].t, 5000 + ev[i].port, 1, 100, ho, &hs, &hd);
      if (ev[i].inj) continue;
      hc_flow(&B, 11, 22, ev[i].t, 5000 + ev[i].port, 1, 100, hcl, &cs, &cd);
      if (ev[i].t < 250000) continue;                         /* age gate: no block decision before 250 s */
      for (int w = 0; w < 2; w++) {                           /* 10 s and 120 s windows, source side */
        double no = ho[w * 2 * NS], nc = hcl[w * 2 * NS];
        double po = ho[w * 2 * NS + 2] * (no > 1 ? no : 1), pc = hcl[w * 2 * NS + 2] * (nc > 1 ? nc : 1);
        checked++;
        double u[4][2]; hc_unc(hs, hd, ev[i].t, u);
        int fl = hc_wide(hs, hd, ev[i].t) & 1; flagged += fl; usum += u[w * 2][0];
        if ((!fl && pc > po + M + 0.5) || pc < po - M - u[w * 2][0] - 0.5) {
          if (viol < 5) printf("VIOL trial %d t=%lld w=%d clean %.0f obs %.0f\n", trial, (long long)ev[i].t, w, pc, po);
          viol++;
        }
      }
    }
  }
  printf("test_bloom: %ld checks, %ld high-side flagged, mean u %.2f, %ld outside [obs-m-u, obs+m(unflagged)]\n",
         checked, flagged, usum / checked, viol);
  return viol != 0;
}
