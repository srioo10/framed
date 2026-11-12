/* Self-check for the gateway eviction fix (2026-11-12): an injected flow that evicts a live port entry must set the
 * side's eviction flag, and the widened certificate must never be tighter than the plain one.
 * Build: cc -O1 -I engine -I firmware/gateway/main engine/test_evict.c -lm -o /tmp/te && /tmp/te */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "certb.h"
#include "hostctx.h"
#include "model_transfer.h"

int main(void) {
  static Host tab[8]; HcTable T = {tab, 8, 0};
  double hc[CB_HC]; Host *hs, *hd;
  for (int p = 0; p < TAB; p++) hc_flow(&T, 11, 22, 1000 + p, 3000 + p, 1, 100, hc, &hs, &hd);
  assert(hc_wide(hs, hd, 1100) == 0);                   /* 16 ports fit: no eviction yet */
  hc_flow(&T, 11, 22, 2000, 9999, 1, 100, hc, &hs, &hd);  /* 17th port evicts an entry seen 1 s ago */
  assert(hc_wide(hs, hd, 2000) & 1);
  assert(hc_wide(hs, hd, 2000 + 131000) == 0);          /* flag expires after the window + bucket */
  const CertB *c = &CB_MODEL; double x[CB_MAXD]; srand(1); int tighter = 0;
  for (int r = 0; r < 2000; r++) {
    for (int i = 0; i < c->D; i++) x[i] = (rand() / (double)RAND_MAX - 0.5) * 4;
    if (cb_block_lower(c, x, 3) > cb_block_lower(c, x, 0) + 1e-12) tighter++;
  }
  assert(!tighter);
  puts("test_evict ok");
  return 0;
}
