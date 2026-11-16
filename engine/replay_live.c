/* engine/replay_live.c: replay logged gateway flows through the deployed decision path (ids.c decide()), so the live
 * block decisions can be checked without hardware. Build per gateway build:
 *   cc -O1 -I engine -I firmware/gateway/main engine/replay_live.c -lm                       (fair_veto: margin build)
 *   ... -DGW_SOUND -DHC_FLAG_MS=250000                                                      (fair_sound)
 *   ... -DGW_SOUND -DHC_FLAG_MS=250000 -DHC_BLOOM                                           (fair_scan, pos)
 *   ... -DORDINARY                                                                          (fair_ord)
 * stdin: one flow per line, tc,src,sport,dst,dport,proto,ib,ipk,ob,opk,dur (time order, one run from a gateway reset).
 * argv[1] = tau. stdout: tc,src,dst,dport,score,lower,alarm,block,abstain,host_evictions. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "certb.h"
#include "hostctx.h"
#include "model_transfer.h"
#define AGE_MS 250000 /* = ids.c */
static Host hosts[32]; /* CONFIG_GW_HOSTS = 32 in every live build */
static HcTable T = {hosts, 32, 0};
int main(int argc, char **argv) {
  const CertB *c = &CB_MODEL;
  double tau = argc > 1 ? atof(argv[1]) : 0.442;
  char ln[512], cs[32], ss[32];
  long long tc, dur; unsigned cport, sport, proto; unsigned long ib, ipk, ob, opk;
  while (fgets(ln, sizeof ln, stdin)) {
    if (sscanf(ln, "%lld,%31[^,],%u,%31[^,],%u,%u,%lu,%lu,%lu,%lu,%lld", &tc, cs, &cport, ss, &sport, &proto, &ib, &ipk, &ob, &opk, &dur) != 11) continue;
    double raw[CB_MAXD], x[CB_MAXD];
    raw[0] = sport; raw[1] = proto; raw[2] = (double)dur; raw[3] = ib; raw[4] = ob; raw[5] = ipk; raw[6] = opk;
    Host *hs, *hd;
    hc_flow(&T, hc_fnv(cs), hc_fnv(ss), tc, sport, opk, ib, raw + c->F, &hs, &hd);
    cb_prep(c, raw, x);
    double s = cb_score(c, x), lower = -INFINITY;
    int alarm = s > tau, blk = 0, abst = 0;
    if (alarm) {
#ifdef ORDINARY
      blk = 1;
#else
#if defined(GW_SOUND) && defined(HC_BLOOM)
      double u[4][2]; hc_unc(hs, hd, tc, u);
      lower = cb_block_lower_unc(c, x, hc_wide(hs, hd, tc), u);
#elif defined(GW_SOUND)
      lower = cb_block_lower_sound(c, x, hc_wide(hs, hd, tc));
#else
      lower = cb_block_lower(c, x, hc_wide(hs, hd, tc));
#endif
      blk = lower > tau;
      if (blk && (tc - hs->created < AGE_MS || tc - hd->created < AGE_MS)) blk = 0, abst = 1;
#endif
    }
    printf("%lld,%s,%s,%u,%.6g,%.6g,%d,%d,%d,%ld\n", tc, cs, ss, sport, s, lower, alarm, blk, abst, T.evictions);
  }
  return 0;
}
