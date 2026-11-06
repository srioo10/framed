/* Veto IDS on the gateway (PREREG §9 DEPLOY). gw_ip4_input_hook runs in lwIP's thread for every received IPv4 packet:
 * it drops packets from blocked sources and queues a 20-byte header record; the ids task meters bidirectional flows
 * with exactly the rules of scripts/live/flowmeter.py, then per finished flow computes the 20 host-context features
 * (engine/hostctx.h, keyed by the flow's close time), B's score and, on an alarm, the certified lower bound
 * (engine/certb.h). Veto blocks the flow's initiator only when the lower bound clears tau; GW_ORDINARY blocks on
 * every alarm. One CSV line per flow on the console (F,...), one per block (B,...), stats every 10 s (S,...). */
#include <stdio.h>
#include <string.h>
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "lwip/pbuf.h"
#include "lwip/ip4_addr.h"
#include "sdkconfig.h"

#ifdef CONFIG_GW_ORDINARY
#define ORDINARY 1
#else
#define ORDINARY 0
#endif
#ifdef CONFIG_GW_MONITOR
#define MONITOR 1
#else
#define MONITOR 0
#endif
#include <stdlib.h>
static double TAU;  /* CONFIG_GW_TAU if set (PREREG tau_lab), else the model's tau */

#include "certb.h"
#include "hostctx.h"
#if CONFIG_GW_MODEL_LAB
#include "model_lab.h"
#else
#include "model_transfer.h"
#endif

#define IDLE_MS 30000    /* = flowmeter.IDLE_S */
#define ACTIVE_MS 120000 /* = flowmeter.ACTIVE_S */
#define NFLOW 256
#define NBLK 16
#define AGE_MS 130000    /* certids: no block while a host entry is younger than one 120 s window + bucket */

typedef struct { int64_t t; uint32_t src, dst; uint16_t sport, dport, len; uint8_t proto, flags; } Pkt;
typedef struct { uint8_t used, proto, fin; uint8_t flags; uint32_t cip, sip; uint16_t cport, sport;
                 int64_t start, last; uint32_t ib, ob, ipk, opk; } Flow;

static QueueHandle_t q;
static volatile uint32_t qdrop, bdrop, npk, nshed;
#ifdef CONFIG_GW_SHED
/* SHED (PREREG 2026-11-06): under saturation, drop packets of sources that opened more than GW_SHED_C flows in the
 * current 10 s. Such a source is far over the certified budget, where frame-proofness never applied; within-rate
 * sources keep the queue. A distributed flood (many sources each under C) still saturates: disclosed, not fixed. */
#define NSRC 32
static uint32_t sh_ip[NSRC], sh_n[NSRC];
static int64_t sh_t0[NSRC];
static int shed(uint32_t src, int64_t t, int opens) {
  if (!opens) return 0;  /* only new-connection packets (SYN / connectionless) drive shedding; return traffic
    (RST, SYN-ACK) never resets a counter and is never shed, so an over-rate source sharing a hash slot with the
    always-active upstream cannot thrash the slot to zero. */
  int i = (src * 2654435761u) >> (32 - 5);  /* Knuth multiplicative hash -> 5-bit slot (NSRC=32); mixes all octets
    so lab /24 addresses (all first-octet 192) do not collapse to one slot. shortcut: a same-slot collision between
    two SYN-sending sources resets the counter (errs toward not shedding). Keep NSRC a power of two = 32. */
  if (sh_ip[i] != src || t - sh_t0[i] >= 10000) { sh_ip[i] = src; sh_t0[i] = t; sh_n[i] = 0; }
  sh_n[i] += opens;
  return sh_n[i] > CONFIG_GW_SHED_C && uxQueueMessagesWaiting(q) > 256;
}
#endif
static volatile uint32_t blk_ip[NBLK];
static volatile int64_t blk_until[NBLK];
static Flow fl[NFLOW];
static Host hosts[CONFIG_GW_HOSTS];
static HcTable T = {hosts, CONFIG_GW_HOSTS, 0};
static uint32_t nflow, nalarm, nblock, nabst, nevict_flow;
static uint32_t self_ip[2];

static inline int64_t now_ms(void) { return esp_timer_get_time() / 1000; }

int gw_ip4_input_hook(struct pbuf *p, struct netif *inp) {
  if (p->len < 20) return 0;
  const uint8_t *h = p->payload;
  if ((h[0] >> 4) != 4) return 0;
  Pkt k = {0};
  memcpy(&k.src, h + 12, 4);
  memcpy(&k.dst, h + 16, 4);
  k.t = now_ms();
  for (int i = 0; i < NBLK; i++)
    if (blk_ip[i] == k.src && blk_until[i] > k.t) { bdrop++; pbuf_free(p); return 1; }  /* eaten: lwIP stops here */
  uint8_t d0 = ((const uint8_t *)&k.dst)[0];
  if (k.dst == self_ip[0] || k.dst == self_ip[1] || k.dst == 0xffffffff || (d0 & 0xf0) == 0xe0 || k.src == 0)
    return 0;  /* traffic to the gateway itself, broadcast, multicast: not metered (the Pi cannot see it either) */
  int ihl = (h[0] & 15) * 4, frag = ((h[6] & 0x1f) << 8) | h[7];
  k.len = (h[2] << 8) | h[3];
  k.proto = h[9];
  if (!frag && (k.proto == 6 || k.proto == 17) && p->len >= ihl + 4) {
    k.sport = (h[ihl] << 8) | h[ihl + 1];
    k.dport = (h[ihl + 2] << 8) | h[ihl + 3];
    if (k.proto == 6 && p->len >= ihl + 14) k.flags = h[ihl + 13];
  }
  npk++;
#ifdef CONFIG_GW_SHED
  if (shed(k.src, k.t, k.proto != 6 || (k.flags & 0x12) == 0x02)) { nshed++; pbuf_free(p); return 1; }
#endif
  if (xQueueSend(q, &k, 0) != pdTRUE) qdrop++;
  return 0;
}

static void ipstr(uint32_t a, char *s) {
  const uint8_t *b = (const uint8_t *)&a;
  sprintf(s, "%u.%u.%u.%u", b[0], b[1], b[2], b[3]);
}

static void block(uint32_t ip, int64_t t) {
  int slot = 0;
  for (int i = 0; i < NBLK; i++) {
    if (blk_ip[i] == ip) { slot = i; break; }
    if (blk_until[i] < blk_until[slot]) slot = i;
  }
  blk_until[slot] = t + (int64_t)CONFIG_GW_BLOCK_TTL_S * 1000;
  blk_ip[slot] = ip;
  char s[16]; ipstr(ip, s);
  printf("B,%lld,%s,%d\n", t, s, CONFIG_GW_BLOCK_TTL_S);
}

/* one finished flow -> features -> B -> decision (tc = close time, the host-context clock) */
static void decide(Flow *f, int64_t tc, const char *why) {
  const CertB *c = &CB_MODEL;
  char cs[16], ss[16];
  ipstr(f->cip, cs); ipstr(f->sip, ss);
  double raw[CB_MAXD], x[CB_MAXD];
  /* model flow-feature order: L4_DST_PORT, PROTOCOL, FLOW_DURATION_MILLISECONDS, IN_BYTES, OUT_BYTES, IN_PKTS, OUT_PKTS */
  raw[0] = f->sport; raw[1] = f->proto; raw[2] = (double)(f->last - f->start);
  raw[3] = f->ib; raw[4] = f->ob; raw[5] = f->ipk; raw[6] = f->opk;
  Host *hs, *hd;
  int64_t t0 = esp_timer_get_time();
#ifdef CONFIG_GW_KEY_PREFIX24
  char kc[16], ks[16];  /* host key = a.b.c.0: every client in the /24 shares one context (PREREG LIVE-SK) */
  ipstr(f->cip & 0x00ffffffu, kc); ipstr(f->sip & 0x00ffffffu, ks);
#else
  const char *kc = cs, *ks = ss;
#endif
  hc_flow(&T, hc_fnv(kc), hc_fnv(ks), tc, f->sport, f->opk, f->ib, raw + c->F, &hs, &hd);
  cb_prep(c, raw, x);
  double s = cb_score(c, x), lower = -INFINITY;
  int alarm = s > TAU, blk = 0;
  if (alarm) {
    nalarm++;
#if ORDINARY
    blk = 1;
#else
    lower = cb_block_lower(c, x);
    blk = lower > TAU;
    if (blk && (tc - hs->created < AGE_MS || tc - hd->created < AGE_MS)) { blk = 0; nabst++; }
#endif
  }
  uint32_t us = (uint32_t)(esp_timer_get_time() - t0);
  nflow++;
  printf("F,%lld,%s,%u,%s,%u,%u,%lu,%lu,%lu,%lu,%lld,%.6g,%.6g,%d,%d,%lu,%s\n", tc, cs, f->cport, ss, f->sport, f->proto,
         (unsigned long)f->ib, (unsigned long)f->ipk, (unsigned long)f->ob, (unsigned long)f->opk, f->last - f->start,
         s, lower, alarm, blk, (unsigned long)us, why);
  if (MONITOR) blk = 0;
  if (blk) { nblock++; block(f->cip, tc); }
  f->used = 0;
}

static Flow *find(const Pkt *k, int *fwd) {
  for (int i = 0; i < NFLOW; i++) {
    Flow *f = &fl[i];
    if (!f->used || f->proto != k->proto) continue;
    if (f->cip == k->src && f->cport == k->sport && f->sip == k->dst && f->sport == k->dport) { *fwd = 1; return f; }
    if (f->cip == k->dst && f->cport == k->dport && f->sip == k->src && f->sport == k->sport) { *fwd = 0; return f; }
  }
  return NULL;
}

static void meter(const Pkt *k) {
  int fwd = 1;
  Flow *f = find(k, &fwd);
  if (f && k->t - f->last > IDLE_MS) { decide(f, f->last + IDLE_MS, "idle"); f = NULL; }
  else if (f && k->t - f->start > ACTIVE_MS) { decide(f, k->t, "active"); f = NULL; }
  if (!f) {
    Flow *old = NULL;
    for (int i = 0; i < NFLOW && !f; i++) {
      if (!fl[i].used) f = &fl[i];
      else if (!old || fl[i].last < old->last) old = &fl[i];
    }
    if (!f) { nevict_flow++; decide(old, k->t, "evict"); f = old; }  /* table full: oldest flow ends now */
    memset(f, 0, sizeof *f);
    int cli_is_src = 1;
    if (k->proto == 6 && !((k->flags & 0x02) && !(k->flags & 0x10))) cli_is_src = k->sport >= k->dport;
    f->used = 1; f->proto = k->proto; f->start = f->last = k->t;
    f->cip = cli_is_src ? k->src : k->dst; f->cport = cli_is_src ? k->sport : k->dport;
    f->sip = cli_is_src ? k->dst : k->src; f->sport = cli_is_src ? k->dport : k->sport;
    fwd = cli_is_src;
  }
  if (fwd) { f->ib += k->len; f->ipk++; } else { f->ob += k->len; f->opk++; }
  f->flags |= k->flags;
  f->last = k->t;
  if (k->proto == 6) {
    if (k->flags & 0x01) f->fin |= fwd ? 1 : 2;
    if (k->flags & 0x04) decide(f, k->t, "rst");
    else if (f->fin == 3) decide(f, k->t, "fin");
  }
}

static void sweep(int64_t t) {
  for (int i = 0; i < NFLOW; i++)
    if (fl[i].used && t - fl[i].last > IDLE_MS) decide(&fl[i], fl[i].last + IDLE_MS, "idle");
}

static void ids_task(void *arg) {
  Pkt k;
  int64_t next_sweep = 0, next_stat = 0;
  for (;;) {
    if (xQueueReceive(q, &k, pdMS_TO_TICKS(100)) == pdTRUE) meter(&k);
    int64_t t = now_ms();
    if (t >= next_sweep) { sweep(t); next_sweep = t + 100; }
    if (t >= next_stat) {
      int open = 0;
      for (int i = 0; i < NFLOW; i++) open += fl[i].used;
      printf("S,%lld,%u,%u,%lu,%lu,%lu,%d,%ld,%lu,%lu,%lu,%lu,%lu\n", t, (unsigned)esp_get_free_heap_size(),
             (unsigned)esp_get_minimum_free_heap_size(), (unsigned long)npk, (unsigned long)qdrop,
             (unsigned long)bdrop, open, T.evictions, (unsigned long)nevict_flow, (unsigned long)nflow,
             (unsigned long)nalarm, (unsigned long)nblock, (unsigned long)nabst);
      printf("Q,%lld,%lu,%lu\n", t, (unsigned long)qdrop, (unsigned long)nshed);
      next_stat = t + 10000;
    }
  }
}

void ids_start(void) {
  TAU = CONFIG_GW_TAU[0] ? strtod(CONFIG_GW_TAU, NULL) : CB_MODEL.tau;
  self_ip[0] = ipaddr_addr(CONFIG_GW_UP_IP);
  self_ip[1] = ipaddr_addr("192.168.4.1");
  q = xQueueCreate(512, sizeof(Pkt));
  printf("I,veto ids: %s%s, model D=%d M=%d W=%d tau=%.6g (model %.6g) m=%d, %d host slots (%u B), %d flow slots\n",
         ORDINARY ? "ORDINARY (block on alarm)" : "VETO (certified block)", MONITOR ? " MONITOR-ONLY" : "",
         CB_MODEL.D, CB_MODEL.M, CB_MODEL.W, TAU, CB_MODEL.tau, CB_MODEL.m_budget, CONFIG_GW_HOSTS, (unsigned)sizeof hosts, NFLOW);
  printf("I,columns F: t_close,src,sport,dst,dport,proto,in_bytes,in_pkts,out_bytes,out_pkts,dur_ms,score,lower,alarm,"
         "block,us,end\n");
  xTaskCreatePinnedToCore(ids_task, "ids", 12288, NULL, 5, NULL, 1);
}
