/* Included by lwipopts.h (ESP_IDF_LWIP_HOOK_FILENAME): every received IPv4 packet goes through the IDS first. */
#pragma once
struct pbuf;
struct netif;
int gw_ip4_input_hook(struct pbuf *p, struct netif *inp);
#define LWIP_HOOK_IP4_INPUT gw_ip4_input_hook
