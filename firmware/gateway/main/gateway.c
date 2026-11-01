/* Veto gateway (ESP32 #2, PREREG §9 DEPLOY): SoftAP for the IoT side + station on the Pi's lab Wi-Fi, plain IPv4
 * routing between them (no NAPT, so the Pi sees real host addresses; the Pi routes 192.168.4.0/24 via GW_UP_IP).
 * Every packet lwIP receives passes gw_ip4_input_hook (ids.c) first. Credentials live only in sdkconfig (gitignored). */
#include <stdio.h>
#include <string.h>
#include "esp_event.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lwip/ip4_addr.h"
#include "nvs_flash.h"

void ids_start(void);
static const char *TAG = "gw";

static void on_wifi(void *arg, esp_event_base_t base, int32_t id, void *data) {
  if (base == WIFI_EVENT && (id == WIFI_EVENT_STA_START || id == WIFI_EVENT_STA_DISCONNECTED)) {
    if (id == WIFI_EVENT_STA_DISCONNECTED) printf("I,upstream down, reconnecting\n");
    esp_wifi_connect();
  } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_CONNECTED) {
    printf("I,upstream connected\n");
  } else if (base == WIFI_EVENT && id == WIFI_EVENT_AP_STACONNECTED) {
    wifi_event_ap_staconnected_t *e = data;
    printf("I,iot client joined " MACSTR "\n", MAC2STR(e->mac));
  }
}

void app_main(void) {
  esp_err_t e = nvs_flash_init();
  if (e == ESP_ERR_NVS_NO_FREE_PAGES || e == ESP_ERR_NVS_NEW_VERSION_FOUND) { nvs_flash_erase(); e = nvs_flash_init(); }
  ESP_ERROR_CHECK(e);
  ESP_ERROR_CHECK(esp_netif_init());
  ESP_ERROR_CHECK(esp_event_loop_create_default());
  esp_netif_t *sta = esp_netif_create_default_wifi_sta();
  esp_netif_create_default_wifi_ap();  // 192.168.4.1/24 with its DHCP server

  ESP_ERROR_CHECK(esp_netif_dhcpc_stop(sta));
  esp_netif_ip_info_t ip = {0};
  ip.ip.addr = ipaddr_addr(CONFIG_GW_UP_IP);
  ip.gw.addr = ipaddr_addr(CONFIG_GW_UP_GW);
  ip.netmask.addr = ipaddr_addr("255.255.255.0");
  ESP_ERROR_CHECK(esp_netif_set_ip_info(sta, &ip));
  esp_netif_set_default_netif(sta);

  wifi_init_config_t wc = WIFI_INIT_CONFIG_DEFAULT();
  ESP_ERROR_CHECK(esp_wifi_init(&wc));
  ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, on_wifi, NULL));
  wifi_config_t up = {0}, ap = {0};
  snprintf((char *)up.sta.ssid, sizeof up.sta.ssid, "%s", CONFIG_GW_UP_SSID);
  snprintf((char *)up.sta.password, sizeof up.sta.password, "%s", CONFIG_GW_UP_PASS);
  up.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
  snprintf((char *)ap.ap.ssid, sizeof ap.ap.ssid, "%s", CONFIG_GW_AP_SSID);
  snprintf((char *)ap.ap.password, sizeof ap.ap.password, "%s", CONFIG_GW_AP_PASS);
  ap.ap.ssid_len = strlen(CONFIG_GW_AP_SSID);
  ap.ap.channel = CONFIG_GW_CHANNEL;
  ap.ap.max_connection = 4;
  ap.ap.authmode = WIFI_AUTH_WPA2_PSK;
  ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_APSTA));
  ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &up));
  ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap));
  ids_start();  // before Wi-Fi starts, so no packet passes unseen
  ESP_ERROR_CHECK(esp_wifi_start());
  ESP_LOGW(TAG, "gateway up: %s <-> %s", CONFIG_GW_AP_SSID, CONFIG_GW_UP_SSID);
}
