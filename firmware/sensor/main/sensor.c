/* ESP32 #2: MQTT temperature sensor for the OnePass lab (PREREG §9 DEPLOY). Joins the Wi-Fi in menuconfig, publishes
 * {"seq":n,"t_c":x,"up_ms":u} to onepass/sensor/temperature every OP_PERIOD_MS. Credentials live only in sdkconfig
 * (gitignored). The reading is simulated: the classic ESP32 has no usable on-chip temperature sensor.
 * shortcut: simulated reading, replace read_temp() when an external sensor (DHT22 / DS18B20) is wired. */
#include <stdio.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "mqtt_client.h"
#include "nvs_flash.h"
#include "esp_random.h"

static const char *TAG = "sensor";
static EventGroupHandle_t ev;
#define GOT_IP BIT0
#define MQTT_UP BIT1

static void on_wifi(void *arg, esp_event_base_t base, int32_t id, void *data) {
  if (base == WIFI_EVENT && (id == WIFI_EVENT_STA_START || id == WIFI_EVENT_STA_DISCONNECTED)) {
    xEventGroupClearBits(ev, GOT_IP);
    if (id == WIFI_EVENT_STA_DISCONNECTED) {
    wifi_event_sta_disconnected_t *d = (wifi_event_sta_disconnected_t *)data;
    ESP_LOGW(TAG, "wifi down, reason=%d, reconnecting", d->reason);
}
    esp_wifi_connect();
  } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
    ESP_LOGI(TAG, "got ip " IPSTR, IP2STR(&((ip_event_got_ip_t *)data)->ip_info.ip));
    xEventGroupSetBits(ev, GOT_IP);
  }
}

static void on_mqtt(void *arg, esp_event_base_t base, int32_t id, void *data) {
  if (id == MQTT_EVENT_CONNECTED) { ESP_LOGI(TAG, "mqtt connected"); xEventGroupSetBits(ev, MQTT_UP); }
  else if (id == MQTT_EVENT_DISCONNECTED) { ESP_LOGW(TAG, "mqtt disconnected"); xEventGroupClearBits(ev, MQTT_UP); }
  else if (id == MQTT_EVENT_ERROR) ESP_LOGE(TAG, "mqtt error (check broker address / credentials)");
}

static float read_temp(int seq) { return 24.0f + 1.5f * sinf(seq / 60.0f) + (float)(esp_random() % 100) / 500.0f; }

void app_main(void) {
  esp_err_t e = nvs_flash_init();
  if (e == ESP_ERR_NVS_NO_FREE_PAGES || e == ESP_ERR_NVS_NEW_VERSION_FOUND) { nvs_flash_erase(); e = nvs_flash_init(); }
  ESP_ERROR_CHECK(e);
  ev = xEventGroupCreate();
  ESP_ERROR_CHECK(esp_netif_init());
  ESP_ERROR_CHECK(esp_event_loop_create_default());
  esp_netif_create_default_wifi_sta();
  wifi_init_config_t wc = WIFI_INIT_CONFIG_DEFAULT();
  ESP_ERROR_CHECK(esp_wifi_init(&wc));
  ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, on_wifi, NULL));
  ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, on_wifi, NULL));
  wifi_config_t sta = {0};
  snprintf((char *)sta.sta.ssid, sizeof sta.sta.ssid, "%s", CONFIG_OP_WIFI_SSID);
  snprintf((char *)sta.sta.password, sizeof sta.sta.password, "%s", CONFIG_OP_WIFI_PASS);
  sta.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
  ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
  ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta));
  ESP_ERROR_CHECK(esp_wifi_start());
  xEventGroupWaitBits(ev, GOT_IP, pdFALSE, pdTRUE, portMAX_DELAY);

  esp_mqtt_client_config_t mc = {.broker.address.uri = CONFIG_OP_BROKER_URI,
                                 .credentials.username = CONFIG_OP_MQTT_USER,
                                 .credentials.authentication.password = CONFIG_OP_MQTT_PASS};
  esp_mqtt_client_handle_t c = esp_mqtt_client_init(&mc);  // the client reconnects on its own
  esp_mqtt_client_register_event(c, ESP_EVENT_ANY_ID, on_mqtt, NULL);
  esp_mqtt_client_start(c);

  char msg[96];
  for (int seq = 0;; seq++) {
    vTaskDelay(pdMS_TO_TICKS(CONFIG_OP_PERIOD_MS));
    if (!(xEventGroupGetBits(ev) & MQTT_UP)) continue;
    snprintf(msg, sizeof msg, "{\"seq\":%d,\"t_c\":%.2f,\"up_ms\":%lld}", seq, read_temp(seq), esp_timer_get_time() / 1000);
    esp_mqtt_client_publish(c, "onepass/sensor/temperature", msg, 0, 1, 0);
    ESP_LOGI(TAG, "pub %s", msg);
  }
}
