idf_component_register(SRCS "gateway.c" "ids.c"
                       INCLUDE_DIRS "." "../../../engine"
                       PRIV_REQUIRES esp_wifi nvs_flash esp_timer lwip)
# lwIP calls gw_ip4_input_hook for every IPv4 packet it receives on any interface (CONFIG_LWIP_HOOK_IP4_INPUT_CUSTOM)
idf_component_get_property(lwip lwip COMPONENT_LIB)
target_compile_options(${lwip} PRIVATE "-I${CMAKE_CURRENT_LIST_DIR}")
target_compile_definitions(${lwip} PRIVATE "ESP_IDF_LWIP_HOOK_FILENAME=\"lwip_hook.h\"")
