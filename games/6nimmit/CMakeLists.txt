add_library(game_6nimmit STATIC
    6nimmit_state.cpp
    6nimmit_rules.cpp
    6nimmit_net_adapter.cpp
    6nimmit_register.cpp
    heuristic/sixnimmit_rule_heuristic.cpp
    heuristic/sixnimmit_lkf_heuristic.cpp
)
target_include_directories(game_6nimmit PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(game_6nimmit PUBLIC dinoboard_core)
