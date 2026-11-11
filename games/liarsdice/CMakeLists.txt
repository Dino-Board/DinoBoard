add_library(game_liarsdice STATIC
    liarsdice_state.cpp
    liarsdice_rules.cpp
    outcome_index.cpp
    liarsdice_net_adapter.cpp
    liarsdice_register.cpp
    heuristic/liarsdice_rule_heuristic.cpp
)
target_include_directories(game_liarsdice PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(game_liarsdice PUBLIC dinoboard_core)
