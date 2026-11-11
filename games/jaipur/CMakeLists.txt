add_library(game_jaipur STATIC
    jaipur_state.cpp
    jaipur_rules.cpp
    jaipur_net_adapter.cpp
    jaipur_belief_tracker.cpp
    jaipur_register.cpp
    heuristic/jaipur_rule_heuristic.cpp
    heuristic/jaipur_heuristic_pv_evaluator.cpp
    heuristic/jaipur_smart_heuristic.cpp
)
target_include_directories(game_jaipur PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_link_libraries(game_jaipur PUBLIC dinoboard_engine)
