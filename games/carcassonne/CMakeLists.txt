add_library(game_carcassonne STATIC
    carcassonne_state.cpp
    carcassonne_tiles.cpp
    carcassonne_rules.cpp
    carcassonne_net_adapter.cpp
    carcassonne_register.cpp
    heuristic/carcassonne_rule_heuristic.cpp
    heuristic/carcassonne_heuristic_pv_evaluator.cpp
)
target_include_directories(game_carcassonne PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(game_carcassonne PUBLIC dinoboard_core)
