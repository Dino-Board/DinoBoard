add_library(game_arknova STATIC
    arknova_cards.cpp
    arknova_map.cpp
    arknova_map_rules.cpp
    arknova_engine.cpp
    arknova_effects.cpp
    arknova_action_cards_build.cpp
    arknova_action_animals.cpp
    arknova_action_association.cpp
    arknova_action_sponsors.cpp
    arknova_abilities.cpp
    arknova_card_effects.cpp
    arknova_json.cpp
    arknova_state.cpp
    arknova_actions.cpp
    arknova_rules.cpp
    arknova_belief.cpp
    arknova_encoder.cpp
    arknova_register.cpp
    heuristic/arknova_greedy_heuristic.cpp
)
target_include_directories(game_arknova PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(game_arknova PUBLIC dinoboard_core)

# Line-protocol driver used by tools/diff_test.py to check the engine against the Python reference.
add_executable(arknova_diff_cli tools/diff_cli.cpp)
target_link_libraries(arknova_diff_cli PRIVATE game_arknova)
