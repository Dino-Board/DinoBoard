add_library(game_harmonies STATIC
    harmonies_state.cpp
    harmonies_rules.cpp
    harmonies_net_adapter.cpp
    harmonies_register.cpp
    heuristic/harmonies_greedy_heuristic.cpp
)
target_include_directories(game_harmonies PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(game_harmonies PUBLIC dinoboard_core)
