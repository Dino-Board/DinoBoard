add_library(game_maskmen STATIC
    maskmen_state.cpp
    maskmen_rules.cpp
    maskmen_net_adapter.cpp
    maskmen_register.cpp
)
target_include_directories(game_maskmen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(game_maskmen PUBLIC dinoboard_core)
