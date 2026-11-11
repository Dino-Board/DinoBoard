cmake_minimum_required(VERSION 3.19)
project(DinoBoard CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(engine)

# Games are listed in games/manifest.json — single source of truth shared
# with setup.py. To add a game, append an entry there; nothing else changes.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/games/manifest.json" _games_manifest)
string(JSON _games_array GET "${_games_manifest}" games)
string(JSON _games_count LENGTH "${_games_array}")
math(EXPR _games_last "${_games_count} - 1")
foreach(idx RANGE 0 ${_games_last})
  string(JSON _game GET "${_games_array}" ${idx})
  string(JSON _game_id GET "${_game}" id)
  # Optional `enabled` field; default true if absent. enabled=false skips
  # the game from the build entirely (no compile, no register).
  set(_game_enabled "true")
  string(JSON _game_enabled ERROR_VARIABLE _err GET "${_game}" enabled)
  if(_err)
    set(_game_enabled "true")
  endif()
  if(_game_enabled STREQUAL "true" OR _game_enabled STREQUAL "1")
    add_subdirectory(games/${_game_id})
  endif()
endforeach()
