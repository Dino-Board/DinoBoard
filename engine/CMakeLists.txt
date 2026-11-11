cmake_minimum_required(VERSION 3.16)
project(dinoboard_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BOARD_AI_WITH_ONNX "Enable ONNX Runtime inference" OFF)
set(BOARD_AI_ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime installation")

# Core library (header-only interfaces + action_constraint)
add_library(dinoboard_core STATIC
    core/action_constraint.cpp
    core/mask_state_impl.cpp
    core/heuristic_registry.cpp
)
target_include_directories(dinoboard_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Search library
add_library(dinoboard_search STATIC
    search/net_mcts.cpp
    search/tail_solver.cpp
)
target_link_libraries(dinoboard_search PUBLIC dinoboard_core)

# Inference library
add_library(dinoboard_infer STATIC
    infer/onnx_policy_value_evaluator.cpp
    infer/split_policy_value_evaluator.cpp
    infer/batched_policy_value_evaluator.cpp
    infer/batched_belief_evaluator.cpp
    infer/evaluator_pool.cpp
)
target_link_libraries(dinoboard_infer PUBLIC dinoboard_core dinoboard_search)

if(BOARD_AI_WITH_ONNX)
    target_compile_definitions(dinoboard_infer PUBLIC BOARD_AI_WITH_ONNX=1)
    if(BOARD_AI_ONNXRUNTIME_ROOT)
        target_include_directories(dinoboard_infer PUBLIC ${BOARD_AI_ONNXRUNTIME_ROOT}/include)
        target_link_directories(dinoboard_infer PUBLIC ${BOARD_AI_ONNXRUNTIME_ROOT}/lib)
    endif()
    target_link_libraries(dinoboard_infer PUBLIC onnxruntime)
endif()

# Runtime library
add_library(dinoboard_runtime STATIC
    runtime/selfplay_runner.cpp
    runtime/arena_runner.cpp
    runtime/heuristic_runner.cpp
)
target_link_libraries(dinoboard_runtime PUBLIC dinoboard_core dinoboard_search)

# Framework-shared heuristics (random + mcts_pv)
add_library(dinoboard_heuristic STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../heuristic/random_heuristic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../heuristic/mcts_pv_heuristic.cpp
)
target_link_libraries(dinoboard_heuristic PUBLIC
    dinoboard_core
    dinoboard_search
    dinoboard_infer
)
target_include_directories(dinoboard_heuristic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Umbrella target
add_library(dinoboard_engine INTERFACE)
target_link_libraries(dinoboard_engine INTERFACE
    dinoboard_core
    dinoboard_search
    dinoboard_infer
    dinoboard_runtime
    dinoboard_heuristic
)
