# Tests CMakeLists (MUSA)
# Only test_encodings is included -- it does not depend on FullyFusedMLP or NVRTC.

set(TCNN_TEST_SOURCES
    test_encodings.mu
    # test_networks.mu excluded: depends on CutlassMLP/FullyFusedMLP
    # test_jit_losses.mu excluded: depends on NVRTC
    # test_grid.mu included conditionally below
)

# test_grid does not use wmma or nvrtc -- include it
list(APPEND TCNN_TEST_SOURCES test_grid.mu)

foreach(test_src IN LISTS TCNN_TEST_SOURCES)
    get_filename_component(test_name "${test_src}" NAME_WE)
    musa_add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} tiny-cuda-nn tcnn-rtc-stub)
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
