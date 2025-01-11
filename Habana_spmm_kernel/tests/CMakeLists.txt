# tests/CMakeLists.txt

include(ExternalProject)

# Required for 'ExternalProject_Add' command
set(TEST_TARGET tpc_kernel_tests)
set(SPARSE_TEST_TARGET sparse_matrix_test)

# Include Directories
include_directories(
    ../src/gaudi_src
    ../src/gaudi2_src
    ../src
    ./gaudi_test
    ./gaudi2_test
    /usr/lib/habanatools/include
    .
)

# Select files to compile
file(GLOB ALL_TESTS_FILES *.cpp gaudi_test/*.cpp gaudi2_test/*.cpp)

# Exclude sparse_matrix_mul_fwd_f32_test.cpp from the combined executable
# list(REMOVE_ITEM ALL_TESTS_FILES ${CMAKE_CURRENT_SOURCE_DIR}/sparse_matrix_mul_fwd_f32_test.cpp)

# Create test executable for all tests except sparse_matrix_mul_fwd_f32_test.cpp
add_executable(${TEST_TARGET} ${ALL_TESTS_FILES})

# Create a separate executable for sparse_matrix_mul_fwd_f32_test.cpp
# add_executable(${SPARSE_TEST_TARGET} /root/Habana_backup/Habana_Custom_Kernel/tests/gaudi_test/sparse_matrix_mul_fwd_f32_test.cpp)

# Link libraries for the combined tests
target_link_libraries(
    ${TEST_TARGET}
    rt
    pthread
    ${TPC_TEST_CORE_LIBRARY}
    ${TPCSIM_SHARED_LIBRARY}
    ${TPC_ELF_READER_LIBRARY}
    custom_tpc_perf_lib
)

# Link libraries for the sparse_matrix_mul_fwd_f32_test.cpp target
# target_link_libraries(
#     ${SPARSE_TEST_TARGET}
#     rt
#     pthread
#     ${TPC_TEST_CORE_LIBRARY}
#     ${TPCSIM_SHARED_LIBRARY}
#     ${TPC_ELF_READER_LIBRARY}
#     custom_tpc_perf_lib
# )
