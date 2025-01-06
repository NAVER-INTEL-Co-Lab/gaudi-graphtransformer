include(ExternalProject)

#required for 'ExternalProject_Add' command
set(TEST_TARGET tpc_kernel_tests)

include_directories(../src/gaudi_src                          # include headers in src
                    ../src/gaudi2_src
                    ../src
                    ./gaudi_test
                    ./gaudi2_test
                    /usr/lib/habanatools/include
                    .
                    )

# Select files to compile
file(GLOB TESTS_FILES *.cpp gaudi_test/*.cpp gaudi2_test/*.cpp)

# Create test executable
add_executable(${TEST_TARGET} ${TESTS_FILES})

# link with test
target_link_libraries(
    ${TEST_TARGET}
    rt
    pthread
    ${TPC_TEST_CORE_LIBRARY} 
    ${TPCSIM_SHARED_LIBRARY}
    ${TPC_ELF_READER_LIBRARY}
    custom_tpc_perf_lib)
