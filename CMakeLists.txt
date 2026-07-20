cmake_minimum_required(VERSION 4.3.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b") #for CMake 4.3.0
set(CMAKE_CXX_MODULE_STD ON)

project(waferinsp VERSION 1.0 LANGUAGES CXX)

add_compile_options(/W4 /std:c++latest /external:anglebrackets /external:W0)

find_package(OpenCV CONFIG REQUIRED) # 
find_package(GTest CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
include(GoogleTest)  # for testing


add_library(algo SHARED algo.cpp algorithm.h)
target_link_libraries(algo PRIVATE ${OpenCV_LIBS})
target_link_libraries(algo PRIVATE nlohmann_json::nlohmann_json)
add_executable(test_algo test_algo.cpp)
target_link_libraries(test_algo PRIVATE GTest::gtest_main ${OpenCV_LIBS} algo)
gtest_discover_tests(test_algo)

if (WIN32)
  set_target_properties(algo PROPERTIES PREFIX "" SUFFIX ".dll")
endif()
