cmake_minimum_required(VERSION 4.3.0)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.4.0")
    set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
elseif(CMAKE_VERSION VERSION_GREATER_EQUAL "4.3.0")
    set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b")
endif()
set(CMAKE_CXX_MODULE_STD ON)

project(waferinsp VERSION 1.0 LANGUAGES CXX)
 
if(MSVC)    
    add_compile_options(/W4 /std:c++latest /external:anglebrackets /external:W0)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")    
    add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-unknown-pragmas -Wno-missing-field-initializers -O3) 
endif()
find_package(OpenCV CONFIG REQUIRED) # 
find_package(GTest CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
include(GoogleTest)  # for testing 

add_library(algodll SHARED algodll.cpp algorithm.h io.h io.cpp)
target_link_libraries(algodll PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
target_link_libraries(algodll PRIVATE nlohmann_json::nlohmann_json)

add_executable(test_algodll test_algodll.cpp io.h io.cpp)
target_link_libraries(test_algodll PRIVATE GTest::gtest_main ${OpenCV_LIBS} algodll)
gtest_discover_tests(test_algodll)
add_executable(test_algocore test_algocore.cpp algocore.cpp algocore.h)
target_link_libraries(test_algocore PRIVATE GTest::gtest_main ${OpenCV_LIBS})
gtest_discover_tests(test_algocore)

if (WIN32)
  set_target_properties(algodll PROPERTIES PREFIX "" SUFFIX ".dll")
endif()
