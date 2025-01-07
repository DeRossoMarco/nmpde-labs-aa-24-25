cmake_minimum_required(VERSION 3.12.0)
project(lab05 LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(exam-vectorial src/exam-vectorial.cpp src/Vectorial.cpp)
deal_ii_setup_target(exam-vectorial)