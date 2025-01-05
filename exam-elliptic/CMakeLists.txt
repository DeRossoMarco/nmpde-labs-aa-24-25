cmake_minimum_required(VERSION 3.12.0)
project(lab05 LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(exam-elliptic src/exam-elliptic.cpp src/Elliptic.cpp)
deal_ii_setup_target(exam-elliptic)