cmake_minimum_required(VERSION 3.22.1)

# LC3 decoder for the G1 microphone (liblc3 from the Even Realities demo).
project("lc3")

include_directories(include)
add_library(${CMAKE_PROJECT_NAME} SHARED
        lc3_jni.cpp
        liblc3/attdet.c
        liblc3/bits.c
        liblc3/bwdet.c
        liblc3/energy.c
        liblc3/lc3.c
        liblc3/ltpf.c
        liblc3/mdct.c
        liblc3/plc.c
        liblc3/sns.c
        liblc3/spec.c
        liblc3/tables.c
        liblc3/tns.c)

target_link_libraries(${CMAKE_PROJECT_NAME} android log)
