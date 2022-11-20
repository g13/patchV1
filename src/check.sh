#!/bin/bash

optflag=O2
#optflag=g
#gptflag=G
gptflag=lineinfo
cpp_version=c++17

sm=sm_86

rm -f check.o 
nvcc -arch=$sm -lcudadevrt -lcudart -$gptflag -std=$cpp_version -$optflag check.cu -o $HOME/bin/check
