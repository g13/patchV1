SRCDIR=./
#OUTPUTDIR=../singleRun/
OUTPUTDIR=../gainCurve/
CC=g++
MATLAB=-lmat -lmx
PYLIB=-lpython2.7
LBOOST=-lboost_program_options
OPT=-msse4.2 -march=westmere -fPIC -O2
DEBUG=-g -fmax-errors=2 -fno-omit-frame-pointer
CXXFLAG=-std=c++11 $(DEBUG)
OBJ=input_args.o
DEP=matFunc.h typedefs.h heap.h

# to do gain curve
V1: $(OBJ) $(DEP)
	$(CC) $(CXXFLAG) $(LBOOST) $(OBJ) -o $(OUTPUTDIR)$@

%.o : %.cpp %.h
	$(CC) $(CXXFLAG) -c $< -o $@

## singleRun
## sR : nsynBeta.cpp
## 	$(CC) $(CXXFLAG) $^ $(MATLIB) $(PYLIB) $(LBOOST) -o $(OUTPUTDIR)$@
clean:
	rm $(OBJ)
