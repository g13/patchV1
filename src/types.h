#ifndef TYPES_H
#define TYPES_H

#include <random>
typedef int Int;

//typedef float Float;
typedef double Float;
//#define SINGLE_PRECISION
typedef double Double;
#define DOUBLE_PRECISION

typedef unsigned int Size;
typedef long unsigned int BigSize;
typedef unsigned int SmallSize;
typedef unsigned int PosInt;
typedef long unsigned int PosIntL;
typedef std::default_random_engine RandomEngine;
typedef Float (*pFeature) (Float, Float, Float);

#define M_PI 3.14159265358979323846  /* pi */

#endif
