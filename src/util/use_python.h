#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "../types.h"
#include "numpy/arrayobject.h"

int sample_2D_Gaussian(Float nsig, int n, Float wv, Float x[], Float y[], Float w[]);
int sample_2D_Gaussian_difference(Float nsig, Float sigRatio, int n, Float x[], Float y[], Float w[]);
