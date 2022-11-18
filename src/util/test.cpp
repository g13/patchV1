#include "use_python.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("need at least 2 args\n");
        return 1;
    }
    Py_Initialize();
    if (_import_array() < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return 1;
    }
    float nsig = atof(argv[1]);
    int n = atoi(argv[2]);
    float wv;
    if (argc > 3) {
        wv = atof(argv[3]);
    } else {
        wv = 1;
    }
    printf("nsig = %.3f, n = %d, wv = %.3f\n", nsig, n, wv);

    float *x, *y, *w;
    x = new float[n];
    y = new float[n];
    w = new float[n];
    sample_2D_Gaussian(nsig, n, wv, x, y, w);
    Py_Finalize();
    printf("(%.3f, %.3f): %.3f\n", x[0], y[0], w[0]);
    printf("(%.3f, %.3f): %.3f\n", x[n-1], y[n-1], w[n-1]);
    delete []x;
    delete []y;
    delete []w;
    return 0;
}
