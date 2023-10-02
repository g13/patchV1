#include "use_python.h"
#include <stdlib.h>
#include <string>

//const wchar_t* path = L".:/root/miniconda3/envs/general/lib/python310.zip:/root/miniconda3/envs/general/lib/python3.10:/root/miniconda3/envs/general/lib/python3.10/lib-dynload:/root/miniconda3/envs/general/lib/python3.10/site-packages";

int sample_2D_Gaussian(Float nsig, int n, Float wv, Float x[], Float y[], Float w[]) {
    //PySys_SetPath(path);  // path to the module to import
    //
    //PyRun_SimpleString("import sys");
    //PyRun_SimpleString("sys.path.insert(0, \"/home/wd/repos/patchV1/src/util\")");
    PyObject *pName = PyUnicode_FromString("sampler");
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    if (pModule != NULL) {
        PyObject *pArg;
        pArg = PyTuple_New(3);
        PyObject *py_nsig = PyFloat_FromDouble(double(nsig));
        PyObject *py_n = PyLong_FromLong(long(n));
        PyObject *py_wv = PyFloat_FromDouble(double(wv));

        PyTuple_SetItem(pArg, 0, py_nsig);
        PyTuple_SetItem(pArg, 1, py_n);
        PyTuple_SetItem(pArg, 2, py_wv);
        PyObject *pFunc = PyObject_GetAttrString(pModule, "sample_2d_gaussian");
        if (pFunc && PyCallable_Check(pFunc)) {
            PyObject *pValue = PyObject_CallObject(pFunc, pArg); 
            if (pValue != NULL) {
                PyArrayObject *py_x = reinterpret_cast<PyArrayObject*>(PyTuple_GetItem(pValue, 0));
                PyArrayObject *py_y = reinterpret_cast<PyArrayObject*>(PyTuple_GetItem(pValue, 1));
                PyArrayObject *py_w = reinterpret_cast<PyArrayObject*>(PyTuple_GetItem(pValue, 2));
                double* d_x = reinterpret_cast<double*>(PyArray_DATA(py_x));
                double* d_y = reinterpret_cast<double*>(PyArray_DATA(py_y));
                double* d_w = reinterpret_cast<double*>(PyArray_DATA(py_w));
                for (int i=0; i<n; i++) {
                    x[i] = static_cast<Float>(d_x[i]);
                    y[i] = static_cast<Float>(d_y[i]);
                    w[i] = static_cast<Float>(d_w[i]);
                    if (i==0 || i == n-1)
                    {
                        printf("x[%i] = %.3f, y[%i] = %.3f, w[%i] = %.3f\n", i, x[i], i, y[i], i, w[i]);
                    }
                }
                Py_DECREF(pValue);
            } else { 
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print(); 
                printf("Call failed\n");
                return 1;
            } 
        } else { 
            if (PyErr_Occurred()) PyErr_Print(); 
            printf("Cannot find function %s\n", "sample_2d_gaussian"); 
            return 1;
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    } else {
        PyErr_Print();
        printf("Failed to load %s\n", "sampler");
        return 1;
    }
    return 0;
}

int sample_2D_Gaussian_difference(Float nsig, Float sigRatio, int n, Float x[], Float y[], Float w[]) {
    return 0;
}
