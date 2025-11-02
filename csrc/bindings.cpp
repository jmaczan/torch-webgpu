#include <Python.h>
#include <vector>

PyMODINIT_FUNC PyInit__C(void)
{
    static std::vector<PyMethodDef> methods;
    static const int python_api_version = 1013;
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "torch_webgpu._C",
        nullptr,
        -1,
        methods.data()};
    PyAPI_FUNC(PyObject *) module = PyModule_Create2(&module_def, python_api_version);
    return module;
}