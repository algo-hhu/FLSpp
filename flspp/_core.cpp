#include <Python.h>

#include "cpp/random_generator.h"
#include "cpp/clustering_algorithm.h"
#include "cpp/additional_vector_stuff.h"
#include "cpp/makros.h"

typedef unsigned int uint;

std::vector<Point> array_to_vector(double *array, double *weight, int rows, int columns)
{

    std::vector<Point> points;

    for (int i = 0; i < rows; ++i)
    {
        std::vector<double> row;
        for (int j = 0; j < columns; ++j)
        {
            row.push_back(array[i * columns + j]);
        }
        Point p = Point(columns, i, weight[i], row);
        points.push_back(p);
    }

    return points;
}

// Thank you https://github.com/dstein64/kmeans1d!

extern "C"
{
// "__declspec(dllexport)" causes the function to be exported when compiling on Windows.
// Otherwise, the function is not exported and the code raises
//   "AttributeError: function 'cluster' not found".
// Exporting is a Windows platform requirement, not just a Visual Studio requirement
// (https://stackoverflow.com/a/22288874/1509433). The _WIN32 macro covers the Visual
// Studio compiler (MSVC) and MinGW. The __CYGWIN__ macro covers gcc and clang under
// Cygwin.
#if defined(_WIN32) || defined(__CYGWIN__)
    __declspec(dllexport)
#endif
    double
    cluster(double *array,
            double *weights,
            uint n,
            uint d,
            uint k,
            uint lloyd_iterations,
            uint local_search_iterations,
            std::uint64_t seed,
            int *labels,
            double *centers,
            int *iterations)
    {

        /*
         * double** array: input points (as an array of coordinate arrays)
         * int n: number of input points
         * int d: dimension/number of features
         * int k: number of centers
         * int lloyd_iterations: how many iterations of lloyd's algorithm are performed (choose -1 for precision-based termination)
         * int local_search_iterations: how many local search steps /center swaps are performed
         * int* cluster: array of length n, which will eventually contain the labels for each point (i.e. cluster membership)
         * double** centers: (k x d)-array, which will eventually contain coordinates of final centers
         */

        // transform input array to vector of "Points" (to be able to use existing constructor for FLSPP object)
        std::vector<Point> points = array_to_vector(array, weights, n, d);

        // create FLSPP object from vector of points
        FLSPP my_flspp(points, -1, seed, lloyd_iterations, local_search_iterations);

        // output_algorithm object contains information such as final labels and final centers
        output_algorithm flspp_output = my_flspp.algorithm(k);

        memcpy(labels, flspp_output.final_labels.data(), n * sizeof(int));

        // flspp_output has vector<Point> attribute final_centers, containing the final centers as instances of class Point
        for (uint j = 0; j < k; ++j)
        {
            // get coordinates of j-th center
            std::vector<double> center_coordinates = flspp_output.final_centers[j].coordinates;
            // We save it in a continuous array
            for (uint l = 0; l < d; ++l)
            {
                centers[j * d + l] = center_coordinates[l];
                // std::cout << centers[j][l] << " ";
            }
            // std::cout << std::endl;
        }

        *iterations = flspp_output.iterations;

        return flspp_output.cost;
    }
} // extern "C"

static PyMethodDef module_methods[] = {
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef _coremodule = {
    PyModuleDef_HEAD_INIT,
    "flspp._core",
    NULL,
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit__core(void)
{
    return PyModule_Create(&_coremodule);
}
