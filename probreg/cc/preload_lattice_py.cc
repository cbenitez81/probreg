#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include "permutohedral.h"
#include "permutohedral_preload_filter.h"
#include "types.h"

namespace py = pybind11;
using namespace probreg;

PYBIND11_MODULE(_preload_lattice, m) {
    // Permutohedral_preload(int N,int M,int d, bool with_blur);
	// void init ( const MatrixXf & features, const MatrixXf &in, bool with_blur = true );
    // void init_with_val(const MatrixXf& feature,const MatrixXf& in, bool with_blur);
    // void apply(float* out,const MatrixXf& feature);
    // void apply(float* out,const MatrixXf& feature);
    py::class_<Permutohedral_preload>(m, "Permutohedral_p")
        .def(py::init<int,int,int,bool>())
        /*.def("init", [](Permutohedral_preload &ph, int N,int M,int d, bool with_blur) {
            ph = new Permutohedral_preload(N,M,d,with_blur);
        }
        )*/
        .def("init_with_val", [] (Permutohedral_preload& ph,const MatrixXf & features, const MatrixXf &in, bool with_blur = true) {
            ph.init_with_val(features,in,with_blur);
        }
        )
        .def("apply", [](Permutohedral_preload& ph, float* out,const MatrixXf& feature) {
            ph.apply(out, feature);
            return out;
        });


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}