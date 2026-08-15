#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(_bundled, module) {
    module.attr("plugins") = nb::make_tuple("reference");
}
