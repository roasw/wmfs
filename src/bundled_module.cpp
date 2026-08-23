#include <nanobind/nanobind.h>

#include <sstream>
#include <string>

namespace nb = nanobind;

NB_MODULE(_bundled, module) {
    nb::list plugins;
    std::istringstream names(WMFS_BUNDLED_PLUGIN_NAMES);
    for (std::string name; std::getline(names, name, ',');) {
        if (!name.empty())
            plugins.append(nb::str(name.c_str()));
    }
    module.attr("plugins") = nb::tuple(plugins);
}
