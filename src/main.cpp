#include <iostream>
#include "Heat3D.hpp"

int main(int argc, char const * argv[])
{
    if (argc != 2) {
        std::cout << "Usage: ./main <path-to-input-mesh>" << std::endl;
        return 1;
    }

    constexpr unsigned int dim = Heat3D::dim;
    const unsigned int r = 1;

    const auto mu  = [](const Point<dim> &/*p*/) {
        return 10.0;
    };

    const auto f  = [](const Point<dim> &/*p*/) {
        return 0.0;
    };

    const auto h  = [](const Point<dim> &/*p*/) {
        return 0.0;
    };

    const double u_d = 5.0;

    Heat3D problem(argv[1], r, mu, f, h, u_d);

    problem.setup();
    problem.assemble();
    problem.solve();
    problem.output();

    return 0;
}
