#include <iostream>
#include "Heat3D.hpp"
#include "HeatEquation.hpp"

int main(int argc, char const *argv[])
{
    double T=0.5, dt=0.01, mu=1.0, uD=5.0;
    std::string mesh_filename = "../mesh.msh";

    if (argc == 5) {
        uD = atof(argv[1]);
        T = atof(argv[2]);
        dt = atof(argv[3]);
        mu = atof(argv[4]);
        mesh_filename = argv[5];
    }

    HeatEquation<3> problem(uD, T, dt, mu, mesh_filename);

    problem.run();

    return 0;
}
