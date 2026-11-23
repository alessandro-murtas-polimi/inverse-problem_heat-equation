#pragma once

#include <deal.II/base/utilities.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <iostream>

#include "RightHandSide.hpp"

using namespace dealii;

template <int dim>
class HeatEquation
{
public:
    HeatEquation(double uD_, double T_, double dt_, double mu_, std::string mesh_filename_) : fe(1),
                     dof_handler(triangulation),
                     dt(dt_),
                     T(T_),
                     theta(0.5),
                     uD(uD_),
                     mu(mu_),
                     mesh_filename(mesh_filename_) {}
    void run();

private:
    void setup_system();
    void solve_dt();
    void output_results();

    Triangulation<dim> triangulation;
    const FE_SimplexP<dim>    fe;
    DoFHandler<dim>    dof_handler;

    // AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> old_solution;
    Vector<double> system_rhs;

    std::vector<std::pair<double,std::string>> times_and_names;

    double       time;
    unsigned int timestep_number;

    const double dt;
    const double T;
    const double theta;
    const double uD;
    const double mu;

    const std::string mesh_filename;
};

template <int dim>
void HeatEquation<dim>::setup_system()
{
    dof_handler.distribute_dofs(fe);

    std::cout << std::endl
            << "===========================================" << std::endl
            << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl
            << std::endl;

    // constraints.clear();
    // DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    // constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
                                    // constraints,
                                    // /*keep_constrained_dofs = */ true);
    sparsity_pattern.copy_from(dsp);

    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(sparsity_pattern);

    MatrixCreator::create_mass_matrix(dof_handler,
                                    QGaussSimplex<dim>(fe.degree + 1),
                                    mass_matrix);
    MatrixCreator::create_laplace_matrix(dof_handler,
                                        QGaussSimplex<dim>(fe.degree + 1),
                                        laplace_matrix);

    laplace_matrix *= mu;
    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
}

template <int dim>
void HeatEquation<dim>::solve_dt()
{
    SolverControl            solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);

    cg.solve(system_matrix, solution, system_rhs, preconditioner);

    // constraints.distribute(solution);

    std::cout << "     " << solver_control.last_step() << " CG iterations."
            << std::endl;
}

template <int dim>
void HeatEquation<dim>::output_results()
{
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "U");

    data_out.build_patches();

    data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));

    const std::string filename = "solution-" + Utilities::to_string(timestep_number, 3) + ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);

    times_and_names.emplace_back(time, filename);
    std::ofstream pvd_output("solution.pvd");
    DataOutBase::write_pvd_record(pvd_output, times_and_names);
}

template <int dim>
void HeatEquation<dim>::run()
{
    {
        std::cout << "Initializing the mesh" << std::endl;

        // Read the mesh from file.
        GridIn<dim> grid_in;
        grid_in.attach_triangulation(triangulation);

        std::ifstream mesh_file(mesh_filename);
        grid_in.read_msh(mesh_file);

        std::cout << "  Number of elements = " << triangulation.n_active_cells() << std::endl;
    }


    for (const auto &cell : triangulation.active_cell_iterators()) {
        for (const auto &face : cell->face_iterators()) {
            if (!face->at_boundary()) continue;

            if (face->center()[dim - 1] < 1e-12)
                face->set_all_boundary_ids(0); // Dirichlet boundary ID = 0
            else
                face->set_all_boundary_ids(1); // Neumann boundary ID = 1
        }
    }

    setup_system();

    Vector<double> tmp;
    Vector<double> forcing_terms;


    time            = 0.0;
    timestep_number = 0;

    tmp.reinit(solution.size());
    forcing_terms.reinit(solution.size());


    VectorTools::interpolate(dof_handler,
                            Functions::ZeroFunction<dim>(),
                            old_solution);
    solution = old_solution;

    output_results();
    while (time <= T)
    {
        time += dt;
        ++timestep_number;
  
        std::cout << "Time step " << timestep_number << " at t=" << time
                    << std::endl;
  
        mass_matrix.vmult(system_rhs, old_solution);
  
        laplace_matrix.vmult(tmp, old_solution);
        system_rhs.add(-(1 - theta) * dt, tmp);

        RightHandSide<dim> rhs_function;
        rhs_function.set_time(time);
        VectorTools::create_right_hand_side(dof_handler,
                                            QGaussSimplex<dim>(fe.degree + 1),
                                            rhs_function,
                                            tmp);
        forcing_terms = tmp;
        forcing_terms *= dt * theta;

        rhs_function.set_time(time - dt);
        VectorTools::create_right_hand_side(dof_handler,
                                            QGaussSimplex<dim>(fe.degree + 1),
                                            rhs_function,
                                            tmp);

        forcing_terms.add(dt * (1 - theta), tmp);
        system_rhs += forcing_terms;

        system_matrix.copy_from(mass_matrix);
        system_matrix.add(theta * dt, laplace_matrix);
  
        // constraints.condense(system_matrix, system_rhs);
        {
            std::vector<double> values{uD};
            Functions::ConstantFunction<dim> boundary_values_function(values);

            std::map<types::global_dof_index, double> boundary_values;
            VectorTools::interpolate_boundary_values(dof_handler,
                                                        0,
                                                        boundary_values_function,
                                                        boundary_values);

            MatrixTools::apply_boundary_values(boundary_values,
                                                system_matrix,
                                                solution,
                                                system_rhs);
        }
        solve_dt();
  
        output_results();
  
        old_solution = solution;
    }
}
