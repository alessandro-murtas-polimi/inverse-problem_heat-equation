#include "Heat3D.hpp"
#include <fstream>
#include <iostream>

void Heat3D::setup() {
    std::cout << "===============================================" << std::endl;
    
    // Create the mesh.
    // {
    //     std::cout << "Initializing the mesh" << std::endl;
    //     GridGenerator::subdivided_hyper_cube_with_simplices(mesh, N_el, 0.0, 1.0, false);
    //     std::cout << "  Number of elements = " << mesh.n_active_cells() << std::endl;

    //     // Write the mesh to file.
    //     //
    //     // Since we generate the mesh internally, we also write it to file for
    //     // possible inspection by the user. This would not be necessary if we read
    //     // the mesh from file, as we will do later on.
    //     const std::string mesh_file_name = "mesh-" + std::to_string(N_el) + ".vtk";
    //     GridOut           grid_out;
    //     std::ofstream     grid_out_file(mesh_file_name);
    //     grid_out.write_vtk(mesh, grid_out_file);
    //     std::cout << "  Mesh saved to " << mesh_file_name << std::endl;
    // }
    // Create the mesh.
    {
        std::cout << "Initializing the mesh" << std::endl;

        // Read the mesh from file.
        GridIn<dim> grid_in;
        grid_in.attach_triangulation(mesh);

        std::ifstream mesh_file(mesh_filename);
        grid_in.read_msh(mesh_file);

        std::cout << "  Number of elements = " << mesh.n_active_cells() << std::endl;
    }

    std::cout << "-----------------------------------------------" << std::endl;

    
    for (const auto &cell : mesh.active_cell_iterators()) {
        for (const auto &face : cell->face_iterators()) {
            if (!face->at_boundary()) continue;

            if (face->center()[2] == 0)
                face->set_all_boundary_ids(0); // Dirichlet boundary ID = 0
            else
                face->set_all_boundary_ids(1); // Dirichlet boundary ID = 1
        }
    }

    
    // Initialize the finite element space.
    {
        std::cout << "Initializing the finite element space" << std::endl;

        fe = std::make_unique<FE_SimplexP<dim>>(r);

        std::cout << "  Degree                     = " << fe->degree << std::endl;
        std::cout << "  DoFs per cell              = " << fe->dofs_per_cell << std::endl;

        quadrature          = std::make_unique<QGaussSimplex<dim>>(r + 1);
        quadrature_boundary = std::make_unique<QGaussSimplex<dim - 1>>(r + 1);

        std::cout << "  Quadrature points per cell = " << quadrature->size() << std::endl;
    }
    
    std::cout << "-----------------------------------------------" << std::endl;

    // Initialize the DoF handler.
    {
        std::cout << "Initializing the DoF handler" << std::endl;

        dof_handler.reinit(mesh);
        dof_handler.distribute_dofs(*fe);

        std::cout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
    }
    
    std::cout << "-----------------------------------------------" << std::endl;

    // Initialize the linear system.
    {
        std::cout << "Initializing the linear system" << std::endl;

        std::cout << "  Initializing the sparsity pattern" << std::endl;
        DynamicSparsityPattern dsp(dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp);
        sparsity_pattern.copy_from(dsp);

        std::cout << "  Initializing the system matrix" << std::endl;
        system_matrix.reinit(sparsity_pattern);

        std::cout << "  Initializing the system right-hand side" << std::endl;
        system_rhs.reinit(dof_handler.n_dofs());
        std::cout << "  Initializing the solution vector" << std::endl;
        solution.reinit(dof_handler.n_dofs());
    }
}

void Heat3D::assemble() {
    std::cout << "===============================================" << std::endl;

    std::cout << "  Assembling the linear system" << std::endl;

    // Number of local DoFs for each element.
    const unsigned int dofs_per_cell = fe->dofs_per_cell;

    // Number of quadrature points for each element.
    const unsigned int n_q = quadrature->size();

    // FEValues instance. This object allows to compute basis functions, their
    // derivatives, the reference-to-current element mapping and its
    // derivatives on all quadrature points of all elements.
    FEValues<dim> fe_values(
        *fe,
        *quadrature,
        // Here we specify what quantities we need FEValues to compute on
        // quadrature points. For our test, we need:
        // - the values of shape functions (update_values);
        // - the derivative of shape functions (update_gradients);
        // - the position of quadrature points (update_quadrature_points);
        // - the quadrature weights (update_JxW_values).
        update_values | update_gradients | update_quadrature_points |
        update_JxW_values);

    // Local matrix and right-hand side vector. We will overwrite them for
    // each element within the loop.
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    // We will use this vector to store the global indices of the DoFs of the
    // current element within the loop.
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    // Reset the global matrix and vector, just in case.
    system_matrix = 0.0;
    system_rhs    = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators()) {
        // Reinitialize the FEValues object on current element. This
        // precomputes all the quantities we requested when constructing
        // FEValues (see the update_* flags above) for all quadrature nodes of
        // the current cell.
        fe_values.reinit(cell);

        // We reset the cell matrix and vector (discarding any leftovers from
        // previous element).
        cell_matrix = 0.0;
        cell_rhs    = 0.0;

        for (unsigned int q = 0; q < n_q; ++q) {
            // Here we assemble the local contribution for current cell and
            // current quadrature point, filling the local matrix and vector.
            const double mu_loc = mu(fe_values.quadrature_point(q));
            const double f_loc  = f(fe_values.quadrature_point(q));

            // Here we iterate over *local* DoF indices.
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    cell_matrix(i, j) += mu_loc * fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) * fe_values.JxW(q);
                }

                cell_rhs(i) += f_loc * fe_values.shape_value(i, q) * fe_values.JxW(q);
            }
        }

        // At this point the local matrix and vector are constructed: we need
        // to sum them into the global matrix and vector. To this end, we need
        // to retrieve the global indices of the DoFs of current cell.
        cell->get_dof_indices(dof_indices);

        // Then, we add the local matrix and vector into the corresponding
        // positions of the global matrix and vector.
        system_matrix.add(dof_indices, cell_matrix);
        system_rhs.add(dof_indices, cell_rhs);
    }

    // Boundary conditions.
    //
    // So far we assembled the matrix as if there were no Dirichlet conditions.
    // Now we want to replace the rows associated to nodes on which Dirichlet
    // conditions are applied with equations like u_i = b_i. We use deal.ii
    // functions to
    {
        // We construct a map that stores, for each DoF corresponding to a Dirichlet
        // condition, the corresponding value. E.g., if the Dirichlet condition is
        // u_i = b_i, the map will contain the pair (i, b_i).
        std::map<types::global_dof_index, double> boundary_values;

        // This object represents our boundary data as a real-valued function (that
        // always evaluates to zero). Other functions may require to implement a
        // custom class derived from dealii::Function<dim>.
        // Functions::ZeroFunction<dim> bc_function;
        std::vector<double> values{u_d};
        Functions::ConstantFunction<dim> bc_function(values);

        // Functions::ConstantFunction<dim> bc_function(values1);

        // Then, we build a map that, for each boundary tag, stores a pointer to the
        // corresponding boundary function.
        std::map<types::boundary_id, const Function<dim> *> boundary_functions;
        boundary_functions[0] = &bc_function;

        // interpolate_boundary_values fills the boundary_values map.
        VectorTools::interpolate_boundary_values(dof_handler,
                                                boundary_functions,
                                                boundary_values);

        // Finally, we modify the linear system to apply the boundary conditions.
        // This replaces the equations for the boundary DoFs with the corresponding
        // u_i = 0 equations.
        MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs, true);
    }
}

void Heat3D::solve() {
    std::cout << "===============================================" << std::endl;
    
    // Here we specify the maximum number of iterations of the iterative solver,
    // and its absolute and relative tolerances.
    ReductionControl solver_control(/* maxiter = */ 1000,
                                    /* tolerance = */ 1.0e-16,
                                    /* reduce = */ 1.0e-6);
                                    
    // Since the system matrix is symmetric and positive definite, we solve the
    // system using the conjugate gradient method.
    SolverCG<Vector<double>> solver(solver_control);

    std::cout << "  Solving the linear system" << std::endl;
    // We use the identity preconditioner for now.
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
    std::cout << "  " << solver_control.last_step() << " CG iterations" << std::endl;
}

void Heat3D::output() const {
    std::cout << "===============================================" << std::endl;
    
    // The DataOut class manages writing the results to a file.
    DataOut<dim> data_out;
    
    // It can write multiple variables (defined on the same mesh) to a single
    // file. Each of them can be added by calling add_data_vector, passing the
    // associated DoFHandler and a name.
    data_out.add_data_vector(dof_handler, solution, "solution");

    // Once all vectors have been inserted, call build_patches to finalize the
    // DataOut object, preparing it for writing to file.
    data_out.build_patches();
    
    // Then, use one of the many write_* methods to write the file in an
    // appropriate format.
    const std::string output_file_name = "output.vtk";
    std::ofstream output_file(output_file_name);
    data_out.write_vtk(output_file);
    
    std::cout << "Output written to " << output_file_name << std::endl;
    std::cout << "===============================================" << std::endl;
}