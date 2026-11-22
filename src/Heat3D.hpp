#pragma once

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <functional>
#include <memory>

using namespace dealii;

class Heat3D {
public:
    static constexpr unsigned int dim = 3;

    // Constructor.
    Heat3D(const std::string                               &mesh_filename_,
           const unsigned int                              &r_,
           const std::function<double(const Point<dim> &)> &mu_,
           const std::function<double(const Point<dim> &)> &f_,
           const std::function<double(const Point<dim> &)> &h_,
           const double                                    &u_d_)
           : mesh_filename(mesh_filename_)
           , r(r_)
           , mu(mu_)
           , f(f_)
           , h(h_)
           , u_d(u_d_)
           {}

    void setup();
    void assemble();
    void solve();
    void output() const;
private:
    const std::string mesh_filename;

    // Polynomial degree.
    const unsigned int r;

    // Diffusion coefficient.
    std::function<double(const Point<dim> &)> mu;

    // Forcing term.
    std::function<double(const Point<dim> &)> f;
    std::function<double(const Point<dim> &)> h;

    //
    const double u_d;

    // Triangulation.
    Triangulation<dim> mesh;
    Triangulation<dim> hypcub;

    std::unique_ptr<FiniteElement<dim>> fe;

    // Quadrature formula.
    //
    // We use a unique_ptr here so that we can choose the type and order of the
    // quadrature formula at runtime (the order is a constructor parameter).
    std::unique_ptr<Quadrature<dim>> quadrature;

    // Quadrature formula for boundary integrals.
    std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;

    // DoF handler.
    DoFHandler<dim> dof_handler;

    // Sparsity pattern.
    SparsityPattern sparsity_pattern;

    // System matrix.
    SparseMatrix<double> system_matrix;

    // System right-hand side.
    Vector<double> system_rhs;

    // System solution.
    Vector<double> solution;
};