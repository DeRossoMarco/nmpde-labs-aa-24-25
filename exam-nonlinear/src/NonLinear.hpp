#ifndef NON_LINEAR_HPP
#define NON_LINEAR_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace dealii;

// Class representing the non-linear diffusion problem.
class NonLinear
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

  // Function for the mu_0 coefficient.
  class FunctionMu0 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      // TODO: Implement the mu_0 coefficient.
      return 1.0;
    }
  };

  // Function for the mu_1 coefficient.
  class FunctionMu1 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      // TODO: Implement the mu_1 coefficient.
      return 10.0;
    }
  };

  // Function for the forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      // TODO: Implement the forcing term.
      return 1.0;
    }
  };

  // Dirichlet boundary conditions.
  class FunctionG : public Function<dim>
  {
  public:
    // Constructor.
    FunctionG()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      // TODO: Implement the Dirichlet boundary conditions.
      return 0.0;
    }
  };

  // Neumann boundary conditions.
  class FunctionH : public Function<dim>
  {
  public:
    // Constructor.
    FunctionH()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      // TODO: Implement the Neumann boundary conditions.
      return 0.0;
    }
  };

  // Exact solution.
  class ExactSolution : public Function<dim>
  {
  public:
    // Constructor.
    ExactSolution()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      // TODO: Implement the exact solution.
      return 0.0;
    }

    // Gradient evaluation.
    virtual Tensor<1, dim>
    gradient(const Point<dim> & /*p*/,
             const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> result;

      // TODO: Implement the gradient of the exact solution.
      result[0] = 0.0;
      result[1] = 0.0;
      result[2] = 0.0;

      return result;
    }
  };

  // Constructor.
  NonLinear(const std::string  &mesh_file_name_,
            const unsigned int &r_,
            const unsigned int &N_ = 0)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , mesh_file_name(mesh_file_name_)
    , N(N_)
    , r(r_)
    , mesh(MPI_COMM_WORLD)
  {}

  // Initialization.
  void
  setup();

  // Solve the problem using Newton's method.
  void
  solve_newton();

  // Output.
  void
  output() const;

  // Compute error.
  double
  compute_error(const VectorTools::NormType &norm_type) const;

protected:
  // Assemble the tangent problem.
  void
  assemble_system();

  // Solve the tangent problem.
  void
  solve_system();

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Problem definition. ///////////////////////////////////////////////////////

  // mu_0 coefficient.
  FunctionMu0 mu_0;

  // mu_1 coefficient.
  FunctionMu1 mu_1;

  // Forcing term.
  ForcingTerm forcing_term;

  // Dirichlet boundary conditions.
  FunctionG function_g;

  // Neumann boundary conditions.
  FunctionH function_h;

  // Discretization. ///////////////////////////////////////////////////////////

  // Mesh file name.
  const std::string mesh_file_name;

  // Number of element in the x direction.
  const unsigned int N;

  // Polynomial degree.
  const unsigned int r;

  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // Quadrature formula for boundary.
  std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // Jacobian matrix.
  TrilinosWrappers::SparseMatrix jacobian_matrix;

  // Residual vector.
  TrilinosWrappers::MPI::Vector residual_vector;

  // Solution increment (without ghost elements).
  TrilinosWrappers::MPI::Vector delta_owned;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;
};

#endif