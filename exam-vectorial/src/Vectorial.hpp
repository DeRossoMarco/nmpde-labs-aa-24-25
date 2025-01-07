#ifndef VECTORIAL_HPP
#define VECTORIAL_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

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
class Vectorial
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

  // Function for the mu_0 coefficient.
  class FunctionMu : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /* p */,
          const unsigned int /* component */ = 0) const override
    {
      // TODO: implement the mu_0 coefficient.
      return 1.0;
    }
  };

  // Function for the lambda coefficient.
  class FunctionLambda : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /* p */,
          const unsigned int /* component */ = 0) const override
    {
      // TODO: implement the lambda coefficient.
      return 10.0;
    }
  };

  // Function for the forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      for (unsigned int i = 0; i < dim; ++i)
        values[i] = value(p, i);
    }

    virtual double
    value(const Point<dim> & /* p */,
          const unsigned int component = 0) const override
    {
      // TODO: implement the forcing term.
      if (component == 0)
        return 0.0;
      else if (component == 1)
        return 0.0;
      else // if (component == 2)
        return val;
    }

  protected:
    const double val = -1.0;
  };

  // Function for the Dirichlet datum.
  class FunctionG : public Function<dim>
  {
  public:
    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      for (unsigned int i = 0; i < dim; ++i)
        values[i] = value(p, i);
    }

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      // TODO: implement the Dirichlet datum.
      if (component == 0)
        return 0.1 * p[0];
      else if (component == 1)
        return 0.1 * p[0];
      else // if (component == 2)
        return 0.0;
    }
  };

  // Function for the Neumann datum.
  /* class FunctionH : public Function<dim>
  {
  public:
    virtual void
    vector_value(const Point<dim> & p,
                 Vector<double> &values) const override
    {
      for (unsigned int i = 0; i < dim; ++i)
        values[i] = value(p, i);
    }

    virtual double
    value(const Point<dim> & p,
          const unsigned int component = 0) const override
    {
      // TODO: implement the Neumann datum.
      return 0.0;
    }
  }; */

  // Exact solution.
  class ExactSolution : public Function<dim>
  {
  public:
    // Constructor.
    ExactSolution()
    {}

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      for (unsigned int i = 0; i < dim; ++i)
        values[i] = value(p, i);
    }

    virtual double
    value(const Point<dim> & /* p */,
          const unsigned int /* component */ = 0) const override
    {
      // TODO: implement the exact solution.
      return 0.0;
    }

    virtual void
    vector_gradient(const Point<dim>            &p,
                    std::vector<Tensor<1, dim>> &gradients) const override
    {
      for (unsigned int i = 0; i < dim; ++i)
        gradients[i] = gradient(p, i);
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim> & /* p */,
             const unsigned int /* component */ = 0) const override
    {
      // TODO: implement the gradient of the exact solution.
      Tensor<1, dim> result;

      result[0] = 0.0;
      result[1] = 0.0;
      result[2] = 0.0;

      return result;
    }
  };

  // Constructor.
  Vectorial(const std::string &mesh_file_name_, const unsigned int &r_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , mesh_file_name(mesh_file_name_)
    , r(r_)
    , mesh(MPI_COMM_WORLD)
  {}

  // Initialization.
  void
  setup();

  // Assemble the tangent problem.
  void
  assemble_system();

  // Solve the tangent problem.
  void
  solve_system();

  // Output.
  void
  output() const;

  // Compute the error.
  double
  compute_error(const VectorTools::NormType &norm_type) const;

protected:
  // MPI parallel.
  // /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Problem definition.
  // ///////////////////////////////////////////////////////

  // mu coefficient.
  FunctionMu mu;

  // lambda coefficient.
  FunctionLambda lambda;

  // Forcing term.
  ForcingTerm forcing_term;

  // Dirichlet datum.
  FunctionG function_g;

  // Discretization.
  // ///////////////////////////////////////////////////////////

  // Mesh file name.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // Quadrature formula for boundary integrals.
  std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // Jacobian matrix.
  TrilinosWrappers::SparseMatrix system_matrix;

  // Residual vector.
  TrilinosWrappers::MPI::Vector system_rhs;

  // System (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;
};

#endif