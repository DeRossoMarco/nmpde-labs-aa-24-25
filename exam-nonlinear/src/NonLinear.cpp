#include "NonLinear.hpp"

void
NonLinear::setup()
{
  pcout << "===============================================" << std::endl;

  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;

    if (N == 0)
      {
        GridIn<dim> grid_in;
        grid_in.attach_triangulation(mesh_serial);

        std::ifstream grid_in_file(mesh_file_name);
        grid_in.read_msh(grid_in_file);
      }
    else
      {
        GridGenerator::subdivided_hyper_cube(
          mesh_serial, N + 1, 0.0, 1.0, true);

        const std::string mesh_file_name =
          "mesh-" + std::to_string(N + 1) + ".vtk";
        GridOut       grid_out;
        std::ofstream grid_out_file(mesh_file_name);
        grid_out.write_vtk(mesh_serial, grid_out_file);
        pcout << "  Mesh saved to " << mesh_file_name << std::endl;
      }

    {
      GridTools::partition_triangulation(mpi_size, mesh_serial);
      const auto construction_data = TriangulationDescription::Utilities::
        create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
      mesh.create_triangulation(construction_data);
    }

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space. This is the same as in serial codes.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;

    quadrature_boundary = std::make_unique<QGaussSimplex<dim - 1>>(r + 1);

    pcout << "  Quadrature points per boundary cell = "
          << quadrature_boundary->size() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);

    sparsity.compress();

    pcout << "  Initializing the system matrix" << std::endl;
    jacobian_matrix.reinit(sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    residual_vector.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    delta_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  }
}

void
NonLinear::assemble_system()
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_values_boundary(*fe,
                                       *quadrature_boundary,
                                       update_values |
                                         update_quadrature_points |
                                         update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  jacobian_matrix = 0.0;
  residual_vector = 0.0;

  std::vector<double>         solution_loc(n_q);
  std::vector<Tensor<1, dim>> solution_gradient_loc(n_q);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      fe_values.get_function_values(solution, solution_loc);
      fe_values.get_function_gradients(solution, solution_gradient_loc);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const double mu_0_loc = mu_0.value(fe_values.quadrature_point(q));
          const double mu_1_loc = mu_1.value(fe_values.quadrature_point(q));
          const double f_loc =
            forcing_term.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // TODO: Implement the assembly of the linear system.
                  cell_matrix(i, j) +=
                    (2.0 * mu_1_loc * solution_loc[q] *
                     fe_values.shape_value(j, q)) *
                    scalar_product(solution_gradient_loc[q],
                                   fe_values.shape_grad(i, q)) *
                    fe_values.JxW(q);

                  cell_matrix(i, j) +=
                    (mu_0_loc + mu_1_loc * solution_loc[q] * solution_loc[q]) *
                    scalar_product(fe_values.shape_grad(j, q),
                                   fe_values.shape_grad(i, q)) *
                    fe_values.JxW(q);
                }

              // -F(v)
              cell_rhs(i) +=
                f_loc * fe_values.shape_value(i, q) * fe_values.JxW(q);

              // a(u)(v)
              cell_rhs(i) -=
                (mu_0_loc + mu_1_loc * solution_loc[q] * solution_loc[q]) *
                scalar_product(solution_gradient_loc[q],
                               fe_values.shape_grad(i, q)) *
                fe_values.JxW(q);
            }
        }

      if (cell->at_boundary())
        {
          for (unsigned int face_number = 0; face_number < cell->n_faces();
               ++face_number)
            {
              // TODO: Fix boundary indexes.
              if (cell->face(face_number)->at_boundary() &&
                  (cell->face(face_number)->boundary_id() == 4 ||
                   cell->face(face_number)->boundary_id() == 5))
                {
                  fe_values_boundary.reinit(cell, face_number);

                  for (unsigned int q = 0; q < quadrature_boundary->size(); ++q)
                    {
                      const double h_loc = function_h.value(
                        fe_values_boundary.quadrature_point(q));

                      // TODO: Fix boundary integral.
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        cell_rhs(i) +=
                          h_loc *                                // h(xq)
                          fe_values_boundary.shape_value(i, q) * // v(xq)
                          fe_values_boundary.JxW(q);             // Jq wq
                    }
                }
            }
        }

      cell->get_dof_indices(dof_indices);

      jacobian_matrix.add(dof_indices, cell_matrix);
      residual_vector.add(dof_indices, cell_rhs);
    }

  jacobian_matrix.compress(VectorOperation::add);
  residual_vector.compress(VectorOperation::add);

  // Boundary conditions.
  {
    std::map<types::global_dof_index, double> boundary_values;

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    Functions::ZeroFunction<dim>                        zero_function;

    // TODO: Fix boundary indexes.
    for (unsigned int i = 0; i < 4; ++i)
      boundary_functions[i] = &zero_function;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    MatrixTools::apply_boundary_values(
      boundary_values, jacobian_matrix, delta_owned, residual_vector, true);
  }
}

void
NonLinear::solve_system()
{
  SolverControl solver_control(1000, 1e-6 * residual_vector.l2_norm());

  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionSSOR         preconditioner;
  preconditioner.initialize(
    jacobian_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  solver.solve(jacobian_matrix, delta_owned, residual_vector, preconditioner);
  pcout << "   " << solver_control.last_step() << " GMRES iterations"
        << std::endl;
}

void
NonLinear::solve_newton()
{
  pcout << "===============================================" << std::endl;

  const unsigned int n_max_iters        = 1000;
  const double       residual_tolerance = 1e-6;

  unsigned int n_iter        = 0;
  double       residual_norm = residual_tolerance + 1;

  while (n_iter < n_max_iters && residual_norm > residual_tolerance)
    {
      assemble_system();
      residual_norm = residual_vector.l2_norm();

      pcout << "Newton iteration " << n_iter << "/" << n_max_iters
            << " - ||r|| = " << std::scientific << std::setprecision(6)
            << residual_norm << std::flush;

      // We actually solve the system only if the residual is larger than the
      // tolerance.
      if (residual_norm > residual_tolerance)
        {
          solve_system();

          solution_owned += delta_owned;
          solution = solution_owned;
        }
      else
        {
          pcout << " < tolerance" << std::endl;
        }

      ++n_iter;
    }

  pcout << "===============================================" << std::endl;
}

void
NonLinear::output() const
{
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "u");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  std::string                 output_file_name;
  const std::filesystem::path mesh_path(mesh_file_name);
  if (N == 0)
    output_file_name = "output-" + mesh_path.stem().string();
  else
    output_file_name = "output-mesh-" + std::to_string(N + 1);

  data_out.write_vtu_with_pvtu_record("./",
                                      output_file_name,
                                      0,
                                      MPI_COMM_WORLD);

  pcout << "Output written to " << output_file_name << "." << std::endl;

  pcout << "===============================================" << std::endl;
}

double
NonLinear::compute_error(const VectorTools::NormType &norm_type) const
{
  FE_SimplexP<dim> fe_linear(1);
  MappingFE        mapping(fe_linear);

  const QGaussSimplex<dim> quadrature_error = QGaussSimplex<dim>(r + 2);

  Vector<double> error_per_cell(mesh.n_active_cells());
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    ExactSolution(),
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  const double error =
    VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}