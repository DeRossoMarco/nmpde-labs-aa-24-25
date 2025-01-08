#include "Elliptic.hpp"

void
Elliptic::setup()
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

        const std::string mesh_file_name = "mesh-" + std::to_string(N + 1) + ".vtk";
        GridOut           grid_out;
        std::ofstream     grid_out_file(mesh_file_name);
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
    system_matrix.reinit(sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  }
}

void
Elliptic::assemble()
{
  pcout << "===============================================" << std::endl;

  pcout << "  Assembling the linear system" << std::endl;

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

  system_matrix = 0.0;
  system_rhs    = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      // If current cell is not owned locally, we skip it.
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // Evaluate coefficients on this quadrature node.
          const double mu_loc =
            diffusion_coefficient.value(fe_values.quadrature_point(q));
          const double sigma_loc =
            reaction_coefficient.value(fe_values.quadrature_point(q));
          Vector<double> beta_loc(dim);
          advection_coefficient.vector_value(fe_values.quadrature_point(q),
                                             beta_loc);

          // Convert beta_loc to a tensor.
          Tensor<1, dim> beta_loc_tensor;
          for (unsigned int i = 0; i < dim; ++i)
            beta_loc_tensor[i] = beta_loc[i];

          // Evaluate the forcing term on this quadrature node.
          const double f_loc =
            forcing_term.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // TODO: Implement the assembly of the linear system.

                  // Diffusion term.
                  cell_matrix(i, j) +=
                    mu_loc *                                     // mu(x)
                    scalar_product(fe_values.shape_grad(j, q),   // (I)
                                   fe_values.shape_grad(i, q)) * // (II)
                    fe_values.JxW(q);                            // (III)

                  // Reaction term.
                  cell_matrix(i, j) += sigma_loc *                   // sigma(x)
                                       fe_values.shape_value(j, q) * // phi_i
                                       fe_values.shape_value(i, q) * // phi_j
                                       fe_values.JxW(q);             // dx

                  // Advection term.
                  cell_matrix(i, j) +=
                    scalar_product(beta_loc_tensor, // beta(x)
                                   fe_values.shape_grad(j, q)) *
                    fe_values.shape_value(i, q) * // phi_i
                    fe_values.JxW(q);             // dx
                }

              cell_rhs(i) +=
                f_loc * fe_values.shape_value(i, q) * fe_values.JxW(q);
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

      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  // Boundary conditions.
  {
    std::map<types::global_dof_index, double> boundary_values;

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    // TODO: Fix the boundary indexes.
    for (unsigned int i = 0; i < 4; ++i)
      boundary_functions[i] = &function_g;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution, system_rhs, true);
  }
}

void
Elliptic::solve()
{
  pcout << "===============================================" << std::endl;

  SolverControl solver_control(10000, 1e-6 * system_rhs.l2_norm());

  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(
    system_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  pcout << "  Solving the linear system" << std::endl;
  if (N == 0)
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
  else
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
  pcout << "  " << solver_control.last_step() << " iterations" << std::endl;
}

void
Elliptic::output() const
{
  pcout << "===============================================" << std::endl;

  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  TrilinosWrappers::MPI::Vector solution_ghost(locally_owned_dofs,
                                               locally_relevant_dofs,
                                               MPI_COMM_WORLD);

  solution_ghost = solution;

  // Then, we build and fill the DataOut class as usual.
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution_ghost, "u");

  // We also add a vector to represent the parallel partitioning of the mesh.
  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  const std::filesystem::path mesh_path(mesh_file_name);
  const std::string output_file_name = "output-" + mesh_path.stem().string();

  data_out.write_vtu_with_pvtu_record("./",
                                      output_file_name,
                                      0,
                                      MPI_COMM_WORLD);

  pcout << "Output written to " << output_file_name << std::endl;

  pcout << "===============================================" << std::endl;
}

double
Elliptic::compute_error(const VectorTools::NormType &norm_type) const
{
  FE_SimplexP<dim> fe_linear(1);
  MappingFE        mapping(fe_linear);

  const QGaussSimplex<dim> quadrature_error = QGaussSimplex<dim>(r + 2);

  // First we compute the norm on each element, and store it in a vector.
  Vector<double> error_per_cell(mesh.n_active_cells());
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    ExactSolution(),
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  // Then, we add out all the cells.
  const double error =
    VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}