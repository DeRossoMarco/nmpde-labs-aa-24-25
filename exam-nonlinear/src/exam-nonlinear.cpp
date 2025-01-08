#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "NonLinear.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const unsigned int               mpi_rank =
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);


  const std::vector<std::string> meshes = {"../mesh/mesh-cube-5.msh",
                                           "../mesh/mesh-cube-10.msh",
                                           "../mesh/mesh-cube-20.msh",
                                           "../mesh/mesh-cube-40.msh"};
  const std::vector<double>      h_vals = {1.0 / 5.0,
                                           1.0 / 10.0,
                                           1.0 / 20.0,
                                           1.0 / 40.0};

  // For 1 dimension problem, must update h_vals (h = 1.0 / (N + 1.0))
  const std::vector<unsigned int> N_vals = {9, 19, 39, 79};

  std::vector<double> errors_L2;
  std::vector<double> errors_H1;

  const unsigned int degree = 1;

  for (unsigned int i = 0; i < meshes.size(); ++i)
    {
      NonLinear problem(meshes[i], degree);

      problem.setup();
      problem.solve_newton();
      problem.output();

      errors_L2.push_back(problem.compute_error(VectorTools::L2_norm));
      errors_H1.push_back(problem.compute_error(VectorTools::H1_norm));
    }

  if (mpi_rank == 0)
    {
      ConvergenceTable table;

      std::ofstream convergence_file("convergence.csv");
      convergence_file << "h,eL2,eH1" << std::endl;

      for (unsigned int i = 0; i < h_vals.size(); ++i)
        {
          table.add_value("h", h_vals[i]);
          table.add_value("L2", errors_L2[i]);
          table.add_value("H1", errors_H1[i]);

          convergence_file << h_vals[i] << "," << errors_L2[i] << ","
                           << errors_H1[i] << std::endl;
        }

      table.evaluate_all_convergence_rates(
        ConvergenceTable::reduction_rate_log2);
      table.set_scientific("L2", true);
      table.set_scientific("H1", true);
      table.write_text(std::cout);
    }

  return 0;
}