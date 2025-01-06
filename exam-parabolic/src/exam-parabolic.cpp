#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "Parabolic.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const unsigned int               mpi_rank =
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  const unsigned int degree = 2;

  const double T     = 1.0;
  const double theta = 0.5;

  // Time convergence study.
  const std::vector<double> deltat_vector = {
    0.25, 0.125, 0.0625, 0.03125, 0.015625};
  std::vector<double> errors_L2_deltat;
  std::vector<double> errors_H1_deltat;

  for (const auto &deltat : deltat_vector)
    {
      Parabolic problem("../mesh/mesh-cube-20.msh", degree, T, deltat, theta);

      problem.setup();
      problem.solve();

      errors_L2_deltat.push_back(problem.compute_error(VectorTools::L2_norm));
      errors_H1_deltat.push_back(problem.compute_error(VectorTools::H1_norm));
    }

  // Spatial convergence study.
  const std::vector<std::string> meshes = {"../mesh/mesh-cube-5.msh",
                                           "../mesh/mesh-cube-10.msh",
                                           "../mesh/mesh-cube-20.msh",
                                           "../mesh/mesh-cube-40.msh"};
  const std::vector<double>      h_vals = {1.0 / 5.0,
                                           1.0 / 10.0,
                                           1.0 / 20.0,
                                           1.0 / 40.0};
  std::vector<double>            errors_L2_h;
  std::vector<double>            errors_H1_h;

  for (unsigned int i = 0; i < meshes.size(); ++i)
    {
      Parabolic problem(meshes[i], degree, T, 0.03125, theta);

      problem.setup();
      problem.solve();

      errors_L2_h.push_back(problem.compute_error(VectorTools::L2_norm));
      errors_H1_h.push_back(problem.compute_error(VectorTools::H1_norm));
    }

  if (mpi_rank == 0)
    {
      std::cout << "==============================================="
                << std::endl;

      std::ofstream convergence_file_deltat("convergence_deltat.csv");
      convergence_file_deltat << "dt,eL2,eH1" << std::endl;

      for (unsigned int i = 0; i < deltat_vector.size(); ++i)
        {
          convergence_file_deltat << deltat_vector[i] << ","
                                  << errors_L2_deltat[i] << ","
                                  << errors_H1_deltat[i] << std::endl;

          std::cout << std::scientific << "dt = " << std::setw(4)
                    << std::setprecision(2) << deltat_vector[i];

          std::cout << std::scientific << " | eL2 = " << errors_L2_deltat[i];

          // Estimate the convergence order.
          if (i > 0)
            {
              const double p =
                std::log(errors_L2_deltat[i] / errors_L2_deltat[i - 1]) /
                std::log(deltat_vector[i] / deltat_vector[i - 1]);

              std::cout << " (" << std::fixed << std::setprecision(2)
                        << std::setw(4) << p << ")";
            }
          else
            std::cout << " (  - )";

          std::cout << std::scientific << " | eH1 = " << errors_H1_deltat[i];

          // Estimate the convergence order.
          if (i > 0)
            {
              const double p =
                std::log(errors_H1_deltat[i] / errors_H1_deltat[i - 1]) /
                std::log(deltat_vector[i] / deltat_vector[i - 1]);

              std::cout << " (" << std::fixed << std::setprecision(2)
                        << std::setw(4) << p << ")";
            }
          else
            std::cout << " (  - )";

          std::cout << "\n";
        }
    }

  if (mpi_rank == 0)
    {
      ConvergenceTable table;

      std::ofstream convergence_file_h("convergence_h.csv");
      convergence_file_h << "h,eL2,eH1" << std::endl;

      for (unsigned int i = 0; i < h_vals.size(); ++i)
        {
          table.add_value("h", h_vals[i]);
          table.add_value("L2", errors_L2_h[i]);
          table.add_value("H1", errors_H1_h[i]);

          convergence_file_h << h_vals[i] << "," << errors_L2_h[i] << ","
                             << errors_H1_h[i] << std::endl;
        }
      table.evaluate_all_convergence_rates(
        ConvergenceTable::reduction_rate_log2);
      table.set_scientific("L2", true);
      table.set_scientific("H1", true);
      table.write_text(std::cout);
    }

  return 0;
}