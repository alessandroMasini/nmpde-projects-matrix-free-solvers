#ifndef PROBLEMDATA_HPP
#define PROBLEMDATA_HPP

#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/function.h>

#include <functional>
#include <memory>

namespace ADR {

/**
 * @brief A common structure to hold the algebraic and analytical data 
 * required defining the Advection-Diffusion-Reaction (ADR) problem.
 * 
 * This ensures that both the Matrix-Based and Matrix-Free solvers
 * solve the exact same mathematical problem.
 */
template <int dim>
struct ProblemData {
    // Polynomial degree of the Finite Elements
    unsigned int fe_degree = 1;

    // Mesh refinement level (if generating a hyper_cube/hyper_ball)
    unsigned int refinement_level = 4;

    // Number of elements in each direction (if using a subdivision)
    unsigned int elements_per_direction = 10;

    // Final time (if time-dependent) - can be set to 0 strictly for steady state
    double T = 1.0;

    // Time step (if time-dependent)
    double delta_t = 0.01;

    // Theta parameter for the Theta-method (time integration)
    double theta = 0.5;

    // --- PDE Coefficients ---
    
    // Diffusion coefficient function: mu(x)
    std::function<double(const dealii::Point<dim>&)> mu;

    // Advection coefficient function: beta(x) (velocity field)
    std::function<dealii::Tensor<1, dim>(const dealii::Point<dim>&)> beta;

    // Reaction coefficient function: gamma(x) (or k in some notations)
    std::function<double(const dealii::Point<dim>&)> gamma;

    // Forcing term: f(x, t)
    std::function<double(const dealii::Point<dim>&, const double&)> forcing_term;

    // Dirichlet boundary condition: g(x, t) for the general lifting
    std::function<double(const dealii::Point<dim>&, const double&)> dirichlet_boundary_value;

    /**
     * @brief Helper to initialize with some default test-case values
     */
    static ProblemData<dim> standard_test_case() {
        ProblemData<dim> data;
        
        data.fe_degree = 1;
        data.refinement_level = 5;

        // Default mu = 1.0
        data.mu = [](const dealii::Point<dim>& /*p*/) { return 1.0; };

        // Default beta = [1.0, 1.0, ...]
        data.beta = [](const dealii::Point<dim>& /*p*/) { 
            dealii::Tensor<1, dim> b;
            for (unsigned int d = 0; d < dim; ++d) b[d] = 1.0;
            return b;
        };

        // Default gamma = 0.0
        data.gamma = [](const dealii::Point<dim>& /*p*/) { return 0.0; };

        // Default forcing term f = 1.0
        data.forcing_term = [](const dealii::Point<dim>& /*p*/, const double& /*t*/) { return 1.0; };

        // Default homogeneous Dirichlet boundaries
        data.dirichlet_boundary_value = [](const dealii::Point<dim>& /*p*/, const double& /*t*/) { return 0.0; };

        return data;
    }
};

} // namespace ADR

#endif // PROBLEMDATA_HPP