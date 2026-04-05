#ifndef PROBLEMDATA_HPP
#define PROBLEMDATA_HPP

#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/function.h>

#include "function_types.hpp"

#include <memory>

namespace ADR {

template <int dim>
class ConstantRealFunction : public MFSolver::RealFunction<dim> {
private:
    double val;
public:
    ConstantRealFunction(double v) : val(v) {}

    virtual double value(const dealii::Point<dim> &p, const unsigned int component = 0) const override {
        return value<double>(p, component);
    }

    template <typename Number>
    Number value(const dealii::Point<dim, Number> &/*p*/, const unsigned int /*component*/ = 0) const {
        return Number(val);
    }
};

template <int dim>
class ConstantVectorFunctionWithGradient : public MFSolver::VectorFunctionWithGradient<dim> {
private:
    double val;
public:
    using Super = typename MFSolver::VectorFunctionWithGradient<dim>;

    ConstantVectorFunctionWithGradient(double v) : val(v) {}

    virtual typename Super::template value_type<double> value(const dealii::Point<dim> &p) const override {
        return value<double>(p);
    }

    template <typename Number>
    typename Super::template value_type<Number> value(const dealii::Point<dim, Number> &/*p*/) const {
        dealii::Tensor<1, dim, Number> b;
        for (unsigned int d = 0; d < dim; ++d) b[d] = Number(val);
        return b;
    }

    virtual double divergence(const dealii::Point<dim> &p) const override {
        return divergence<double>(p);
    }

    template <typename Number>
    Number divergence(const dealii::Point<dim, Number> &/*p*/) const {
        return Number(0.0);
    }

    virtual typename Super::template gradient_type<double> gradient(const dealii::Point<dim> &p) const override {
        return gradient<double>(p);
    }

    template <typename Number>
    typename Super::template gradient_type<Number> gradient(const dealii::Point<dim, Number> &/*p*/) const {
        return typename Super::template gradient_type<Number>();
    }
};

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
    std::shared_ptr<MFSolver::RealFunction<dim>> mu;

    // Advection coefficient function: beta(x) (velocity field)
    std::shared_ptr<MFSolver::VectorFunctionWithGradient<dim>> beta;

    // Reaction coefficient function: gamma(x) (or k in some notations)
    std::shared_ptr<MFSolver::RealFunction<dim>> gamma;

    // Forcing term: f(x, t)
    std::shared_ptr<MFSolver::RealFunction<dim>> forcing_term;

    // Dirichlet boundary condition: g(x, t) for the general lifting
    std::shared_ptr<MFSolver::RealFunction<dim>> dirichlet_boundary_value;

    /**
     * @brief Helper to initialize with some default test-case values
     */
    static ProblemData<dim> standard_test_case() {
        ProblemData<dim> data;
        
        data.fe_degree = 1;
        data.refinement_level = 5;

        data.mu = std::make_shared<ConstantRealFunction<dim>>(1.0);
        data.beta = std::make_shared<ConstantVectorFunctionWithGradient<dim>>(1.0);
        data.gamma = std::make_shared<ConstantRealFunction<dim>>(0.0);
        data.forcing_term = std::make_shared<ConstantRealFunction<dim>>(1.0);
        data.dirichlet_boundary_value = std::make_shared<ConstantRealFunction<dim>>(0.0);

        return data;
    }
};

} // namespace ADR

#endif // PROBLEMDATA_HPP
