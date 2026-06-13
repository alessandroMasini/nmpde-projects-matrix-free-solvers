/**
 * @file boundaries.hpp
 * @brief Boundary-condition aliases used by the ADR solvers.
 *
 * @details Defines the common Dirichlet and Neumann boundary abstractions and
 * the maps from deal.II boundary identifiers to boundary-condition functions.
 */

#pragma once

#include <unordered_map>

#include <deal.II/base/function.h>
#include <deal.II/base/types.h>

#include "function_types.hpp"

namespace MFSolver
{
    /**
     * @brief Represents a function that describes a Dirichlet boundary condition.
     * @tparam dim The dimensionality of the space the ADR problem is living in.
     */
    template <int dim>
    using DirichletBoundary = dealii::Function<dim>;

    /**
     * @brief Represents a function that describes a Neumann boundary condition.
     * @tparam dim The dimensionality of the space the ADR problem is living in.
     */
    template <int dim>
    using NeumannBoundary = RealFunction<dim>;

    /**
     * @brief Maps boundary identifiers to their corresponding boundary condition.
     * @tparam T The type of boundary condition.
     */
    template <typename T>
    using Boundaries = std::unordered_map<types::boundary_id, T>;

    /**
     * @brief Maps boundary identifiers to Dirichlet boundary conditions.
     * @tparam dim The dimensionality of the space the ADR problem is living in.
     */
    template <int dim>
    using DirichletBoundaries = Boundaries<std::shared_ptr<DirichletBoundary<dim>>>;

    /**
     * @brief Maps boundary identifiers to Neumann boundary conditions.
     * @tparam dim The dimensionality of the space the ADR problem is living in.
     */
    template <int dim>
    using NeumannBoundaries = Boundaries<std::shared_ptr<NeumannBoundary<dim>>>;
};
