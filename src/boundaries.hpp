#pragma once

#include <unordered_map>

#include <deal.II/base/types.h>

#include "function_types.hpp"

namespace MFSolver
{
    /**
     * \brief Represents a function that describes a Dirichlet boundary condition.
     * \tparam dim The dimensionality of the space the ADR problem is living in.
     */
    template <int dim>
    using DirichletBoundary = RealFunction<dim>;

    /**
     * \brief Represents a function tht describes a Neumann boundary condition.
     * \tparam dim The dimensionality of the space the ADR problem is living in.
     */
    template <int dim>
    using NeumannBoundary = RealFunction<dim>;

    /**
     * \brief Represents a mapping between boundaries (identified by boundary IDs) and the corresponding boundary condition.
     * \tparam T The type of boundary condition.
     */
    template <typename T>
    using Boundaries = std::unordered_map<types::boundary_id, T>;

    /**
     * \brief Represents a mapping between boundaries (represented by boundary IDs) and the corresponding Dirichlet boundary condition.
     * \tparam dim The dimensionality of the space the ADR problem is living in.
     */
    template <int dim>
    using DirichletBoundaries = Boundaries<const DirichletBoundary<dim> *>;

    /**
     * \brief Represents a mapping between boundaries (represented by boundary IDs) and the corresponding Neumann boundary condition.
     * \tparam dim The dimensionality of the space the ADR problem is living in.
     */
    template <int dim>
    using NeumannBoundaries = Boundaries<const NeumannBoundary<dim> *>;
};