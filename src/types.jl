"""
    AbstractState

Abstract supertype for all thermodynamic state representations.
"""
abstract type AbstractState end

"""
    AbstractFlux

Abstract supertype for flux representations.
"""
abstract type AbstractFlux end

"""
    AbstractEOS

Abstract supertype for equations of state.
"""
abstract type AbstractEOS end

"""
    AbstractRiemannSolution

Abstract supertype for Riemann problem solutions. Can be sampled at arbitrary ``(x, t)`` via [`sample_exact_solution`](@ref).
"""
abstract type AbstractRiemannSolution end

"""
    AbstractRiemannSolver

Abstract supertype for Riemann solvers.
"""
abstract type AbstractRiemannSolver end

"""
    AbstractGrid

Abstract supertype for grid representations.
"""
abstract type AbstractGrid end

"""
    AbstractBoundaryCondition

Abstract supertype for boundary condition representations.
"""
abstract type AbstractBoundaryCondition end

"""
    AbstractReconstructMethod

Abstract supertype for reconstruction methods.
"""
abstract type AbstractReconstructMethod end

"""
    AbstractLimiter

Abstract supertype for slope limiter implementations.
"""
abstract type AbstractLimiter end

"""
    AbstractIntegrator

Abstract supertype for time integrator.
"""
abstract type AbstractIntegrator end
