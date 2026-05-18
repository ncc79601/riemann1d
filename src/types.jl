"""
    AbstractState

Abstract supertype for all thermodynamic state representations.
"""
abstract type AbstractState end


"""
    AbstractEOS

Abstract supertype for equations of state.
"""
abstract type AbstractEOS end


"""
    AbstractRiemannSolution

Abstract supertype for Riemann problem solutions. Can be sampled at arbitrary ``(x, t)`` via [`sample_solution`](@ref).
"""
abstract type AbstractRiemannSolution end

