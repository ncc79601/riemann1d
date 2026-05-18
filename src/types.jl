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

Abstract supertype for Riemann problem solutions. Can be sampled at arbitrary ``(x, t)`` via [`sample_solution`](@ref).
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
    AbstractLimiter

Abstract supertype for slope limiter implementations.
"""
abstract type AbstractLimiter end


"""
    WaveSpeedMethod

Enumeration of wave speed estimation methods.

- `Physical`: use physical eigenvalues ``\\lambda_\\pm = u \\pm a``
"""
@enum WaveSpeedMethod Physical


"""
    SolverConfig{S<:AbstractRiemannSolver, T<:Real}

Configuration bundle for a finite-volume simulation.

# Fields
- `solver::S`: the Riemann solver used at interfaces
- `limiter`: slope limiter (`nothing` for first-order)
- `cfl::T`: CFL number (typically 0.5–0.9)
- `max_time::T`: final simulation time
- `max_steps::Int`: maximum allowed time steps (safety guard)
"""
struct SolverConfig{S<:AbstractRiemannSolver, T<:Real}
    solver::S
    limiter          # Union{AbstractLimiter, Nothing} — will be typed later
    cfl::T
    max_time::T
    max_steps::Int
end

function SolverConfig(solver::S, cfl::T, max_time::T, max_steps::Integer;
                      limiter=nothing) where {S<:AbstractRiemannSolver, T<:Real}
    return SolverConfig{S, T}(solver, limiter, cfl, max_time, Int(max_steps))
end
