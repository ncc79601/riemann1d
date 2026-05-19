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
