"""
    SolverConfig{S<:AbstractRiemannSolver, T<:Real}

Configuration bundle for a finite-volume simulation.

# Fields
- `solver::S`: the Riemann solver used at interfaces
- `limiter`: slope limiter (`nothing` for first-order)
- `cfl::T`: CFL number (typically 0.5–0.9)
- `max_time::T`: final simulation time
- `max_steps::Int`: maximum allowed time steps (safety guard)
- `ramp_steps::Int`: number of initial steps using a reduced CFL (default 5)
- `ramp_cfl::T`: reduced CFL number used during ramp-up (default 0.2)
"""
struct SolverConfig{S<:AbstractRiemannSolver, T<:Real}
    solver::S
    limiter
    cfl::T
    max_time::T
    max_steps::Int
    ramp_steps::Int
    ramp_cfl::T
end

function SolverConfig(solver::S, cfl::T, max_time::T, max_steps::Integer;
                      limiter=nothing,
                      ramp_steps::Integer=5,
                      ramp_cfl::T=convert(T, 0.2)) where {S<:AbstractRiemannSolver, T<:Real}
    return SolverConfig{S, T}(solver, limiter, cfl, max_time, Int(max_steps),
                              Int(ramp_steps), ramp_cfl)
end
