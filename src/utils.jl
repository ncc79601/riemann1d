"""
    SolverConfig{S<:AbstractRiemannSolver, T<:Real}

Configuration bundle for a finite-volume simulation.

# Fields # TODO: modify
- `solver::S`: the Riemann solver used at interfaces
- `limiter`: slope limiter (`nothing` for first-order)
- `cfl::T`: CFL number (typically 0.5–0.9)
- `max_time::T`: final simulation time
- `max_steps::Int`: maximum allowed time steps (safety guard)
- `init_steps::Int`: number of initial steps using a reduced CFL (default 5)
- `init_cfl::T`: reduced CFL number used during initial setup (default 0.2)
"""
struct SolverConfig{
    S<:AbstractRiemannSolver,
    L<:AbstractLimiter,
    R<:AbstractReconstructMethod,
    T<:Real
}
    solver::S
    reconstruction::R
    limiter::L
    cfl::T
    max_time::T
    max_steps::Int
    init_steps::Int
    init_cfl::T
end

function SolverConfig(
    solver::S,
    cfl::T,
    max_time::T,
    max_steps::Integer;
    # kwargs
    reconstruction::R=SecondOrderReconstruct(),
    limiter::L=SuperBeeLimiter(),
    init_steps::Integer=5,
    init_cfl::T=convert(T, 0.2)
) where {S<:AbstractRiemannSolver, T<:Real, R<:AbstractReconstructMethod, L<:AbstractLimiter}
    return SolverConfig{S, L, R, T}(
        solver, reconstruction, limiter,
        cfl, max_time, Int(max_steps), Int(init_steps), init_cfl
    )
end
