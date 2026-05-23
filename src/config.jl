"""
    SolverConfig{S, R, L, I, T}

Configuration bundle for a finite-volume simulation.

# Fields
- `solver::S`: the Riemann solver used at interfaces
- `reconstruction::R`: reconstruction method (e.g. `SecondOrderReconstruct()`)
- `limiter::L`: slope limiter (`NoLimiter()` for first-order)
- `integrator::I`: time integrator (`ExplicitEuler()` or `TVDRK2()`)
- `cfl::T`: CFL number (typically 0.4–0.9)
- `max_time::T`: final simulation time
- `max_steps::Int`: maximum allowed time steps (safety guard)
- `init_steps::Int`: number of initial steps with reduced CFL (default 5)
- `init_cfl::T`: reduced CFL number during ramp-up (default 0.2)
"""
struct SolverConfig{
    S <: AbstractRiemannSolver,
    R <: AbstractReconstructMethod,
    L <: AbstractLimiter,
    I <: AbstractIntegrator,
    T <: Real
}
    solver::S
    reconstruction::R
    limiter::L
    integrator::I
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
        reconstruction::R = SecondOrderReconstruct(),
        limiter::L = SuperBeeLimiter(),
        integrator::I = TVDRK2(),
        init_steps::Integer = 5,
        init_cfl::T = convert(T, 0.2)
) where {
        S <: AbstractRiemannSolver,
        T <: Real,
        I <: AbstractIntegrator,
        R <: AbstractReconstructMethod,
        L <: AbstractLimiter
}
    # solver config checks
    # reconstruction & limiter
    if isa(reconstruction, SecondOrderReconstruct) && isa(limiter, NoLimiter)
        throw(
            ArgumentError(
            "If second order reconstruction is used, a valid limiter must be specified",
        ),
        )
    end

    # cfl & integrator
    if cfl > convert(T, 1.0)
        throw(ArgumentError("CFL number larger than 1"))
    end

    # limiter & integrator
    if isa(reconstruction, SecondOrderReconstruct) && !(isa(limiter, NoLimiter))
        if isa(integrator, ExplicitEuler)
            throw(ArgumentError(
                "In semi-discrete schemes, second order reconstruction cannot be used in conjunction with explicit Euler integrator (otherwise unconditioned unstable). Consider using TVD-RK2 integrator instead"
            ))
        end
        # check TVD criterion
        ξ_max = ξ(1000.0, limiter)
        cfl_TVD = 1 / (1 + 0.5 * ξ_max)
        if cfl > convert(T, cfl_TVD)
            @warn "Current CFL number may violate TVD criterion and cause oscillation. Consider using smaller CFL numbers"
        end
    end

    # cfl & init_cfl
    if init_cfl > cfl
        @warn "init_cfl is larger than cfl"
    end

    return SolverConfig{S, R, L, I, T}(
        solver,
        reconstruction,
        limiter,
        integrator,
        cfl,
        max_time,
        Int(max_steps),
        Int(init_steps),
        init_cfl
    )
end
