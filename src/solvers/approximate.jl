"""
    PVRS <: AbstractRiemannSolver

Primitive-variable Riemann solver (PVRS). Uses linearization assumption and treats all waves as discontinuities. Not accurate, but fast.
"""
struct PVRS <: AbstractRiemannSolver end


"""
    compute_numerical_flux(solver::PVRS, W_L, W_R, eos)

PVRS flux: solve the Riemann problem and evaluate the physical flux at ``x/t = 0``. Evaluate the flux by comparing ``S_L=u_L-a_L``, ``u_*``, ``S_R=u_R+a_R``, and ``0`` (treat all waves as discontinuities).

```math

```

# Reference
RmSv-9.3
"""
function compute_numerical_flux(
    solver::PVRS,
    W_L::PrimitiveState,
    W_R::PrimitiveState,
    eos::PerfectGasEOS,
)
    ρ_L, u_L, p_L = W_L.ρ, W_L.u, W_L.p
    ρ_R, u_R, p_R = W_R.ρ, W_R.u, W_R.p
    a_L = sound_speed(W_L, eos)
    a_R = sound_speed(W_R, eos)

    ρ̄    = 0.5 * (ρ_L + ρ_R)
    ā    = 0.5 * (a_L + a_R)
    p★   = 0.5 * (p_L + p_R) + 0.5 * (u_L - u_R) * ρ̄ * ā
    u★   = 0.5 * (u_L + u_R) + 0.5 * (p_L - p_R) / (ρ̄ * ā)
    ρ★_L = ρ_L + (u_L - u★) * ρ̄ / ā
    ρ★_R = ρ_R + (u★ - u_R) * ρ̄ / ā

    # sample the solution at x/t = 0
    if 0 <= u_L - a_L # left data state
        return Flux(W_L, eos)
    elseif u_L - a_L < 0 <= u★ # left star-region
        return Flux(PrimitiveState(ρ=ρ★_L, u=u★, p=p★), eos)
    elseif u★ < 0 <= u_R + a_R # right star-region
        return Flux(PrimitiveState(ρ=ρ★_R, u=u★, p=p★), eos)
    else # right data state
        return Flux(W_R, eos)
    end
end


"""
    TRRS <: AbstractRiemannSolver

Two-rarefaction Riemann solver (TRRS).
"""
struct TRRS <: AbstractRiemannSolver end


"""
    compute_numerical_flux(solver::TRRS, W_L, W_R, eos)

TRRS flux: # TODO

# Reference:
RmSv-9.4
"""
function compute_numerical_flux(
    solver::TRRS,
    W_L::PrimitiveState,
    W_R::PrimitiveState,
    eos::PerfectGasEOS,
)
    ρ_L, u_L, p_L = W_L.ρ, W_L.u, W_L.p
    ρ_R, u_R, p_R = W_R.ρ, W_R.u, W_R.p
    a_L = sound_speed(W_L, eos)
    a_R = sound_speed(W_R, eos)

    γ = eos.γ
    z = (γ - 1) / (2γ)

    P_LR = (p_L / p_R) ^ z
    u★ = (P_LR * u_L/a_L + u_R/a_R + 2 * (P_LR-1)/(γ-1)) / (P_LR/a_L + 1/a_R)
    p★ = (
        p_L * (1 + (γ-1)/(2a_L) * (u_L - u★)) ^ (1/z) +
        p_R * (1 + (γ-1)/(2a_R) * (u★ - u_R)) ^ (1/z)
    ) / 2
    ρ★_L = ρ_L * (p★ / p_L) ^ (1/γ)
    ρ★_R = ρ_R * (p★ / p_R) ^ (1/γ)

    head_L = u_L - a_L
    tail_L = u★ - sound_speed(PrimitiveState(ρ=ρ★_L, u=u★, p=p★), eos)
    head_R = u_R + a_R
    tail_R = u★ + sound_speed(PrimitiveState(ρ=ρ★_R, u=u★, p=p★), eos)

    # utilize tools from src/solvers/exact.jl to sample the solution at x/t = 0
    wave_structure_L = NonlinearWaveStructure(
        Rarefaction,
        ρ★_L,
        NaN,
        head_L,
        tail_L
    )
    wave_structure_R = NonlinearWaveStructure(
        Rarefaction,
        ρ★_R,
        NaN,
        head_R,
        tail_R
    )
    sol = ExactRiemannSolution(
        eos, W_L, W_R, p★, u★,
        wave_structure_L, wave_structure_R,
    )
    return sample_exact_solution(0.0, 1.0, sol) |> w -> Flux(w, eos)
end
