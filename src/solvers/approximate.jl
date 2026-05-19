"""
    PVRS <: AbstractRiemannSolver

First-order Godunov scheme.  Uses the exact Riemann solver to compute
numerical fluxes at each cell interface.
"""
struct PVRS <: AbstractRiemannSolver end


"""
    compute_numerical_flux(solver::PVRS, W_L, W_R, eos)

PVRS flux: solve the  Riemann problem and evaluate the physical flux at ``x/t = 0``. Evaluate the flux by comparing ``S_L=u_L-a_L``, ``u_*``, ``S_R=u_R+a_R``, and ``0`` (treat all waves as discontinuities).

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

    ρ̄ = 0.5 * (ρ_L + ρ_R)
    ā = 0.5 * (a_L + a_R)
    p★ = 0.5 * (p_L + p_R) + 0.5 * (u_L - u_R) * ρ̄ * ā
    u★ = 0.5 * (u_L + u_R) + 0.5 * (p_L - p_R) / (ρ̄ * ā)
    ρ★_L = ρ_L + (u_L - u★) * ρ̄ / ā
    ρ★_R = ρ_R + (u★ - u_R) * ρ̄ / ā

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