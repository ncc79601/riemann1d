"""
    HLLCWaveSpeedMethod

Abstract supertype for HLLC wave speed estimation methods.
"""
abstract type HLLCWaveSpeedMethod end

"""
    RoeEstimate <: HLLCWaveSpeedMethod

Roe-Einfeldt wave-speed estimate: use the Roe-averaged eigenvalues as bounds and take the minimum / maximum with the physical eigenvalues for robustness (better compatibility with rarefactions).

``S_L = \\text{min}(u_L - a_L, \\; \\tilde{u} - \\tilde{a})`` \\
``S_R = \\text{max}(u_R + a_R, \\; \\tilde{u} + \\tilde{a})``
"""
struct RoeEstimate <: HLLCWaveSpeedMethod end

"""
    DavisEstimate <: HLLCWaveSpeedMethod

Davis wave-speed estimate: take the minimum and maximum of the physical eigenvalues on both sides. Simpler and more diffusive than `RoeEstimate`.

``S_L = \\text{min}(u_L - a_L, \\; u_R - a_R)`` \\
``S_R = \\text{max}(u_L + a_L, \\; u_R + a_R)``
"""
struct DavisEstimate <: HLLCWaveSpeedMethod end

"""
    wave_speed_estimate(estimator, W_L, W_R, eos) -> (S_L, S_R)

Compute ``S_L`` and ``S_R`` for the HLLC solver using the given estimator. Dispatches on the type of `estimator`.
"""
function wave_speed_estimate end

function wave_speed_estimate(
        ::RoeEstimate,
        W_L::PrimitiveState,
        W_R::PrimitiveState,
        eos::PerfectGasEOS
)
    u_L, u_R = W_L.u, W_R.u
    a_L = sound_speed(W_L, eos)
    a_R = sound_speed(W_R, eos)
    _, ũ, _, ã = Roe_average(W_L, W_R, eos)
    S_L = min(u_L - a_L, ũ - ã)
    S_R = max(u_R + a_R, ũ + ã)
    return S_L, S_R
end

function wave_speed_estimate(
        ::DavisEstimate,
        W_L::PrimitiveState,
        W_R::PrimitiveState,
        eos::PerfectGasEOS
)
    a_L = sound_speed(W_L, eos)
    a_R = sound_speed(W_R, eos)
    S_L = min(W_L.u - a_L, W_R.u - a_R)
    S_R = max(W_L.u + a_L, W_R.u + a_R)
    return S_L, S_R
end

"""
    HLLC <: AbstractRiemannSolver

HLLC Solver. Only consider the fastest left and right waves in the non-linear wave structure, along with contact discontinuity in the middle.
"""
struct HLLC <: AbstractRiemannSolver
    estimate_method::HLLCWaveSpeedMethod
end
function HLLC(; estimate_method::HLLCWaveSpeedMethod = RoeEstimate())
    HLLC(estimate_method)
end

"""
    compute_numerical_flux(solver::HLLC, W_L, W_R, eos)
HLLC flux: approximate the solution by a 3-wave model (left wave, contact discontinuity, right wave) and compute the flux based on the wave speeds and intermediate states. More accurate than HLL for contact discontinuities, but more expensive to compute.

# Reference:
RmSv-10.4
"""
function compute_numerical_flux(
        solver::HLLC,
        W_L::PrimitiveState,
        W_R::PrimitiveState,
        eos::PerfectGasEOS
)
    ρ_L, u_L, p_L = W_L.ρ, W_L.u, W_L.p
    ρ_R, u_R, p_R = W_R.ρ, W_R.u, W_R.p

    S_L, S_R = wave_speed_estimate(solver.estimate_method, W_L, W_R, eos)

    S★ = ((p_R - p_L) + (ρ_L * u_L * (S_L - u_L) - ρ_R * u_R * (S_R - u_R))) /
         (ρ_L * (S_L - u_L) - ρ_R * (S_R - u_R))

    # branch select
    if 0 < S_L
        # left state flux
        return Flux(W_L, eos)

    elseif S_L <= 0 < S★
        # left star region flux
        U_L = primitive_to_conserved(W_L, eos)
        U★_L = ρ_L *
               ((S_L - u_L) / (S_L - S★)) *
               ConservedState(
                   1.0,
                   S★,
                   U_L.E / ρ_L + (S★ - u_L) * (S★ + (p_L / (ρ_L * (S_L - u_L))))
               )
        return Flux(W_L, eos) + S_L * (U★_L - U_L)

    elseif S★ <= 0 < S_R
        # right star region flux
        U_R = primitive_to_conserved(W_R, eos)
        U★_R = ρ_R *
               ((S_R - u_R) / (S_R - S★)) *
               ConservedState(
                   1.0,
                   S★,
                   U_R.E / ρ_R + (S★ - u_R) * (S★ + (p_R / (ρ_R * (S_R - u_R))))
               )
        return Flux(W_R, eos) + S_R * (U★_R - U_R)

    else # S_R <= 0
        return Flux(W_R, eos)
    end
end
