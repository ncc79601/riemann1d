@enum HLLCEstimateMethod begin
    RoeEstimate
end


"""
    HLLC <: AbstractRiemannSolver

HLLC Solver. Only consider the fastest left and right waves in the non-linear wave structure, along with contact discontinuity in the middle.
"""
struct HLLC <: AbstractRiemannSolver
    estimate_method::HLLCEstimateMethod
end
HLLC(; estimate_method::HLLCEstimateMethod = RoeEstimate) = HLLC(estimate_method)


"""
    compute_HLLC_wave_speeds(W_L, W_R, eos, method)

Compute the wave speeds ``S_L``, ``S_R`` for the HLLC solver using the specified method (default is Roe estimate).

# Returns:
- `(S_L, S_R)`: left and right wave speeds
"""
function compute_HLLC_wave_speeds(W_L::PrimitiveState, W_R::PrimitiveState, eos::PerfectGasEOS, method::HLLCEstimateMethod = RoeEstimate)
    # compute wave speeds S_L, S_R, and S_M using the HLLC solver
    u_L, u_R = W_L.u, W_R.u
    a_L = sound_speed(W_L, eos)
    a_R = sound_speed(W_R, eos)

    if method == RoeEstimate
        _, ũ, _, ã = Roe_average(W_L, W_R, eos)
        S_L = min(u_L - a_L, ũ - ã)
        S_R = max(u_R + a_R, ũ + ã)
    else
        throw(ArgumentError("Unsupported HLLC wave speed method: $(method)"))
    end

    return S_L, S_R
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
    eos::PerfectGasEOS,
)
    ρ_L, u_L, p_L = W_L.ρ, W_L.u, W_L.p
    ρ_R, u_R, p_R = W_R.ρ, W_R.u, W_R.p

    S_L, S_R = compute_HLLC_wave_speeds(W_L, W_R, eos, solver.estimate_method)
    # compute wave speeds and intermediate states using the HLLC solver
    
    S★ = ((p_R-p_L) + (ρ_L * u_L * (S_L-u_L) - ρ_R * u_R * (S_R-u_R))) / (ρ_L * (S_L-u_L) - ρ_R * (S_R-u_R))

    # branch select
    if 0 < S_L
         # left state flux
        return Flux(W_L, eos)
    
    elseif S_L <= 0 < S★
        # left star region flux
        U_L = primitive_to_conserved(W_L, eos)
        U★_L = ρ_L * ((S_L-u_L) / (S_L-S★)) * ConservedState(
            1.0,
            S★,
            U_L.E/ρ_L + (S★-u_L) * (S★ + (p_L / (ρ_L*(S_L-u_L))))
        )
        return Flux(W_L, eos) + S_L * (U★_L - U_L)
    
    elseif S★ <= 0 < S_R
        # right star region flux
        U_R = primitive_to_conserved(W_R, eos)
        U★_R = ρ_R * ((S_R-u_R) / (S_R-S★)) * ConservedState(
            1.0,
            S★,
            U_R.E/ρ_R + (S★-u_R) * (S★ + (p_R / (ρ_R*(S_R-u_R))))
        )
        return Flux(W_R, eos) + S_R * (U★_R - U_R)
    
    else # S_R <= 0
        return Flux(W_R, eos)
    end

end