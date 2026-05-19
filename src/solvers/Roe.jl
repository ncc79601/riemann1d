@enum RoeEntropyFix begin
    None
    HartenHyman
end


"""
    RoeSolver <: AbstractRiemannSolver

Roe's approximate Riemann solver.
"""
struct RoeSolver <: AbstractRiemannSolver
    entropy_fix_method::RoeEntropyFix
end
RoeSolver(; entropy_fix_method::RoeEntropyFix = None) = RoeSolver(entropy_fix_method)


"""
    Roe_average(W_L::PrimitiveState, W_R::PrimitiveState, eos::AbstractEOS)

Compute Roe-averaged state variables (ρ̃, ũ, H̃, ã) for the left and right primitive states W_L and W_R, given the equation of state eos.
"""
function Roe_average(W_L::PrimitiveState, W_R::PrimitiveState, eos::AbstractEOS)
    ρ_L, u_L, p_L = W_L.ρ, W_L.u, W_L.p
    ρ_R, u_R, p_R = W_R.ρ, W_R.u, W_R.p
    H_L = total_enthalpy(W_L, eos)
    H_R = total_enthalpy(W_R, eos)
    γ = eos.γ

    q_L = √(ρ_L)
    q_R = √(ρ_R)

    ρ̃ = q_L * q_R
    ũ = (q_L * u_L + q_R * u_R) / (q_L + q_R)
    H̃ = (q_L * H_L + q_R * H_R) / (q_L + q_R)
    ã = √((γ-1) * (H̃ - 0.5 * ũ^2))

    return ρ̃, ũ, H̃, ã
end


"""
    compute_numerical_flux(solver::RoeSolver, W_L, W_R, eos)
Roe flux: linearize the problem around the Roe-averaged state and compute the flux using the eigenvalues and eigenvectors of the Jacobian matrix (Roe-Pike method), but can violate entropy condition. Thus an entropy fix (e.g. Harten-Hyman) is often applied.

# Reference:
RmSv-11.3
"""
function compute_numerical_flux(
    solver::RoeSolver,
    W_L::PrimitiveState,
    W_R::PrimitiveState,
    eos::PerfectGasEOS,
)
    ρ_L, u_L, p_L = W_L.ρ, W_L.u, W_L.p
    ρ_R, u_R, p_R = W_R.ρ, W_R.u, W_R.p
    γ = eos.γ

    ρ̃, ũ, H̃, ã = Roe_average(W_L, W_R, eos)

    # compute eigenvalues
    λ̃₁ = ũ - ã
    λ̃₂ = ũ
    λ̃₅ = ũ + ã

    # compute fluxes for left and right states
    F_L = Flux(W_L, eos)
    F_R = Flux(W_R, eos)

    # compute differences in conserved variables
    U_L = primitive_to_conserved(W_L, eos)
    U_R = primitive_to_conserved(W_R, eos)
    ΔU = U_R - U_L

    # compute wave modes (right eigen vectors)
    K̃¹ = ConservedState(1, ũ - ã, H̃ - ũ*ã  )
    K̃² = ConservedState(1, ũ,     0.5 * ũ^2)
    K̃⁵ = ConservedState(1, ũ + ã, H̃ + ũ*ã  )

    # compute wave strengths
    Δu₁, Δu₂, Δu₅ = ΔU.ρ, ΔU.ρu, ΔU.E

    α̃₂ = (γ-1)/ (ã^2) * (Δu₁ * (H̃-ũ^2) + ũ * Δu₂ - Δu₅)
    α̃₁ = 1/(2ã) * (Δu₁*(ũ+ã) - Δu₂ - ã * α̃₂)
    α̃₅ = Δu₁ - (α̃₁ + α̃₂)

    # entropy fix
    # TODO

    # compute Roe flux using the eigenvalues and differences in conserved variables
    return 0.5 * (F_L + F_R) - 0.5 * (α̃₁ * abs(λ̃₁) * K̃¹ + α̃₂ * abs(λ̃₂) * K̃² + α̃₅ * abs(λ̃₅) * K̃⁵)
end