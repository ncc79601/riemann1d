

"""
    Roe_average(W_L::PrimitiveState, W_R::PrimitiveState, eos::AbstractEOS)

Compute Roe-averaged state variables (ρ̃, ũ, ã) for the left and right primitive states W_L and W_R, given the equation of state eos.
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

    return ρ̃, ũ, ã
end