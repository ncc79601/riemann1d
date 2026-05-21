struct FirstOrderReconstruct <: AbstractReconstructMethod end
struct SecondOrderReconstruct <: AbstractReconstructMethod end


"""
    reconstruct!(W_L, W_R, W_padded, grid, method, limiter)

Reconstruct left and right face values from a padded cell-averaged array.
`W_padded` is an `OffsetArray` with physical cells at `1:grid.N` and ghost
cells at `1-ng:0` (left) and `N+1:N+ng` (right).

`W_L` and `W_R` are pre-allocated vectors of length `N`(#FIXME) receiving the
left and right face values for each interface.

`limiter = nothing` is treated as first-order (no reconstruction).
"""
function reconstruct!(
    W_L::AbstractVector{PrimitiveState},
    W_R::AbstractVector{PrimitiveState},
    W_padded,
    grid::UniformGrid1D,
    method::FirstOrderReconstruct,
    limiter::Union{AbstractLimiter, Nothing},
)
    N = grid.N
    for i in 0:N+1
        W_L[i] = W_padded[i - 1]
        W_R[i] = W_padded[i]
    end
end


function reconstruct!(
    W_L::AbstractVector{PrimitiveState},
    W_R::AbstractVector{PrimitiveState},
    W_padded,
    grid::UniformGrid1D,
    method::SecondOrderReconstruct,
    limiter::AbstractLimiter,
)
    N  = grid.N
    ng = grid.ghost_cells

    # extract component arrays
    ρ_arr = [W_padded[k].ρ for k in 1-ng:N+ng]
    ρ_arr = OffsetArray(ρ_arr, 1-ng:N+ng)
    
    u_arr = [W_padded[k].u for k in 1-ng:N+ng]
    u_arr = OffsetArray(u_arr, 1-ng:N+ng)
    
    p_arr = [W_padded[k].p for k in 1-ng:N+ng]
    p_arr = OffsetArray(p_arr, 1-ng:N+ng)

    # reconstruct components
    for i in 0:N+1
        ρ_L, ρ_R = muscl_scalar(ρ_arr, i, limiter)
        u_L, u_R = muscl_scalar(u_arr, i, limiter)
        p_L, p_R = muscl_scalar(p_arr, i, limiter)

        W_L[i] = PrimitiveState(ρ_L, u_L, p_L)
        W_R[i] = PrimitiveState(ρ_R, u_R, p_R)
    end
end


"""
    safe_slope(Δ::Real)

Return `abs(Δ)` clamped to a minimum of `eps(typeof(Δ))` so that slope-ratio
division never divides by zero.
"""
safe_slope(Δ::Real) = sign(Δ) * max(abs(Δ), eps(typeof(Δ)))


"""
    _muscl_scalar(arr, j, limiter)

Return `(w_L, w_R)` — the MUSCL-extrapolated left and right face values of a
scalar field at the face between cells `j` and `j+1` (1-indexed into the padded
component array `arr`).

Formula:
    w_L = arr[j] + ½ φ(r_j) ⋅ (arr[j] - arr[j-1])
    w_R = arr[j+1] - ½ φ(r_{j+1}) ⋅ (arr[j+2] - arr[j+1])
"""
function muscl_scalar(arr::AbstractVector, j::Int, limiter::AbstractLimiter)
    Δ_l = arr[j] - arr[j-1]
    Δ_r = arr[j+1] - arr[j]
    Δ = Δ_r

    r = Δ_l / safe_slope(Δ_r)
    Δ̄ = Δ * ξ(r, limiter)
    
    w_L = arr[j] - 0.5 * Δ̄
    w_R = arr[j] + 0.5 * Δ̄

    return w_L, w_R
end
