struct NoReconstruct <: AbstractReconstructMethod end
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
    method::Union{NoReconstruct, Nothing},
    limiter::Union{AbstractLimiter, Nothing},
)
    #TODO debug
    # @warn "No reconstruction"
    N = grid.N
    for i in 0:N+1
        W_L[i] = W_R[i] = W_padded[i]
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
    #TODO debug
    # @warn "Second order reconstruction"
    N  = grid.N
    ng = grid.ghost_cells

    # extract component arrays
    # TODO: too slow, needs refactor
    ρ_arr = [W_padded[k].ρ for k in 1-ng:N+ng]
    ρ_arr = OffsetArray(ρ_arr, 1-ng:N+ng)
    
    u_arr = [W_padded[k].u for k in 1-ng:N+ng]
    u_arr = OffsetArray(u_arr, 1-ng:N+ng)
    
    p_arr = [W_padded[k].p for k in 1-ng:N+ng]
    p_arr = OffsetArray(p_arr, 1-ng:N+ng)

    #TODO: debug
    # @warn "after extracting components"
    # @show ρ_arr
    # @show u_arr
    # @show p_arr

    # reconstruction, including ghost cell 0 and N+1
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
safe_slope(Δ::Real) = copysign(max(abs(Δ), eps(typeof(Δ))), Δ)


"""
    muscl_scalar(arr, j, limiter)
"""
function muscl_scalar(u::AbstractVector, i::Int, limiter::AbstractLimiter)
    Δ_L = u[i] - u[i-1]
    Δ_R = u[i+1] - u[i]
    Δ = Δ_R # right slope as base

    #TODO debug
    # @show (Δ_L, Δ_R, Δ)

    r = Δ_L / safe_slope(Δ_R)
    #TODO debug
    # @show r
    Δ̄ = Δ * ξ(r, limiter)
    
    u_L = u[i] - 0.5 * Δ̄
    u_R = u[i] + 0.5 * Δ̄

    #TODO debug
    # println("reconstructed #$i: u_L=$u_L, u_R=$u_R")

    return u_L, u_R
end
