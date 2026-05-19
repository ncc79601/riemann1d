"""
    reconstruct_face_values(W_padded, limiter, grid::UniformGrid1D) -> (W_L, W_R)

Reconstruct left and right face values from a padded cell-averaged array.

When `limiter` is `nothing` (first-order), face values are simply the adjacent
cell-averaged states:

``W_{i-1/2}^R = W_i^{\\,n}, \\quad W_{i+1/2}^L = W_i^{\\,n}``

When `limiter` is an `AbstractLimiter`, perform MUSCL reconstruction (future).

# Arguments
- `W_padded`: padded array indexed `(1-ghost):(N+ghost)` where `1:N` are physical
- `limiter`: `nothing` (first-order) or `AbstractLimiter` (MUSCL)
- `grid::UniformGrid1D`

# Returns
- `W_L::Vector{PrimitiveState}`: left state of each interface (length `N+1`)
- `W_R::Vector{PrimitiveState}`: right state of each interface (length `N+1`)
"""
function reconstruct_face_values(W_padded, limiter, grid::UniformGrid1D)
    return reconstruct_first_order(W_padded, grid)
end


function reconstruct_first_order(W_padded, grid::UniformGrid1D)
    N = grid.N
    W_L = Vector{PrimitiveState}(undef, N + 1)
    W_R = Vector{PrimitiveState}(undef, N + 1)

    for i in 1:(N + 1)
        W_L[i] = W_padded[i-1]   # left  of face i is cell i-1
        W_R[i] = W_padded[i]     # right of face i is cell i
    end

    return W_L, W_R
end
