"""
    TransmissiveBC <: AbstractBoundaryCondition

Transmissive boundary condition. Ghost state equals the adjacent interior state.
"""
struct TransmissiveBC <: AbstractBoundaryCondition end


"""
    ghost_state(bc::AbstractBoundaryCondition, W::AbstractState) -> AbstractState

Compute the ghost state from the adjacent interior cell state.
"""
function ghost_state end

ghost_state(::TransmissiveBC, W::AbstractState) = W


"""
    BoundaryFace

Descriptor for a single boundary face.

# Fields
- `ghost_idx::Int`: index of the ghost cell to fill
- `interior_idx::Int`: index of the adjacent interior cell
- `bc::AbstractBoundaryCondition`: boundary condition at this face
"""
struct BoundaryFace
    ghost_idx::Int
    interior_idx::Int
    bc::AbstractBoundaryCondition
end


"""
    make_boundary_faces(grid::UniformGrid1D, bc::AbstractBoundaryCondition) -> Vector{BoundaryFace}

Construct boundary faces for a 1D domain described by `grid`, all using the same `bc`.

Ghost cell indexing (OffsetArray convention):
- Physical cells: `1:N`
- Left ghost cells: `1-ng : 0`
- Right ghost cells: `N+1 : N+ng`
"""
function make_boundary_faces(grid::UniformGrid1D, bc::AbstractBoundaryCondition)
    N = grid.N
    ng = grid.ghost_cells # number of ghost cells
    faces = BoundaryFace[]
    for g in 1:ng
        push!(faces, BoundaryFace(g - ng, 1, bc))       # left boundary
        push!(faces, BoundaryFace(N + g, N, bc))        # right boundary
    end
    return faces
end


"""
    fill_ghost_cells!(U_padded, faces::Vector{BoundaryFace})

Fill ghost cells on a padded array `U_padded` by iterating over `faces`. Each face specifies the ghost index, the adjacent interior index, and the
boundary condition.
"""
function fill_ghost_cells!(U_padded, faces::Vector{BoundaryFace})
    for face in faces
        U_padded[face.ghost_idx] = ghost_state(face.bc, U_padded[face.interior_idx])
    end
end


"""
    apply_bc!(W_arr, W_padded, grid::UniformGrid1D, boundaries::Vector{BoundaryFace})

Copy primitive states from `W_arr` (indexed `1:grid.N`) into the
pre-allocated padded array `W_padded`, then fill ghost cells by applying
boundary conditions.

The caller is responsible for pre-allocating `W_padded` with the correct
`OffsetArray` indexing (physical cells at `1:grid.N`, ghosts at `1-ng:0`
and `N+1:N+ng`).
"""
function apply_bc!(W_arr, W_padded, grid::UniformGrid1D,
                   boundaries::Vector{BoundaryFace})
    for i in 1:grid.N
        W_padded[i] = W_arr[i]
    end
    fill_ghost_cells!(W_padded, boundaries)
    return W_padded
end
