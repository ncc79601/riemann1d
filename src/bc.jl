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
    apply_bc!(U, faces::Vector{BoundaryFace})

Apply boundary conditions to the state array `U` by filling each ghost cell
specified in `faces`.
"""
function apply_bc!(U, faces::Vector{BoundaryFace})
    for face in faces
        U[face.ghost_idx] = ghost_state(face.bc, U[face.interior_idx])
    end
end
