"""
    UniformGrid1D{T} <: AbstractGrid

Uniform grid representation for 1D spatial domain.

# Fields
- `xmin::T`: min x-coordinate of the domain
- `xmax::T`: max x-coordinate of the domain
- `N::Int`: number of cells in the grid
- `Δx::T`: cell width
- `x_centers::Vector{T}`: coordinates of cell centers
- `x_faces::Vector{T}`: coordinates of cell faces (edges)

# Constructors
- `UniformGrid1D(xmin::Real, xmax::Real, N::Int; ghost_cells::Int=NaN)`: creates a uniform grid with specified bounds and number of cells, auto-promoting numeric types
"""
struct UniformGrid1D{T} <: AbstractGrid
    xmin::T
    xmax::T
    N::Int # number of cells
    ghost_cells::Int # number of ghost cells on each boundary
    Δx::T # cell width
    x_centers::Vector{T} # cell center coordinates
    x_faces::Vector{T} # cell edge coordinates
end

function UniformGrid1D(xmin::Real, xmax::Real, N::Int; ghost_cells::Int = 1)
    xmin_prom, xmax_prom = promote(xmin, xmax)
    T = typeof(xmin_prom)
    Δx = (xmax_prom - xmin_prom) / T(N)
    x_centers = range(start = xmin_prom + Δx/2, stop = xmax_prom - Δx/2, length = N)
    x_faces = range(start = xmin_prom, stop = xmax_prom, length = N+1)

    # check number for ghost cells
    if ghost_cells <= 0
        throw(ArgumentError("number of ghost cells must be positive"))
    elseif ghost_cells > 2
        @warn "creating grid with more than 2 ghost cells per side is not recommended for 1D problems."
    end

    return UniformGrid1D{T}(
        xmin_prom,
        xmax_prom,
        N,
        ghost_cells,
        Δx,
        collect(x_centers),
        collect(x_faces)
    )
end
