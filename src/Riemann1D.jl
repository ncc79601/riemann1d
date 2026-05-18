"""
    Riemann1D

Exact and approximate Riemann solvers for the one-dimensional
compressible Euler equations.
"""
module Riemann1D

using ForwardDiff
using StructArrays
using OffsetArrays

# abstract type hierarchy
include("types.jl")
export AbstractState, AbstractEOS, AbstractRiemannSolution, AbstractRiemannSolver

# equations of state
include("eos.jl")
export PerfectGasEOS

# primitive / conserved variables, flux structs and conversions
include("states.jl")
export PrimitiveState, ConservedState, Flux
export conserved_to_primitive, primitive_to_conserved

# SolverConfig
include("utils.jl")
export SolverConfig

# 1D uniform mesh
include("grid.jl")
export UniformGrid1D

# boundary conditions
include("bc.jl")
export TransmissiveBC, BoundaryFace, ghost_state, make_boundary_faces
export apply_bc!

# MUSCL (placeholder for now)
include("reconstruction.jl")
export reconstruct_face_values

# CFL, wave speeds, Forward Euler
include("time_stepping.jl")
export WaveSpeedMethod, Physical
export max_wave_speed, compute_Δt, forward_euler_step!

# dispatch stubs
include("solvers/interface.jl")
export compute_numerical_flux, evolve!

# exact Riemann solver
include("solvers/exact.jl")
export NonlinearWaveStructure
export PressureGuessMethod, PV, TR, TS
export ExactRiemannSolution
export solve_Riemann_problem, sample_solution

# first-order Godunov method
include("solvers/Godunov.jl")
export GodunovSolver
export compute_intercell_fluxes!

end # module Riemann1D
