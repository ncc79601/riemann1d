"""
    Riemann1D

Exact and approximate Riemann solvers for the one-dimensional
compressible Euler equations.
"""
module Riemann1D

using ForwardDiff
using StructArrays
using OffsetArrays

include("types.jl")
include("eos.jl")
include("states.jl")

include("grid.jl")
export UniformGrid1D
include("bc.jl")
export TransmissiveBC, BoundaryFace, ghost_state, make_boundary_faces
export apply_bc!

include("reconstruction.jl")
export reconstruct_face_values

include("utils.jl")
include("time_stepping.jl")
export max_wave_speed, compute_cfl_dt, forward_euler_step!

include("solvers/interface.jl")
include("solvers/exact.jl")
include("solvers/Godunov.jl")
export GodunovSolver, compute_numerical_flux, compute_intercell_fluxes!, evolve!

export AbstractState, PrimitiveState, ConservedState, Flux
export AbstractEOS, PerfectGasEOS
export NonlinearWaveStructure
export AbstractRiemannSolution, ExactRiemannSolution
export AbstractRiemannSolver

export PressureGuessMethod, PV, TR, TS
export WaveSpeedMethod, Physical

export SolverConfig

export solve_Riemann_problem
export sample_solution
export conserved_to_primitive, primitive_to_conserved

end # module Riemann1D
