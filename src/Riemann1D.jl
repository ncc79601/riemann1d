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
export conserved_to_primitive, primitive_to_conserved, sound_speed

# SolverConfig
include("utils.jl")
export SolverConfig

# 1D uniform mesh
include("grid.jl")
export UniformGrid1D

# boundary conditions
include("bc.jl")

# MUSCL (placeholder for now)
include("reconstruction.jl")
export reconstruct_face_values

# dispatch stubs
include("solvers/interface.jl")
export compute_numerical_flux

# CFL, wave speeds, Forward Euler, intercell flux loop, main time loop
include("evolve.jl")
export WaveSpeedMethod, Physical
export evolve!

# exact Riemann solver
include("solvers/exact.jl")
export NonlinearWaveStructure
export PressureGuessMethod, PV, TR, TS
export ExactRiemannSolution
export solve_Riemann_problem_exact, sample_exact_solution

# first-order Godunov method
include("solvers/Godunov.jl")
export GodunovSolver
# approximate Riemann solvers: PVRS
include("solvers/approximate.jl")
export PVRS

end # module Riemann1D
