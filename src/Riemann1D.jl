"""
    Riemann1D

Exact and approximate Riemann solvers for the one-dimensional
compressible Euler equations.
"""
module Riemann1D

# dependencies
using ForwardDiff
using StructArrays
using OffsetArrays

# abstract type hierarchy
include("types.jl")
export AbstractState, AbstractEOS, AbstractRiemannSolution, AbstractRiemannSolver, AbstractGrid, AbstractBoundaryCondition, AbstractReconstructMethod, AbstractLimiter, AbstractIntegrator

# equations of state
include("eos.jl")
export PerfectGasEOS

# primitive / conserved variables, flux structs and conversions
include("states.jl")
export PrimitiveState, ConservedState, Flux
export conserved_to_primitive, primitive_to_conserved, sound_speed, total_enthalpy, internal_energy

# SolverConfig
include("config.jl")
export SolverConfig

# 1D uniform mesh
include("grid.jl")
export UniformGrid1D

# boundary conditions
include("bc.jl")

# MUSCL (placeholder for now)
include("reconstruction.jl")
export NoReconstruct, SecondOrderReconstruct

# limiters
include("limiters.jl")
export NoLimiter, SwebyLimiter, MinBeeLimiter, SuperBeeLimiter, UltraBeeLimiter, vanLeerLimiter, MCLimiter

# dispatch stubs
include("solvers/interface.jl")
export compute_numerical_flux

# CFL, wave speeds, Forward Euler, intercell flux loop, main time loop
include("evolve.jl")
export WaveSpeedMethod, Physical
export ExplicitEuler, TVDRK2
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

# approximate Riemann solvers: PVRS, TRRS, TSRS, AIRS, ANRS
include("solvers/approximate.jl")
export PVRS, TRRS, TSRS, AIRS, ANRS

# Roe scheme
include("solvers/Roe.jl")
export RoeEntropyFix, NoFix, HartenHyman, HartenYee
export RoeEstimate
export RoeSolver

# HLLC scheme
include("solvers/HLLC.jl")
export HLLCEstimateMethod
export HLLC

# helper functions
include("utils.jl")
export RiemannProblem, SodProblem, LaxProblem
export init_simulation, run_simulation!
export extract_fields, extract_field

end # module Riemann1D
