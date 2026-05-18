"""
    Riemann1D

Exact and approximate Riemann solvers for the one-dimensional
compressible Euler equations.
"""
module Riemann1D

using ForwardDiff

include("types.jl")
include("states.jl")
include("eos.jl")
include("utils.jl")

include("solvers/exact.jl")
export AbstractState, PrimitiveState
export AbstractEOS, PerfectGasEOS
export NonlinearWaveStructure
export AbstractRiemannSolution, ExactRiemannSolution

export PressureGuessMethod, PV, TR, TS

export solve_Riemann_problem
export sample_solution

end # module Riemann1D
