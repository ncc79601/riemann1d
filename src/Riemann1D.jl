module Riemann1D

using ForwardDiff

include("types.jl")
include("exact_Riemann_solver.jl")
export AbstractState, PrimitiveState
export AbstractEOS, PerfectGasEOS
export NonlinearWaveStructure
export AbstractRiemannSolution, ExactRiemannSolution

export solve_Riemann_problem
export sample_solution

end # module Riemann1D
