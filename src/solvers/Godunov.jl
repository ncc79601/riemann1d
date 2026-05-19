"""
    GodunovSolver <: AbstractRiemannSolver

First-order Godunov scheme.  Uses the exact Riemann solver to compute
numerical fluxes at each cell interface.
"""
struct GodunovSolver <: AbstractRiemannSolver end


"""
    compute_numerical_flux(solver::GodunovSolver, W_L, W_R, eos)

Godunov flux: solve the exact Riemann problem and evaluate the physical flux
at ``x/t = 0``.
"""
function compute_numerical_flux(
    solver::GodunovSolver,
    W_L::PrimitiveState,
    W_R::PrimitiveState,
    eos::PerfectGasEOS,
)
    W₀ = sample_exact_solution(0.0, 1.0, solve_Riemann_problem_exact(W_L, W_R, eos))
    return Flux(W₀, eos)
end
