"""
    RiemannProblem

Initial condition for a Riemann problem: two constant states separated by a discontinuity at `x₀`.

# Fields
- `W_L::PrimitiveState`: left state (x < x₀)
- `W_R::PrimitiveState`: right state (x > x₀)
- `x₀::Float64`: position of the discontinuity
"""
struct RiemannProblem
    W_L::PrimitiveState
    W_R::PrimitiveState
    x₀::Real
end

RiemannProblem(W_L::PrimitiveState, W_R::PrimitiveState) = RiemannProblem(W_L, W_R, 0.0)
RiemannProblem(; W_L::PrimitiveState, W_R::PrimitiveState, x₀::Real=0.0) = RiemannProblem(W_L, W_R, x₀)

# ---------------------------------------------------------------------------
# standard test cases
# ---------------------------------------------------------------------------
SodProblem() = RiemannProblem(
    PrimitiveState(ρ=1.0,   u=0.0,    p=1.0),
    PrimitiveState(ρ=0.125, u=0.0,    p=0.1),
    0.0
)
LaxProblem() = RiemannProblem(
    PrimitiveState(ρ=0.445, u=0.698,  p=3.528),
    PrimitiveState(ρ=0.5,   u=0.0,    p=0.571),
    0.0
)

# ---------------------------------------------------------------------------
# initialise conserved state on a uniform grid from a Riemann problem
# ---------------------------------------------------------------------------
"""
    init_simulation(grid, problem::RiemannProblem, eos) -> Vector{ConservedState}

Allocate, fill and return `U[1:grid.N]` using init field defined by `problem`; cells whose center lies to the left of `problem.x₀` get `W_L`, others get `W_R`. Returns pre-allocated array of conserved variables

# Arguments:
- `problem::RiemannProblem`: Riemann problem definition
- `grid::UniformGird1D`: description for uniform 1d grid
- `eos::PerfectGasEOS`: equation of state

# Returns:
- `U`: pre-allocated conserved variable work array of size `grid.N`
"""
function init_simulation(problem::RiemannProblem, grid::UniformGrid1D, eos::PerfectGasEOS)
    U = Vector{ConservedState}(undef, grid.N)
    xc = grid.x_centers # cell center coords

    for i in 1:grid.N
        W = xc[i] < problem.x₀ ? problem.W_L : problem.W_R
        U[i] = primitive_to_conserved(W, eos)
    end

    return U
end

# ---------------------------------------------------------------------------
# simulation runner
# ---------------------------------------------------------------------------
"""
    run_simulation!(U, grid, eos, config) -> (t_final, n_steps, U, runtime)

Initialise `U` from `problem` on `grid`, run `evolve!`, measure wall time,
and return the final conserved state vector.
"""
function run_simulation!(
    U     ::AbstractArray{ConservedState},
    grid  ::UniformGrid1D,
    eos   ::PerfectGasEOS,
    config::SolverConfig
)
    runtime = @elapsed t_final, n_steps = evolve!(U, grid, eos, config)
    return t_final, n_steps, runtime
end

# ---------------------------------------------------------------------------
# post-processing
# ---------------------------------------------------------------------------
"""
    extract_fields(U, eos) -> (; ρ, u, p, e)

Convert `Vector{ConservedState}` to primitive fields plus specific internal
energy.  Returns a `NamedTuple` suitable for plotting.
"""
function extract_fields(U, eos)
    ρ = Vector{Float64}(undef, length(U))
    u = Vector{Float64}(undef, length(U))
    p = Vector{Float64}(undef, length(U))
    e = Vector{Float64}(undef, length(U))
    for i in axes(U, 1)
        W   = conserved_to_primitive(U[i], eos)
        ρ[i] = W.ρ
        u[i] = W.u
        p[i] = W.p
        e[i] = internal_energy(W, eos)
    end
    return (; ρ, u, p, e)
end

"""
    extract_field(U, eos, field::Symbol) -> Vector{Float64}

Extract a single scalar field from a conserved-state vector.

Valid fields: `:ρ`, `:u`, `:p`, `:e` (specific internal energy).
"""
function extract_field(U, eos, field::Symbol)
    if field == :e
        return [internal_energy(conserved_to_primitive(U[i], eos), eos) for i in axes(U, 1)]
    else
        W = [conserved_to_primitive(U[i], eos) for i in axes(U, 1)]
        return getproperty.(W, field)
    end
end

"""
    exact_fields(sol::ExactRiemannSolution, x, t, eos) -> (; ρ, u, p, e)

Sample the exact Riemann solution at a set of `x` points and return fields as
a `NamedTuple`.
"""
function exact_fields(sol::ExactRiemannSolution, x, t, eos)
    W = [sample_exact_solution(xi, t, sol) for xi in x]
    ρ = getproperty.(W, :ρ)
    u = getproperty.(W, :u)
    p = getproperty.(W, :p)
    e = [internal_energy(W[i], eos) for i in axes(W, 1)]
    return (; ρ, u, p, e)
end

"""
    exact_field(sol::ExactRiemannSolution, x, t, eos, field::Symbol) -> Vector{Float64}

Extract a single field from the exact solution at `(x, t)`.
"""
function exact_field(sol::ExactRiemannSolution, x, t, eos, field::Symbol)
    if field == :e
        W = [sample_exact_solution(xi, t, sol) for xi in x]
        return [internal_energy(W[i], eos) for i in axes(W, 1)]
    else
        W = [sample_exact_solution(xi, t, sol) for xi in x]
        return getproperty.(W, field)
    end
end

# ---------------------------------------------------------------------------
# solution metrics
# ---------------------------------------------------------------------------
"""
    solution_metrics(results)

Print a table of metrics (n_steps, runtime) for each result in `results`.
Each element should be a `NamedTuple` with `:name`, `:n_steps`, `:runtime`.
"""
function solution_metrics(results)
    for r in results
        name    = r isa NamedTuple ? r.name    : r[1]
        steps   = r isa NamedTuple ? r.n_steps : r[3]
        runtime = r isa NamedTuple ? r.runtime : r[4]
        @info "$(rpad(name, 18)) steps=$(lpad(steps, 5))  runtime=$(round(runtime, digits=4)) s"
    end
end
