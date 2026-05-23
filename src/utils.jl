"""
    RiemannProblem

Initial condition for a Riemann problem: two constant states separated by a discontinuity at `x0`.

# Fields
- `W_L::PrimitiveState`: left state (x < x0)
- `W_R::PrimitiveState`: right state (x > x0)
- `x0::Float64`: position of the discontinuity
"""
struct RiemannProblem
    W_L::PrimitiveState
    W_R::PrimitiveState
    x0::Real
    name::String
end

function RiemannProblem(W_L::PrimitiveState, W_R::PrimitiveState)
    RiemannProblem(W_L, W_R, 0.0, "User-defined Riemann Problem")
end
function RiemannProblem(;
        W_L::PrimitiveState,
        W_R::PrimitiveState,
        x0::Real = 0.0,
        name::String = "User-defined Riemann Problem"
)
    RiemannProblem(W_L, W_R, x0, name)
end

# ---------------------------------------------------------------------------
# standard test cases from RmSv-4.3.3
# ---------------------------------------------------------------------------
"""
    SodProblem()

Popular Sod's shock tube problem. Consists of a left rarefaction, a contact discontinuity, and a right shock.
"""
function SodProblem()
    RiemannProblem(
        W_L = PrimitiveState(ρ = 1.0, u = 0.0, p = 1.0),
        W_R = PrimitiveState(ρ = 0.125, u = 0.0, p = 0.1),
        x0 = 0.0,
        name = "Sod Problem"
    )
end
"""
    ModifiedSodProblem()

Modified version of Sod's shock tube problem, good for testing entropy satisfaction. Similar to [`SodProblem`](@ref), but the left rarefaction is sonic.
"""
function ModifiedSodProblem()
    RiemannProblem(
        W_L = PrimitiveState(ρ = 1.0, u = 0.75, p = 1.0),
        W_R = PrimitiveState(ρ = 0.125, u = 0.0, p = 0.1),
        x0 = 0.0,
        name = "Modified Sod Problem"
    )
end
"""
    OneTwoThreeProblem()

123 Problem. Consists of two symmetric rarefactions, and a zero speed contact discontinuity. Suitable for assessing performance for low-density flows.
"""
function OneTwoThreeProblem()
    RiemannProblem(
        W_L = PrimitiveState(ρ = 1.0, u = -2.0, p = 0.4),
        W_R = PrimitiveState(ρ = 1.0, u = 2.0, p = 0.4),
        x0 = 0.0,
        name = "123 Problem"
    )
end
"""
    WoodwardLeftBlastProblem()

Left half of the blast wave problem of Woodward and Colella. Consists of a super strong shock.
"""
function WoodwardLeftBlastProblem()
    RiemannProblem(
        W_L = PrimitiveState(ρ = 1.0, u = 0.0, p = 1000.0),
        W_R = PrimitiveState(ρ = 1.0, u = 0.0, p = 0.01),
        x0 = 0.0,
        name = "Woodward Left Blast Problem"
    )
end
"""
    WoodwardRightBlastProblem()

Right half of the blast wave problem of Woodward and Colella. Quite symmetry of the [`WoodwardLeftBlastProblem`](@ref).
"""
function WoodwardRightBlastProblem()
    RiemannProblem(
        W_L = PrimitiveState(ρ = 1.0, u = 0.0, p = 0.01),
        W_R = PrimitiveState(ρ = 1.0, u = 0.0, p = 100.0),
        x0 = 0.0,
        name = "Woodward Right Blast Problem"
    )
end
"""
    ShockCollisionProblem()

Made up of the right and left shocks emerging from the solution to [`WoodwardLeftBlastProblem``](@ref) and [`WoodwardRightBlastProblem`](@ref). Represents strong shock collision.
"""
function ShockCollisionProblem()
    RiemannProblem(
        W_L = PrimitiveState(ρ = 5.99924, u = 19.5975, p = 460.894),
        W_R = PrimitiveState(ρ = 5.99242, u = -6.19633, p = 46.0950),
        x0 = 0.0,
        name = "Shock Collision Problem"
    )
end

# ---------------------------------------------------------------------------
# initialise conserved state on a uniform grid from a Riemann problem
# ---------------------------------------------------------------------------
"""
    init_simulation(grid, problem::RiemannProblem, eos) -> Vector{ConservedState}

Allocate, fill and return `U[1:grid.N]` using init field defined by `problem`; cells whose center lies to the left of `problem.x0` get `W_L`, others get `W_R`. Returns pre-allocated array of conserved variables

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

    for i in 1:(grid.N)
        W = xc[i] < problem.x0 ? problem.W_L : problem.W_R
        U[i] = primitive_to_conserved(W, eos)
    end

    return U
end

# ---------------------------------------------------------------------------
# simulation runner
# ---------------------------------------------------------------------------
"""
    run_simulation!(U, grid, eos, config) -> (t_final, n_steps, U, runtime)

Initialise `U` from `problem` on `grid`, run `evolve!`, measure wall time, and return the final conserved state vector.
"""
function run_simulation!(
        U::AbstractArray{ConservedState},
        grid::UniformGrid1D,
        eos::PerfectGasEOS,
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

Convert `Vector{ConservedState}` to primitive fields plus specific internal energy.  Returns a `NamedTuple` suitable for plotting.
"""
function extract_fields(U::AbstractArray{ConservedState}, eos::PerfectGasEOS)
    ρ = Vector{Float64}(undef, length(U))
    u = Vector{Float64}(undef, length(U))
    p = Vector{Float64}(undef, length(U))
    e = Vector{Float64}(undef, length(U))
    for i in axes(U, 1)
        W = conserved_to_primitive(U[i], eos)
        ρ[i] = W.ρ
        u[i] = W.u
        p[i] = W.p
        e[i] = internal_energy(W, eos)
    end
    return (; ρ, u, p, e)
end

"""
    extract_field(U::AbstractArray{ConservedState}, eos, field::Symbol) -> Vector{Float64}

Extract a single scalar field from a conserved-state vector.

Valid fields: `:ρ`, `:u`, `:p`, `:e` (specific internal energy).
"""
function extract_field(U::AbstractArray{ConservedState}, eos::PerfectGasEOS, field::Symbol)
    if field == :e
        return [internal_energy(conserved_to_primitive(U[i], eos), eos) for i in axes(U, 1)]
    else
        W = [conserved_to_primitive(U[i], eos) for i in axes(U, 1)]
        return getproperty.(W, field) # extract field array
    end
end

"""
    extract_fields(sol::ExactRiemannSolution, x, t, eos) -> (; ρ, u, p, e)

Sample the exact Riemann solution at a set of `x` points and return fields as a `NamedTuple`.
"""
function extract_fields(
        sol::ExactRiemannSolution,
        x::AbstractArray{<:Real},
        t::Real,
        eos::PerfectGasEOS
)
    W = [sample_exact_solution(xi, t, sol) for xi in x]
    ρ = getproperty.(W, :ρ)
    u = getproperty.(W, :u)
    p = getproperty.(W, :p)
    e = [internal_energy(W[i], eos) for i in axes(W, 1)]
    return (; ρ, u, p, e)
end

"""
    extract_field(sol::ExactRiemannSolution, x, t, eos, field::Symbol) -> Vector{Float64}

Extract a single field from the exact solution at `(x, t)`.
"""
function extract_field(
        sol::ExactRiemannSolution,
        x::AbstractArray{<:Real},
        t::Real,
        eos::PerfectGasEOS,
        field::Symbol
)
    if field == :e
        W = [sample_exact_solution(xi, t, sol) for xi in x]
        return [internal_energy(W[i], eos) for i in axes(W, 1)]
    else
        W = [sample_exact_solution(xi, t, sol) for xi in x]
        return getproperty.(W, field)
    end
end
