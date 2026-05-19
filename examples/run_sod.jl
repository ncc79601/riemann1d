using Riemann1D
using Plots
using Printf
gr()

# =============================================================================
# Common problem setup
# =============================================================================
W_L  = PrimitiveState(ρ=1.0, u=0.0, p=1.0)
W_R  = PrimitiveState(ρ=0.125, u=0.0, p=0.1)
eos  = PerfectGasEOS(γ=1.4)
t_end = 0.2

# =============================================================================
# 1. Exact Riemann solution (reference)
# =============================================================================
sol = solve_Riemann_problem_exact(W_L, W_R, eos)

x_range = range(-0.5, 0.5, length=1000)

function sample_field(f, xs, ts, sol)
    data = zeros(length(xs), length(ts))
    for j in axes(ts, 1), i in axes(xs, 1)
        state = sample_exact_solution(xs[i], ts[j], sol)
        data[i, j] = f(state)
    end
    return data
end

# =============================================================================
# 2. Godunov finite-volume simulation
# =============================================================================
N = 100
grid = UniformGrid1D(-0.5, 0.5, N; ghost_cells=1)

# initial condition: Sod shock tube
function init_sod(grid, W_L, W_R, eos)
    U = Vector{ConservedState}(undef, grid.N)
    for i in 1:grid.N
        W = grid.x_centers[i] < 0.0 ? W_L : W_R
        U[i] = primitive_to_conserved(W, eos)
    end
    return U
end

U0 = init_sod(grid, W_L, W_R, eos)
U  = deepcopy(U0)

config = SolverConfig(GodunovSolver(), 0.9, t_end, 10_000)

@info "Running Godunov solver (N = $N, CFL = $(config.cfl))"
runtime = @elapsed t_final, n_steps = evolve!(U, grid, eos, config)
@info "Finished: t_final = $(round(t_final, digits=6)), n_steps = $n_steps, runtime = $(round(runtime, digits=3)) s"

# =============================================================================
# 3. Extract final fields
# =============================================================================
W_final = [conserved_to_primitive(U[i], eos) for i in 1:grid.N]
ρ_fv = [w.ρ for w in W_final]
u_fv = [w.u for w in W_final]
p_fv = [w.p for w in W_final]

# =============================================================================
# 4. Exact solution at t_end on the same grid
# =============================================================================
ρ_exact = [sample_exact_solution(x, t_end, sol).ρ for x in grid.x_centers]
u_exact = [sample_exact_solution(x, t_end, sol).u for x in grid.x_centers]
p_exact = [sample_exact_solution(x, t_end, sol).p for x in grid.x_centers]

# =============================================================================
# 5. Plot: exact (line) vs Godunov (markers)
# =============================================================================
times_to_plot = [t_end]

labels = permutedims(["t = $(t)" for t in times_to_plot])

fields = (
    (:ρ, "Density",  "ρ"),
    (:u, "Velocity", "u"),
    (:p, "Pressure", "p"),
)

plots = Any[]
for (field, title, ylabel) in fields
    data = sample_field(s -> getproperty(s, field), x_range, times_to_plot, sol)
    p = plot(x_range, data,
        title = title, xlabel = "x", ylabel = ylabel,
        label = labels, linewidth = 1.5, legend = :topright)
    scatter!(p, grid.x_centers, getfield.(W_final, field),
        label = "Godunov (N=$N)", markersize = 2, markerstrokewidth = 0)
    push!(plots, p)
end

plt = plot(plots..., layout = (1, 3), size = (1500, 400),
    plot_title = "Sod shock tube — Exact Riemann vs Godunov (N=$N, $(n_steps) steps)",
    titlefontsize = 10)

# =============================================================================
# Output
# =============================================================================
outdir = joinpath(@__DIR__, "..", "outputs")
mkpath(outdir)

outpath_exact = joinpath(outdir, "sod_exact.png")
savefig(plt, outpath_exact)
println("Plot saved to ", outpath_exact)
