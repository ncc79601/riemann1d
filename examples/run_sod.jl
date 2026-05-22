using Riemann1D
using Plots
plotly()

# ---------------------------------------------------------------------------
# problem & grid
# ---------------------------------------------------------------------------
problem  = SodProblem()
eos      = PerfectGasEOS(γ=1.4)
t_end    = 0.2

N         = 200
cfl       = 0.4
init_cfl  = 0.1
init_steps = 5

x_min, x_max = -0.3, 0.7
grid = UniformGrid1D(x_min, x_max, N; ghost_cells=2)

# ---------------------------------------------------------------------------
# exact solution
# ---------------------------------------------------------------------------
sol     = solve_Riemann_problem_exact(problem.W_L, problem.W_R, eos)
x_range = range(x_min, x_max, length=1000)

# ---------------------------------------------------------------------------
# solver configurations
# ---------------------------------------------------------------------------
base = (; reconstruction = SecondOrderReconstruct(), integrator = TVDRK2(),
          init_steps, init_cfl)

configs = [
    ("HLLC MinBee",   HLLC(; estimate_method=RoeEstimate), MinBeeLimiter()),
    ("HLLC vanLeer",  HLLC(; estimate_method=RoeEstimate), vanLeerLimiter()),
    ("HLLC MC",       HLLC(; estimate_method=RoeEstimate), MCLimiter()),
    ("HLLC SuperBee", HLLC(; estimate_method=RoeEstimate), SuperBeeLimiter()),
    ("HLLC UltraBee", HLLC(; estimate_method=RoeEstimate), UltraBeeLimiter()),
]

# ---------------------------------------------------------------------------
# run all
# ---------------------------------------------------------------------------
results = []
for (name, solver, limiter) in configs
    # prepare solver config
    config = SolverConfig(solver, cfl, t_end, 10_000; base..., limiter)
    
    # initialize
    U = init_simulation(problem, grid, eos)

    # run simulation
    t_final, n_steps, runtime = run_simulation!(U, grid, eos, config)

    # save results
    push!(results, (; name, U, n_steps, runtime))
end
solution_metrics(results)

# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------
field_labels = Dict(
    :ρ => ("Density", "ρ"),
    :u => ("Velocity", "u"),
    :p => ("Pressure", "p"),
    :e => ("Internal energy", "e"),
)

panels = Plots.Plot[]
for field in [:ρ, :u, :p, :e]
    title, ylabel = field_labels[field]
    exact_vals = exact_field(sol, x_range, t_end, eos, field)

    p = plot(x_range, exact_vals;
        title = title, xlabel = "x", ylabel = ylabel,
        label = "Exact", linewidth = 1, legend = :topright)

    for r in results
        vals = extract_field(r.U, eos, field)
        scatter!(p, grid.x_centers, vals;
            label = r.name, markersize = 1.5, markerstrokewidth = 0)
    end
    push!(panels, p)
end

l = @layout [
    a{0.4w,0.45h} b{0.4w} _
    _             _       _
    c{0.45h}      d       _ 
]

plt = plot(panels..., layout = l, size = (800, 600),
    plot_title = "Sod shock tube benchmark (N=$N)", titlefontsize = 10)

gui()
@info "Plot displayed in browser."