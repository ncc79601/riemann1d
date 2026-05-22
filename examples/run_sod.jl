using Riemann1D
using Plots
plotly()

#%% problem setup
problem   = SodProblem()
eos       = PerfectGasEOS(γ=1.4)

cfl        = 0.4
init_cfl   = 0.1
init_steps = 5

x_min = -0.3
x_max = 0.7
t_end = 0.2

N    = 200 # grid num
grid = UniformGrid1D(x_min, x_max, N; ghost_cells=2)

#%% solver configs
reconstruction = SecondOrderReconstruct()
integrator = TVDRK2()

configs = [
    ("HLLC-minbee",   HLLC(estimate_method=RoeEstimate), MinBeeLimiter()),
    ("HLLC-vanleer",  HLLC(estimate_method=RoeEstimate), vanLeerLimiter()),
    ("HLLC-mc",       HLLC(estimate_method=RoeEstimate), MCLimiter()),
    ("HLLC-superbee", HLLC(estimate_method=RoeEstimate), SuperBeeLimiter()),
    ("HLLC-ultrabee", HLLC(estimate_method=RoeEstimate), UltraBeeLimiter()),
]

#%% run exact solution as ground truth
x_exact_points = range(x_min, x_max, length=1000)
exact_sol = solve_Riemann_problem_exact(problem.W_L, problem.W_R, eos)

#%% run all solvers
results = []

for (name, solver, limiter) in configs
    # prepare solver config
    config = SolverConfig(
        solver, cfl, t_end, 10_000;
        reconstruction, limiter, integrator, init_steps, init_cfl
    )
    
    # initialize
    U = init_simulation(problem, grid, eos)

    # run simulation
    t_final, n_steps, runtime = run_simulation!(U, grid, eos, config)

    # save results
    push!(results, (; name, U))

    # print performance metrics
    @info "$(rpad(name, 18)) steps=$(lpad(n_steps, 5))  runtime=$(round(runtime, digits=4)) s"
end

#%% post processing
field_labels = Dict(
    :ρ => ("Density", "ρ"),
    :u => ("Velocity", "u"),
    :p => ("Pressure", "p"),
    :e => ("Internal energy", "e"),
)

panels = Plots.Plot[]

for field in [:ρ, :u, :p, :e]
    # plot exact solution
    title, ylabel = field_labels[field]
    exact_vals = extract_field(exact_sol, x_exact_points, t_end, eos, field)

    p = plot(
        x_exact_points, exact_vals;
        linewidth = 1, title = title, xlabel = "x", ylabel = ylabel,
        label = "Exact", legend = :topright
    )

    # plot approximate solutions
    for r in results
        vals = extract_field(r.U, eos, field)
        scatter!(
            p, grid.x_centers, vals;
            label = r.name, markersize = 1.5, markerstrokewidth = 0
        )
    end
    push!(panels, p)
end

# subfigure layout
l = @layout [
    a{0.4w,0.45h} b{0.4w} _
    _             _       _
    c{0.45h}      d       _ 
]

plt = plot(
    panels..., layout = l, size = (800, 600),
    plot_title = "Sod shock tube benchmark (N=$N)", titlefontsize = 10
)

gui()
@info "Plot displayed in browser."