using Riemann1D
using Plots
using Printf
# gr()
# use plotly backend for interactive
plotly()

# =============================================================================
# Common problem setup
# =============================================================================
W_L  = PrimitiveState(ρ=1.0, u=0.75, p=1.0) # sod shock tube modified
W_R  = PrimitiveState(ρ=0.125, u=0.0, p=0.1)
eos  = PerfectGasEOS(γ=1.4)
t_end = 0.2
N = 200
cfl = 0.4 # for TVD methods
init_steps = 5
init_cfl = 0.1
x_max, x_min = 0.7, -0.3
grid = UniformGrid1D(x_min, x_max, N; ghost_cells=2) # TODO: ghost cell check

function init_sod(grid, W_L, W_R, eos)
    U = Vector{ConservedState}(undef, grid.N)
    xc = grid.x_centers
    for i in 1:grid.N
        W = xc[i] < 0.0 ? W_L : W_R
        U[i] = primitive_to_conserved(W, eos)
    end
    return U
end

# =============================================================================
# Exact Riemann solution (reference)
# =============================================================================
sol = solve_Riemann_problem_exact(W_L, W_R, eos)
x_range = range(x_min, x_max, length=1000)

# =============================================================================
# Run both solvers
# =============================================================================
configs = [
    # ("Godunov", GodunovSolver(), SecondOrderReconstruct(), vanLeerLimiter(), TVDRK2()),
    # ("PVRS",    PVRS(), SecondOrderReconstruct(), vanLeerLimiter(), TVDRK2()),
    # ("TRRS",    TRRS(), SecondOrderReconstruct(), vanLeerLimiter(), TVDRK2()),
    # ("TSRS",    TSRS(), SecondOrderReconstruct(), vanLeerLimiter(), TVDRK2()),
    # ("AIRS",    AIRS(), SecondOrderReconstruct(), vanLeerLimiter(), TVDRK2()),
    # ("ANRS",    ANRS(), SecondOrderReconstruct(), vanLeerLimiter(), TVDRK2()),
    ("HLLC-no-limiter",    HLLC(estimate_method=RoeEstimate), SecondOrderReconstruct(), NoLimiter(), TVDRK2()),
    ("HLLC-minbee",    HLLC(estimate_method=RoeEstimate), SecondOrderReconstruct(), MinBeeLimiter(), TVDRK2()),
    ("HLLC-vanleer",    HLLC(estimate_method=RoeEstimate), SecondOrderReconstruct(), vanLeerLimiter(), TVDRK2()),
    ("HLLC-mc",    HLLC(estimate_method=RoeEstimate), SecondOrderReconstruct(), MCLimiter(), TVDRK2()),
    ("HLLC-superbee",    HLLC(estimate_method=RoeEstimate), SecondOrderReconstruct(), SuperBeeLimiter(), TVDRK2()),
    ("HLLC-ultrabee",    HLLC(estimate_method=RoeEstimate), SecondOrderReconstruct(), UltraBeeLimiter(), TVDRK2()),
    # ("Roe-NoFix",         RoeSolver(entropy_fix_method=NoFix), SecondOrderReconstruct(), vanLeerLimiter(), TVDRK2()),
    # ("Roe-HartenYee",     RoeSolver(entropy_fix_method=HartenYee, ϵ=0.05), SecondOrderReconstruct(), vanLeerLimiter(), TVDRK2()),
    # ("Roe-HartenHyman",   RoeSolver(entropy_fix_method=HartenHyman), SecondOrderReconstruct(), vanLeerLimiter())
]

results = NamedTuple[]

for (name, solver, reconstruction, limiter, integrator) in configs
    # try
        U = init_sod(grid, W_L, W_R, eos)
        config = SolverConfig(
            solver, cfl, t_end, 10_000;
            reconstruction=reconstruction,
            limiter=limiter,
            integrator=integrator,
            init_steps=init_steps,
            init_cfl=init_cfl
        )

        @info "Running $name (N=$N, CFL=$(config.cfl))"
        runtime = @elapsed t_final, n_steps = evolve!(U, grid, eos, config)
        @info "  $name: t_final=$(round(t_final, digits=6)), steps=$n_steps, time=$(round(runtime, digits=4)) s"

        W_final = [conserved_to_primitive(U[i], eos) for i in 1:grid.N]
        push!(results, (;
            name    = name,
            ρ       = [W.ρ for W in W_final],
            u       = [W.u for W in W_final],
            p       = [W.p for W in W_final],
            e       = [internal_energy(W, eos) for W in W_final], 
            n_steps = n_steps,
            runtime = runtime,
        ))
    # catch e
    #     @error "Error running $name: $e"
    # end
end

# =============================================================================
# Plot: exact line + two solver scatters per panel
# =============================================================================
exact_at = [sample_exact_solution(x, t_end, sol) for x in x_range]

fields = (
    (:ρ, "density",  "ρ"),
    (:u, "velocity", "u"),
    (:p, "pressure", "p"),
    (:e, "Internal energy", "e")
)

panels = Any[]
for (i, (field, title, ylabel)) in enumerate(fields)
    if field == :e
        # calculate internal energy
        exact_vals = [internal_energy(W, eos) for W in exact_at]
    else
        exact_vals = getproperty.(exact_at, field)
    end

    exact_label = "Exact"

    legend_pos = :right

    p = plot(x_range, exact_vals;
        title     = title,
        xlabel    = "x",
        ylabel    = ylabel,
        label     = exact_label,
        linewidth = 1,
        legend    = legend_pos,
    )

    for r in results
        scatter_label = "$(r.name)"

        scatter!(p, grid.x_centers, getfield(r, field);
            label = scatter_label,
            markersize = 1.5,
            markerstrokewidth = 0,
        )
    end

    push!(panels, p)
end

# using Measures

# insert!(panels, 3, plot())

l = @layout [
    a{0.4w,0.45h} b{0.4w} _
    _             _       _
    c{0.45h}      d       _ 
]

plt = plot(
    panels..., layout = l, size = (800, 600),
    plot_title = "Sod shock tube benchmark (N=$N)",
    titlefontsize = 10,
    plot_titlevspan = 0.1, # 20% heigth for title
    # bottom_margin = 10 * Plots.mm
)

gui(plt)

# =============================================================================
# Output
# =============================================================================
# outdir = joinpath(@__DIR__, "..", "outputs")
# mkpath(outdir)
# savefig(plt, joinpath(outdir, "sod_compare.png"))
# println("Plot saved to ", joinpath(outdir, "sod_compare.png"))
