using Riemann1D
using Plots
using Printf
# gr()
plotly()

# =============================================================================
# Common problem setup
# =============================================================================
W_L  = PrimitiveState(ρ=1.0, u=0.75, p=1.0) # sod shock tube modified
W_R  = PrimitiveState(ρ=0.125, u=0.0, p=0.1)
eos  = PerfectGasEOS(γ=1.4)
t_end = 0.2
N = 100
cfl = 0.4
init_steps = 5
init_cfl = 0.2
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
    ("Godunov", GodunovSolver(), SecondOrderReconstruct(), vanLeerLimiter()),
    # ("PVRS",    PVRS(), SecondOrderReconstruct(), vanLeerLimiter()),
    # ("TRRS",    TRRS(), SecondOrderReconstruct(), vanLeerLimiter()),
    # ("TSRS",    TSRS(), SecondOrderReconstruct(), vanLeerLimiter()),
    # ("AIRS",    AIRS(), SecondOrderReconstruct(), vanLeerLimiter()),
    # ("ANRS",    ANRS(), SecondOrderReconstruct(), vanLeerLimiter()),
    ("HLLC",    HLLC(estimate_method=RoeEstimate), SecondOrderReconstruct(), vanLeerLimiter()),
    # ("Roe-NoFix",         RoeSolver(entropy_fix_method=NoFix), SecondOrderReconstruct(), vanLeerLimiter()),
    # ("Roe-HartenYee",     RoeSolver(entropy_fix_method=HartenYee, ϵ=0.05), SecondOrderReconstruct(), vanLeerLimiter()),
    ("Roe-HartenHyman",   RoeSolver(entropy_fix_method=HartenHyman), SecondOrderReconstruct(), vanLeerLimiter())
]

results = NamedTuple[]
for (name, solver, reconstruction, limiter) in configs
    # try
        U = init_sod(grid, W_L, W_R, eos)
        config = SolverConfig(
            solver, cfl, t_end, 10_000;
            reconstruction=reconstruction,
            limiter=limiter,
            init_steps=init_steps,
            init_cfl=init_cfl
        )

        @info "Running $name (N=$N, CFL=$(config.cfl))"
        runtime = @elapsed t_final, n_steps = evolve!(U, grid, eos, config)
        @info "  $name: t_final=$(round(t_final, digits=6)), steps=$n_steps, time=$(round(runtime, digits=4)) s"

        W_final = [conserved_to_primitive(U[i], eos) for i in 1:grid.N]
        push!(results, (;
            name    = name,
            ρ       = [w.ρ for w in W_final],
            u       = [w.u for w in W_final],
            p       = [w.p for w in W_final],
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
    (:ρ, "Density",  "ρ"),
    (:u, "Velocity", "u"),
    (:p, "Pressure", "p"),
)

panels = Any[]
for (field, title, ylabel) in fields
    exact_vals = getproperty.(exact_at, field)

    p = plot(x_range, exact_vals;
        title     = title,
        xlabel    = "x",
        ylabel    = ylabel,
        label     = "Exact",
        linewidth = 1.5,
        legend    = :topright,
    )

    for r in results
        scatter!(p, grid.x_centers, getfield(r, field);
            label = "$(r.name) ($(r.n_steps) steps)",
            markersize = 2,
            markerstrokewidth = 0,
        )
    end

    push!(panels, p)
end

plt = plot(panels..., layout = (1, 3), size = (1500, 400),
    plot_title = "Sod shock tube benchmark (N=$N)",
    titlefontsize = 10)

gui(plt)

# =============================================================================
# Output
# =============================================================================
# outdir = joinpath(@__DIR__, "..", "outputs")
# mkpath(outdir)
# savefig(plt, joinpath(outdir, "sod_compare.png"))
# println("Plot saved to ", joinpath(outdir, "sod_compare.png"))
