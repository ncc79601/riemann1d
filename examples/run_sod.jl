import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, ".."))

using Riemann1D
using Plots
gr()

# --- Problem setup ---
W_L = PrimitiveState(ρ=1.0, u=0.0, p=1.0)
W_R = PrimitiveState(ρ=0.125, u=0.0, p=0.1)
eos = PerfectGasEOS(γ=1.4)
sol = solve_Riemann_problem(W_L, W_R, eos)

# --- Sampling parameters ---
x_range = range(-0.5, 0.5, length=1000)
t_values = [0.02, 0.05, 0.10, 0.15, 0.20]

function sample_field(f, xs, ts, sol)
    data = zeros(length(xs), length(ts))
    for j in axes(ts, 1), i in axes(xs, 1)
        state = sample_solution(xs[i], ts[j], sol)
        data[i, j] = f(state)
    end
    return data
end

# --- Plotting: one loop over (field_symbol, title, ylabel) ---
labels = permutedims(["t = $(t)" for t in t_values])

fields = (
    (:ρ, "Density",  "ρ"),
    (:u, "Velocity", "u"),
    (:p, "Pressure", "p"),
)

plots = Any[]
for (field, title, ylabel) in fields
    data = sample_field(s -> getproperty(s, field), x_range, t_values, sol)
    push!(plots, plot(x_range, data,
        title = title, xlabel = "x", ylabel = ylabel,
        label = labels, linewidth = 1.5, legend = :topright))
end

plt = plot(plots..., layout = (1, 3), size = (1500, 400),
    plot_title = "Sod shock tube — exact Riemann solution",
    titlefontsize = 10)

# --- Output ---
outdir = joinpath(@__DIR__, "..", "outputs")
mkpath(outdir)
outpath = joinpath(outdir, "sod_exact.png")
savefig(plt, outpath)
println("Plot saved to ", outpath)