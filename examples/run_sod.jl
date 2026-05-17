import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, ".."))

using Riemann1D

# Riemann problem of Sod shock tube
W_L = PrimitiveState(ρ=1.0, u=0.0, p=1.0)
W_R = PrimitiveState(ρ=0.125, u=0.0, p=0.1)
eos = PerfectGasEOS(γ=1.4)
sol = solve_Riemann_problem(W_L, W_R, eos)

# create 2d grid
x_range = range(-0.5, 0.5, length=500)   # x grid
t_range = range(0.02, 0.2, length=500)   # t grid
X = repeat(x_range, 1, length(t_range))
T = repeat(t_range', length(x_range), 1)

# sample solution
ρ_arr = zeros(size(X))
u_arr = zeros(size(X))
p_arr = zeros(size(X))

for i in axes(x_range, 1), j in axes(t_range, 1)
    state = sample_solution(x_range[i], t_range[j], sol)
    ρ_arr[i, j] = state.ρ
    u_arr[i, j] = state.u
    p_arr[i, j] = state.p
end

# plot
using Plots
plotlyjs()
p1 = surface(X, T, ρ_arr,
    xlabel = "x", ylabel = "t", zlabel = "ρ",
    title  = "rho",
    camera = (30, 45),
    c = :RdBu)
p2 = surface(X, T, u_arr,
    xlabel = "x", ylabel = "t", zlabel = "u",
    title  = "u",
    camera = (30, 45),
    c = :RdBu)
p3 = surface(X, T, p_arr,
    xlabel = "x", ylabel = "t", zlabel = "p",
    title  = "p",
    camera = (30, 45),
    c = :RdBu)
plot(p1, p2, p3, layout = (1, 3), size = (1200, 400))

# save to HTML
savefig("sod_3d.html")
println("solution plot saved to sod_3d.html")