using Riemann1D
using Printf

eos = PerfectGasEOS(γ = 1.4)

# --- Problem bank: add new problems by appending tuples ---
problems = [(
    name = "Sod",
    W_L = PrimitiveState(ρ = 1.0, u = 0.0, p = 1.0),
    W_R = PrimitiveState(ρ = 0.125, u = 0.0, p = 0.1)
),
# (name = "Lax",  W_L = ..., W_R = ...),
]

methods = [PV, TR, TS]

println("Problem │ Method │   p★        │   u★        │ Iterations")
println("────────┼────────┼─────────────┼─────────────┼───────────")

for prob in problems
    for method in methods
        p0 = Riemann1D.guess_p★(prob.W_L, prob.W_R, eos, method = method)
        p★, n_iter = Riemann1D.solve_p★_Newton_loop(prob.W_L, prob.W_R, eos, p0)
        sol = solve_Riemann_problem_exact(prob.W_L, prob.W_R, eos, init_guess_method = method)
        @printf("%-7s │ %-6s │ %11.8f │ %11.8f │ %2d\n",
            prob.name,
            method,
            sol.p★,
            sol.u★,
            n_iter)
    end
    println("────────┼────────┼─────────────┼─────────────┼───────────")
end
