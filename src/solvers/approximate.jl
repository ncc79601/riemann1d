"""
    PVRS <: AbstractRiemannSolver

Primitive-variable Riemann solver (PVRS). Uses linearization assumption and treats all waves as discontinuities. Not accurate, but fast.
"""
struct PVRS <: AbstractRiemannSolver end


"""
    compute_numerical_flux(solver::PVRS, W_L, W_R, eos)

PVRS flux: solve the Riemann problem and evaluate the physical flux at ``x/t = 0``. Evaluate the flux by comparing ``S_L=u_L-a_L``, ``u_*``, ``S_R=u_R+a_R``, and ``0`` (treat all waves as discontinuities).

```math

```

# Reference
RmSv-9.3
"""
function compute_numerical_flux(
    solver::PVRS,
    W_L::PrimitiveState,
    W_R::PrimitiveState,
    eos::PerfectGasEOS,
)
    ρ_L, u_L, p_L = W_L.ρ, W_L.u, W_L.p
    ρ_R, u_R, p_R = W_R.ρ, W_R.u, W_R.p
    a_L = sound_speed(W_L, eos)
    a_R = sound_speed(W_R, eos)

    ρ̄    = 0.5 * (ρ_L + ρ_R)
    ā    = 0.5 * (a_L + a_R)
    p★   = 0.5 * (p_L + p_R) + 0.5 * (u_L - u_R) * ρ̄ * ā
    u★   = 0.5 * (u_L + u_R) + 0.5 * (p_L - p_R) / (ρ̄ * ā)
    ρ★_L = ρ_L + (u_L - u★) * ρ̄ / ā
    ρ★_R = ρ_R + (u★ - u_R) * ρ̄ / ā

    # sample the solution at x/t = 0
    if 0 <= u_L - a_L # left data state
        return Flux(W_L, eos)
    elseif u_L - a_L < 0 <= u★ # left star-region
        return Flux(PrimitiveState(ρ=ρ★_L, u=u★, p=p★), eos)
    elseif u★ < 0 <= u_R + a_R # right star-region
        return Flux(PrimitiveState(ρ=ρ★_R, u=u★, p=p★), eos)
    else # right data state
        return Flux(W_R, eos)
    end
end


"""
    TRRS <: AbstractRiemannSolver

Two-rarefaction Riemann solver (TRRS).
"""
struct TRRS <: AbstractRiemannSolver end


"""
    compute_numerical_flux(solver::TRRS, W_L, W_R, eos)

TRRS flux: assumes both waves are rarefactions and uses the exact solution formula for the star region. Exact when both waves are rarefactions, but not accurate if shocks are present. Fast.

# Reference:
RmSv-9.4
"""
function compute_numerical_flux(
    solver::TRRS,
    W_L::PrimitiveState,
    W_R::PrimitiveState,
    eos::PerfectGasEOS,
)
    ρ_L, u_L, p_L = W_L.ρ, W_L.u, W_L.p
    ρ_R, u_R, p_R = W_R.ρ, W_R.u, W_R.p
    a_L = sound_speed(W_L, eos)
    a_R = sound_speed(W_R, eos)

    γ = eos.γ
    z = (γ - 1) / (2γ)

    P_LR = (p_L / p_R) ^ z
    u★ = (P_LR * u_L/a_L + u_R/a_R + 2 * (P_LR-1)/(γ-1)) / (P_LR/a_L + 1/a_R)
    p★ = (
        p_L * (1 + (γ-1)/(2a_L) * (u_L - u★)) ^ (1/z) +
        p_R * (1 + (γ-1)/(2a_R) * (u★ - u_R)) ^ (1/z)
    ) / 2
    ρ★_L = ρ_L * (p★ / p_L) ^ (1/γ)
    ρ★_R = ρ_R * (p★ / p_R) ^ (1/γ)

    head_L = u_L - a_L
    tail_L = u★ - sound_speed(PrimitiveState(ρ=ρ★_L, u=u★, p=p★), eos)
    head_R = u_R + a_R
    tail_R = u★ + sound_speed(PrimitiveState(ρ=ρ★_R, u=u★, p=p★), eos)

    # utilize tools from src/solvers/exact.jl to sample the solution at x/t = 0
    wave_structure_L = NonlinearWaveStructure(
        Rarefaction,
        ρ★_L,
        NaN,
        head_L,
        tail_L
    )
    wave_structure_R = NonlinearWaveStructure(
        Rarefaction,
        ρ★_R,
        NaN,
        head_R,
        tail_R
    )
    sol = ExactRiemannSolution(
        eos, W_L, W_R, p★, u★,
        wave_structure_L, wave_structure_R,
    )
    return sample_exact_solution(0.0, 1.0, sol) |> (W -> Flux(W, eos))
end


"""
    TSRS <: AbstractRiemannSolver

Two-shock Riemann solver (TSRS).
"""
struct TSRS <: AbstractRiemannSolver end


"""
    compute_numerical_flux(solver::TSRS, W_L, W_R, eos)

TSRS flux: assumes both waves are shock waves and uses the exact solution formula for the star region. Not exact (even for shock waves because it requires an initial guess for ``p_0``), but more accurate than PVRS if shocks are present. Fast.

# Reference:
RmSv-9.4
"""
function compute_numerical_flux(
    solver::TSRS,
    W_L::PrimitiveState,
    W_R::PrimitiveState,
    eos::PerfectGasEOS,
)
    p₀ = max(0, guess_p★(W_L, W_R, eos, method=PV))
    
    γ = eos.γ
    ρ_L, u_L, p_L = W_L.ρ, W_L.u, W_L.p
    ρ_R, u_R, p_R = W_R.ρ, W_R.u, W_R.p

    A_L = 2 / ((γ+1) * ρ_L)
    A_R = 2 / ((γ+1) * ρ_R)
    B_L = (γ-1)/(γ+1) * p_L
    B_R = (γ-1)/(γ+1) * p_R
    g_L(p) = √(A_L / (p + B_L))
    g_R(p) = √(A_R / (p + B_R))

    p★ = (g_L(p₀)*p_L + g_R(p₀)*p_R - (u_R - u_L)) / (g_L(p₀) + g_R(p₀))
    u★ = 0.5 * (u_L + u_R) + 0.5 * ((p★ - p_R) * g_L(p₀) - (p★ - p_L) * g_R(p₀))
    
    ρ★_L = ρ_L * (p★/p_L + (γ-1)/(γ+1)) / ((γ-1)/(γ+1) * p★/p_L + 1)
    ρ★_R = ρ_R * (p★/p_R + (γ-1)/(γ+1)) / ((γ-1)/(γ+1) * p★/p_R + 1)

    # same as src/solvers/exact.jl
    a_L = sound_speed(W_L, eos)
    a_R = sound_speed(W_R, eos)
    S_L = u_L - a_L * √((γ+1)/(2γ) * (p★/p_L) + (γ-1)/(2γ))
    S_R = u_R + a_R * √((γ+1)/(2γ) * (p★/p_R) + (γ-1)/(2γ))

    # utilize tools from src/solvers/exact.jl to sample the solution at x/t = 0
    wave_structure_L = NonlinearWaveStructure(
        Shock,
        ρ★_L,
        S_L,
        NaN,
        NaN
    )
    wave_structure_R = NonlinearWaveStructure(
        Shock,
        ρ★_R,
        S_R,
        NaN,
        NaN
    )
    sol = ExactRiemannSolution(
        eos, W_L, W_R, p★, u★,
        wave_structure_L, wave_structure_R,
    )

    return sample_exact_solution(0.0, 1.0, sol) |> (W -> Flux(W, eos))
end


"""
    AIRS <: AbstractRiemannSolver

Adaptive iterative Riemann solver (AIRS). Use PVRS if ``Q:=\\frac{p_\\max}{p_\\min} < Q_\\text{user}``, else use exact solver. Default value for ``Q_\\text{user}`` is 2.

# Fields:
 - `Q_user::T`: user-specified threshold for switching between PVRS and exact solver.
"""
struct AIRS{T} <: AbstractRiemannSolver where T <: Real
    Q_user::T
end
function AIRS(; Q_user::Real = 2.0)
    return AIRS(Q_user)
end


"""
    compute_numerical_flux(solver::AIRS, W_L, W_R, eos)

Adaptive iterative Riemann solver (AIRS) flux: compute the pressure ratio ``Q = p_\\max / p_\\min``. Use PVRS if ``Q < Q_\\text{user}``, else use the exact solver.

# Reference:
RmSv-9.5
"""
function compute_numerical_flux(
    solver::AIRS,
    W_L::PrimitiveState,
    W_R::PrimitiveState,
    eos::PerfectGasEOS,
)
    p_L, p_R = W_L.p, W_R.p
    p_max = max(p_L, p_R)
    p_min = min(p_L, p_R)
    Q = p_max / p_min
    if Q < solver.Q_user
        return compute_numerical_flux(PVRS(), W_L, W_R, eos)
    else
        return compute_numerical_flux(GodunovSolver(), W_L, W_R, eos)
    end
end


"""
    ANRS <: AbstractRiemannSolver

Adaptive non-iterative Riemann solver (ANRS). Calculate ``p_*`` using PVRS. If ``Q:=\\frac{p_\\max}{p_\\min} < Q_\\text{user}``, use PVRS. Else, if ``p_* < p_\\min`` use TRRS, otherwise use TSRS. Default value for ``Q_\\text{user}`` is 2.

# Fields:
 - `Q_user::T`: user-specified threshold for switching between PVRS and exact solver.
"""
struct ANRS{T} <: AbstractRiemannSolver where T <: Real
    Q_user::T
end
function ANRS(; Q_user::Real = 2.0)
    return ANRS(Q_user)
end


"""
    compute_numerical_flux(solver::ANRS, W_L, W_R, eos)

Adaptive non-iterative Riemann solver (ANRS) flux: compute the pressure ratio ``Q = p_\\max / p_\\min``. If ``Q < Q_\\text{user}``, use PVRS. Else, compute the PVRS guess for ``p_*``. If ``p_* < p_\\min``, use TRRS, else use TSRS.

# Reference:
RmSv-9.5
"""
function compute_numerical_flux(
    solver::ANRS,
    W_L::PrimitiveState,
    W_R::PrimitiveState,
    eos::PerfectGasEOS,
)
    p_L, p_R = W_L.p, W_R.p
    p_max = max(p_L, p_R)
    p_min = min(p_L, p_R)
    Q = p_max / p_min
    if Q < solver.Q_user
        return compute_numerical_flux(PVRS(), W_L, W_R, eos)
    else
        p★_PVRS = guess_p★(W_L, W_R, eos, method=PV)
        if p★_PVRS < p_min
            return compute_numerical_flux(TRRS(), W_L, W_R, eos)
        else
            return compute_numerical_flux(TSRS(), W_L, W_R, eos)
        end
    end
end