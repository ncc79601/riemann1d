"""
    NonlinearWaveType

Enumeration of nonlinear wave character in the Riemann solution.

- `Shock`
- `Rarefaction`
"""
@enum NonlinearWaveType Shock Rarefaction


"""
    NonlinearWaveStructure{T<:Real}

Describes a single nonlinear wave in the Riemann solution.

# Fields
- `wave_type::NonlinearWaveType`: `Shock` or `Rarefaction`
- `ρ★::T`: star-region density
- `S::T`: shock speed (`NaN` for rarefaction)
- `head::T`: head velocity of rarefaction fan (`NaN` for shock)
- `tail::T`: tail velocity of rarefaction fan (`NaN` for shock)
"""
struct NonlinearWaveStructure{T <: Real}
    # type of nonlinear wave
    wave_type::NonlinearWaveType # Shock or Rarefaction
    ρ★  ::T

    # wave velocities
    S   ::T # shock velocities (NaN if rarefaction)
    head::T # rarefaction head velocity
    tail::T # rarefaction tail velocity
end


"""
    ExactRiemannSolution{T<:Real} <: AbstractRiemannSolution

Complete exact solution of the one-dimensional Riemann problem
for the compressible Euler equations with a perfect gas EOS.

# Fields
- `eos::PerfectGasEOS`: equation of state
- `W_L::PrimitiveState{T}`: initial left state
- `W_R::PrimitiveState{T}`: initial right state
- `p★::T`: star-region pressure
- `u★::T`: star-region velocity
- `left_wave::NonlinearWaveStructure{T}`: left nonlinear wave
- `right_wave::NonlinearWaveStructure{T}`: right nonlinear wave
"""
struct ExactRiemannSolution{T <: Real} <: AbstractRiemannSolution
    eos::PerfectGasEOS
    W_L::PrimitiveState{T}
    W_R::PrimitiveState{T}
    p★::T
    u★::T
    left_wave::NonlinearWaveStructure{T}
    right_wave::NonlinearWaveStructure{T}
end


"""
    pressure_function(p::Real, W_K::PrimitiveState, eos::PerfectGasEOS)

Single-sided pressure function ``f_K(p)`` for state `K ∈ {L, R}`

# Arguments
- `p::Real`: guessed star-region pressure
- `W_K::PrimitiveState`: initial state on side `K`
- `eos::PerfectGasEOS`: equation of state

# Returns
- `f_K(p)`: the velocity jump across the wave

# References
- RmSv-4.2
"""
function pressure_function(p::Real, W_K::PrimitiveState, eos::PerfectGasEOS)
    # f_L and f_R
    # [reference] RmSv-4.2
    # extract primitive variables
    ρ_K, u_K, p_K = W_K.ρ, W_K.u, W_K.p
    γ = eos.γ
    A_K = 2 / (ρ_K * (γ + 1))
    B_K = (γ - 1) / (γ + 1) * p_K

    if p > p_K # shock, from Rankine-Hugoniot condition
        return (p - p_K) * sqrt(A_K / (p + B_K))
    else # rarefaction, from generalized Riemann invariants
        a_K = √(γ * p_K / ρ_K)
        return 2a_K / (γ - 1) * ((p / p_K) ^ ((γ - 1) / (2γ)) - 1)
    end
end


"""
    pressure_function(p::Real, W_L::PrimitiveState, W_R::PrimitiveState, eos::PerfectGasEOS)

Full pressure function ``f(p) = f_L(p) + f_R(p) + u_R - u_L``. The star-region pressure `p★` is the root of this function. Used by the Newton–Raphson solver in [`solve_p★_Newton_loop`](@ref).

# Arguments
- `p::Real`: guessed star-region pressure
- `W_L::PrimitiveState`: initial left state
- `W_R::PrimitiveState`: initial right state
- `eos::PerfectGasEOS`: equation of state

# Returns
- `f(p)`: pressure function value
"""
function pressure_function(p::Real, W_L::PrimitiveState, W_R::PrimitiveState, eos::PerfectGasEOS)
    u_L, u_R = W_L.u, W_R.u
    return pressure_function(p, W_L, eos) + pressure_function(p, W_R, eos) + (u_R - u_L)
end


"""
    PressureGuessMethod

Enumeration of initial pressure guess methods:
- `TR`: two rarefactions
- `TS`: two shocks
- `PV`: primitive variable linearisation
"""
@enum PressureGuessMethod TR TS PV


"""
    guess_p★(W_L, W_R, eos; method = TS)

Compute an initial guess for the star-region pressure `p★`.

# Arguments
- `W_L::PrimitiveState`: initial left state
- `W_R::PrimitiveState`: initial right state
- `eos::PerfectGasEOS`: equation of state
- `method::PressureGuessMethod = TS`: guess method (`PV`, `TR`, or `TS`)

# Returns
- Initial estimate of `p★`
"""
function guess_p★(W_L::PrimitiveState, W_R::PrimitiveState, eos::PerfectGasEOS; method::PressureGuessMethod = TS)
    # extract primitive variables
    ρ_L, u_L, p_L = W_L.ρ, W_L.u, W_L.p
    ρ_R, u_R, p_R = W_R.ρ, W_R.u, W_R.p
    γ = eos.γ
    
    if method == TS # two-shock guess
        A_L = 2 / (ρ_L * (γ + 1))
        B_L = (γ - 1) / (γ + 1) * p_L
        A_R = 2 / (ρ_R * (γ + 1))
        B_R = (γ - 1) / (γ + 1) * p_R
        g_L(p) = √(A_L / (p + B_L))
        g_R(p) = √(A_R / (p + B_R))

        p₀ = guess_p★(W_L, W_R, eos, method=PV) # PV guess as initial guess
        return (g_L(p₀) * p_L + g_R(p₀) * p_R - (u_R - u_L)) / (g_L(p₀) + g_R(p₀))
    
    elseif method == TR # two-rarefaction guess
        z = (γ - 1) / (2γ)
        a_L = √(γ * p_L / ρ_L)
        a_R = √(γ * p_R / ρ_R)

        return ((a_L + a_R - (γ-1)/2 * (u_R - u_L)) / (a_L / p_L^z + a_R / p_R^z)) ^ (1/z)
    
    else # primitive-variable guess
        a_L = √(γ * p_L / ρ_L)
        a_R = √(γ * p_R / ρ_R)

        return (p_L + p_R) / 2 + ((u_L - u_R) * (ρ_L + ρ_R) * (a_L + a_R)) / 8
    end
end


"""
    solve_p★_Newton_loop(W_L, W_R, eos, p0; max_iter=50, tol=1e-10)

Low-level Newton–Raphson iteration for the star-region pressure solution. Separated from [`solve_p★`](@ref) so that iteration counts can be inspected (for benchmarking initial guess methods). Uses `ForwardDiff.derivative` for automatic differentiation of [`pressure_function`](@ref).

# Arguments
- `W_L::PrimitiveState`: initial left state
- `W_R::PrimitiveState`: initial right state
- `eos::PerfectGasEOS`: equation of state
- `p0`: initial guess for `p★` (obtained using [`guess_p★`](@ref))
- `max_iter::Int = 50`: maximum iteration number
- `tol::Real = 1e-10`: convergence tolerance on ``|f(p)|``

# Returns
- `(p★, n_iter)`: converged star-region pressure and number of iterations taken
"""
function solve_p★_Newton_loop(W_L::PrimitiveState, W_R::PrimitiveState, eos::PerfectGasEOS, p0;
                              max_iter=50, tol=1e-10)
    f(p) = pressure_function(p, W_L, W_R, eos) # currying the pressure function
    p★ = p0 # initial guess

    for i in 1:max_iter
        residual = f(p★)
        if isnan(residual)
            error("pressure function returned NaN at p★ = $p★")
        end
        if abs(residual) < tol
            return (p★, i)
        end

        deriv = ForwardDiff.derivative(f, p★) # automatic differentiation
        if abs(deriv) < tol
            @warn "derivative close to zero at iteration $i, stopping"
            return (p★, i)
        end

        Δp = -residual / deriv
        p★ = max(p★ + Δp, 1e-14)
    end

    @warn "Newton-Raphson did not converge after $max_iter iterations"
    return (p★, max_iter)
end


"""
    solve_p★(W_L, W_R, eos; init_guess_method=TS, max_iter=50, tol=1e-10)

Solve for the star-region pressure ``p_*`` in the Riemann problem. Use [`guess_p★`](@ref) to get the initial guess of `p★` and then call [`solve_p★_Newton_loop`](@ref).

# Arguments
- `W_L::PrimitiveState`: initial left state
- `W_R::PrimitiveState`: initial right state
- `eos::PerfectGasEOS`: equation of state
- `init_guess_method::PressureGuessMethod = TS`: initial guess method
- `max_iter::Int = 50`: maximum iteration number
- `tol::Real = 1e-10`: convergence tolerance

# Returns
- `p★`: the star-region pressure
"""
function solve_p★(W_L::PrimitiveState, W_R::PrimitiveState, eos::PerfectGasEOS;
                  init_guess_method::PressureGuessMethod=TS, max_iter=50, tol=1e-10)
    p0 = guess_p★(W_L, W_R, eos, method=init_guess_method)
    p★, _ = solve_p★_Newton_loop(W_L, W_R, eos, p0; max_iter=max_iter, tol=tol)
    return p★
end


"""
    calc_u★_from_p★(W_L, W_R, eos, p★)

Compute ``u_*`` from ``p_*`` using the relation ``u_* = \\frac{1}{2}(u_L + u_R) + \\frac{1}{2}(f_R(p_*) - f_L(p_*))``.

# Arguments
- `W_L::PrimitiveState`: initial left state
- `W_R::PrimitiveState`: initial right state
- `eos::PerfectGasEOS`: equation of state
- `p★::Real`: star-region pressure (from [`solve_p★`](@ref))

# Returns
- `u★`: star-region velocity

# References
RmSv-4.2
"""
function calc_u★_from_p★(
    W_L::PrimitiveState,
    W_R::PrimitiveState,
    eos::PerfectGasEOS,
    p★ ::Real
)
    u_L, u_R = W_L.u, W_R.u
    f_L = pressure_function(p★, W_L, eos)
    f_R = pressure_function(p★, W_R, eos)
    return 0.5 * (u_L + u_R) + 0.5 * (f_R - f_L)
end


"""
    calc_wave_structure_from_p★_and_u★(W_L, W_R, eos, p★, u★)

Determine the full wave structure given the star-region values.

For each side (left / right), decides whether the wave is a shock or a
rarefaction, and computes:
- For a **shock**: shock speed ``S`` and post-shock density ``\\rho_*``.
- For a **rarefaction**: head and tail velocities of the expansion fan,
  and post-rarefaction density ``\\rho_*``.

# Arguments
- `W_L::PrimitiveState`: initial left state
- `W_R::PrimitiveState`: initial right state
- `eos::PerfectGasEOS`: equation of state
- `p★::Real`: star-region pressure
- `u★::Real`: star-region velocity

# Returns
- `(left_wave, right_wave)`: a pair of [`NonlinearWaveStructure`](@ref) objects

# References
RmSv-3.1, RmSv-4.4
"""
function calc_wave_structure_from_p★_and_u★(
    W_L::PrimitiveState,
    W_R::PrimitiveState,
    eos::PerfectGasEOS,
    p★ ::Real,
    u★ ::Real
)
    # extract primitive variables
    ρ_L, u_L, p_L = W_L.ρ, W_L.u, W_L.p
    ρ_R, u_R, p_R = W_R.ρ, W_R.u, W_R.p
    γ = eos.γ

    # speed of sound
    a_L = √(γ * p_L / ρ_L)
    a_R = √(γ * p_R / ρ_R)

    # derive left and right wave structures
    # [reference] RmSv-3.1
    # left wave
    if p★ > p_L # left shock
        wave_type_L = Shock
        # shock velocity:
        S_L = u_L - a_L * √((γ+1)/(2γ) * (p★ / p_L) + (γ-1)/(2γ))
        ρ★_L = ρ_L * (p★/p_L + (γ-1)/(γ+1)) / 
               (((γ-1)/(γ+1)) * (p★/p_L) + 1)
        head_L = tail_L = NaN
    else # left rarefaction
        wave_type_L = Rarefaction
        # compute a★_L using generalized Riemann invariants
        a★_L = a_L + (γ-1)/2 * (u_L - u★)
        # compute ρ★_L by definition of speed of sound
        ρ★_L = γ * p★ / (a★_L^2)
        # [reference] RmSv-4.4
        head_L = u_L - a_L
        tail_L = u★ - a★_L
        S_L = NaN
    end
    # right wave
    if p★ > p_R # right shock
        wave_type_R = Shock
        # shock velocity:
        S_R = u_R + a_R * √((γ+1)/(2γ) * (p★ / p_R) + (γ-1)/(2γ))
        ρ★_R = ρ_R * (p★/p_R + (γ-1)/(γ+1)) / 
               (((γ-1)/(γ+1)) * (p★/p_R) + 1)
        head_R = tail_R = NaN
    else # right rarefaction
        wave_type_R = Rarefaction
        a★_R = a_R - (γ-1)/2 * (u_R - u★)
        ρ★_R = γ * p★ / (a★_R^2)
        head_R = u_R + a_R
        tail_R = u★ + a★_R
        S_R = NaN
    end

    wave_structure_L = NonlinearWaveStructure(
        wave_type_L,
        ρ★_L,
        S_L,
        head_L,
        tail_L
    )
    wave_structure_R = NonlinearWaveStructure(
        wave_type_R,
        ρ★_R,
        S_R,
        head_R,
        tail_R
    )

    return (wave_structure_L, wave_structure_R)
end


"""
    isvacuum(W_L, W_R, eos)

Test whether the given initial states lead to a vacuum in the solution.

# Arguments
- `W_L::PrimitiveState`: initial left state
- `W_R::PrimitiveState`: initial right state
- `eos::PerfectGasEOS`: equation of state

# Returns
- `true` if a vacuum region would appear, `false` otherwise

# References
RmSv-4.6
"""
function isvacuum(W_L::PrimitiveState, W_R::PrimitiveState, eos::PerfectGasEOS)
    ρ_L, u_L, p_L = W_L.ρ, W_L.u, W_L.p
    ρ_R, u_R, p_R = W_R.ρ, W_R.u, W_R.p
    γ = eos.γ

    a_L = √(γ * p_L / ρ_L)
    a_R = √(γ * p_R / ρ_R)

    return (u_R - u_L) >= (2/(γ-1)) * (a_L + a_R)
end


"""
    solve_Riemann_problem(W_L, W_R, eos; init_guess_method=TS, max_iter=50, tol=1e-10)

Solve the one-dimensional Riemann problem exactly for the compressible
Euler equations with a perfect gas equation of state:
1. Check for vacuum condition ([`isvacuum`](@ref)).
2. Solve ``p_*`` via Newton–Raphson ([`solve_p★`](@ref)).
3. Compute ``u_*`` from ``p_*`` ([`calc_u★_from_p★`](@ref)).
4. Determine wave structure ([`calc_wave_structure_from_p★_and_u★`](@ref)).
5. Assemble and return the complete solution.

# Arguments
- `W_L::PrimitiveState`: initial left state ``(\\rho_L, u_L, p_L)``
- `W_R::PrimitiveState`: initial right state ``(\\rho_R, u_R, p_R)``
- `eos::PerfectGasEOS`: equation of state
- `init_guess_method::PressureGuessMethod = TS`: initial guess strategy
- `max_iter::Int = 50`: maximum Newton iterations
- `tol::Real = 1e-10`: convergence tolerance

# Returns
- `ExactRiemannSolution`: the complete exact solution

# Examples
```julia
W_L = PrimitiveState(ρ=1.0, u=0.0, p=1.0)
W_R = PrimitiveState(ρ=0.125, u=0.0, p=0.1)
eos = PerfectGasEOS(γ=1.4)
sol = solve_Riemann_problem(W_L, W_R, eos)
```
"""
function solve_Riemann_problem(W_L::PrimitiveState, W_R::PrimitiveState, eos::PerfectGasEOS;
                              init_guess_method::PressureGuessMethod=TS,
                              max_iter=50, tol=1e-10)

    if isvacuum(W_L, W_R, eos)
        throw(ArgumentError("initial states lead to presence of vacuum in the solution, which is not supported by this solver"))
    end

    p★ = solve_p★(W_L, W_R, eos; init_guess_method=init_guess_method, max_iter=max_iter, tol=tol)
    u★ = calc_u★_from_p★(W_L, W_R, eos, p★)
    left_wave, right_wave =
        calc_wave_structure_from_p★_and_u★(W_L, W_R, eos, p★, u★)
    return ExactRiemannSolution(eos, W_L, W_R, p★, u★, left_wave, right_wave)
end


"""
    sample_solution(x, t, solution::ExactRiemannSolution)

Sample the exact Riemann solution at a given point ``(x, t)``. The solution is self-similar and is solely determined by ``\\xi = x/t``.

# Arguments
- `x`: spatial coordinate
- `t`: time coordinate (must be > 0)
- `solution::ExactRiemannSolution`: the solution of Riemann problem (obtained from [`solve_Riemann_problem`](@ref))

# Returns
- `PrimitiveState`: ``(\\rho, u, p)`` at ``(x, t)``

# References
RmSv-4.5
"""
function sample_solution(x, t, solution::ExactRiemannSolution)
    t <= 0 && throw(ArgumentError("t must be larger than 0"))
    
    ξ = x / t # self similar variable

    γ = solution.eos.γ
    W_L, W_R = solution.W_L, solution.W_R
    ρ_L, u_L, p_L = W_L.ρ, W_L.u, W_L.p
    ρ_R, u_R, p_R = W_R.ρ, W_R.u, W_R.p
    p★, u★ = solution.p★, solution.u★
    L = solution.left_wave
    R = solution.right_wave

    # sample solution
    # [reference] RmSv-4.5

    # LEFT OF CONTACT WAVE
    # ahead of left wave (left shock & rarefaction)
    if L.wave_type == Shock # left shock wave
        if ξ < L.S
            return W_L
        end
    else # left rarefaction
        if ξ < L.head # outside of WLfan
            return W_L
        end
    end
    # inside WLfan (if left rarefaction)
    if L.wave_type == Rarefaction && ξ < L.tail
        a_L = √(γ * p_L / ρ_L)
        ρ = ρ_L * (2/(γ+1) + (γ-1)/((γ+1)*a_L) * (u_L - ξ)) ^ (2/(γ-1))
        u = 2/(γ+1) * ((γ-1)/2 * u_L + ξ + a_L)
        p = p_L * (ρ / ρ_L)^γ
        return PrimitiveState(ρ, u, p)
    end
    # left star region
    if ξ < u★ # left star region
        return PrimitiveState(L.ρ★, u★, p★)
    end
    
    # RIGHT OF CONTACT WAVE
    # ahead of right wave (right shock & rarefaction)
    if R.wave_type == Shock # right shock wave
        if ξ > R.S
            return W_R
        end
    else # right rarefaction
        if ξ > R.head # outside of WRfan
            return W_R
        end
    end
    # inside WRfan (if right rarefaction)
    if R.wave_type == Rarefaction && ξ > R.tail
        a_R = √(γ * p_R / ρ_R)
        ρ = ρ_R * (2/(γ+1) + (γ-1)/((γ+1)*a_R) * (ξ - u_R)) ^ (2/(γ-1))
        u = 2/(γ+1) * ((γ-1)/2 * u_R + ξ - a_R)
        p = p_R * (ρ / ρ_R)^γ
        return PrimitiveState(ρ, u, p)
    else # right star region
        return PrimitiveState(R.ρ★, u★, p★)
    end
end
