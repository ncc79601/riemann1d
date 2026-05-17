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


function pressure_function(p::Real, W_L::PrimitiveState, W_R::PrimitiveState, eos::PerfectGasEOS)
    # extract primitive variables
    u_L, u_R = W_L.u, W_R.u
    return pressure_function(p, W_L, eos) + pressure_function(p, W_R, eos) + (u_R - u_L)
end


@enum PressureGuessMethod TR TS PV


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

        p₀ = guess_p★(W_L, W_R, eos,method=PV) # PV guess as initial guess
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


function solve_p★_Newton_loop(W_L::PrimitiveState, W_R::PrimitiveState, eos::PerfectGasEOS, p0;
                             max_iter=50, tol=1e-10)
    f(p) = pressure_function(p, W_L, W_R, eos)
    p★ = p0

    for i in 1:max_iter
        residual = f(p★)
        if isnan(residual)
            error("pressure function returned NaN at p★ = $p★")
        end
        if abs(residual) < tol
            return (p★, i)
        end

        deriv = ForwardDiff.derivative(f, p★)
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


function solve_p★(W_L::PrimitiveState, W_R::PrimitiveState, eos::PerfectGasEOS;
                  init_guess_method::PressureGuessMethod=TS, max_iter=50, tol=1e-10)
    p0 = guess_p★(W_L, W_R, eos, method=init_guess_method)
    p★, _ = solve_p★_Newton_loop(W_L, W_R, eos, p0; max_iter=max_iter, tol=tol)
    return p★
end


function calc_u★_from_p★(
    W_L::PrimitiveState,
    W_R::PrimitiveState,
    eos::PerfectGasEOS,
    p★ ::Real
)
    # [reference] RmSv-4.2
    u_L, u_R = W_L.u, W_R.u
    f_L = pressure_function(p★, W_L, eos)
    f_R = pressure_function(p★, W_R, eos)
    return 0.5 * (u_L + u_R) + 0.5 * (f_R - f_L)
end


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


function isvacuum(W_L::PrimitiveState, W_R::PrimitiveState, eos::PerfectGasEOS)
    # [reference] RmSv-4.6
    ρ_L, u_L, p_L = W_L.ρ, W_L.u, W_L.p
    ρ_R, u_R, p_R = W_R.ρ, W_R.u, W_R.p
    γ = eos.γ

    a_L = √(γ * p_L / ρ_L)
    a_R = √(γ * p_R / ρ_R)

    return (u_R - u_L) >= (2/(γ-1)) * (a_L + a_R)
end


function solve_Riemann_problem(W_L::PrimitiveState, W_R::PrimitiveState, eos::PerfectGasEOS;
                              init_guess_method::PressureGuessMethod=TS,
                              max_iter=50, tol=1e-10)

    if isvacuum(W_L, W_R, eos)
        @error "initial states lead to presence of vacuum in the solution, which is not supported by this solver"
    end

    p★ = solve_p★(W_L, W_R, eos; init_guess_method=init_guess_method, max_iter=max_iter, tol=tol)
    u★ = calc_u★_from_p★(W_L, W_R, eos, p★)
    left_wave, right_wave =
        calc_wave_structure_from_p★_and_u★(W_L, W_R, eos, p★, u★)
    return ExactRiemannSolution(eos, W_L, W_R, p★, u★, left_wave, right_wave)
end


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