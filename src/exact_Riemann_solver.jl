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


function guess_p★(W_L::PrimitiveState, W_R::PrimitiveState)
    # extract primitive variables
    p_L = W_L.p; p_R = W_R.p
    # simple arithmetic mean
    return (p_L + p_R) / 2
end


function solve_p★(W_L::PrimitiveState, W_R::PrimitiveState, eos::PerfectGasEOS;
                  max_iter=50, tol=1e-10)
    # TODO: vacuum detection?

    f(p) = pressure_function(p, W_L, W_R, eos) # currying pressure function
    p★ = guess_p★(W_L, W_R)

    for i in 1:max_iter
        residual = f(p★)
        if abs(residual) < tol # converged
            return p★
        end

        deriv = ForwardDiff.derivative(f, p★) # auto differentiation
        if abs(deriv) < 1e-14 # TODO: make it an argument?
            @warn "derivative close to zero, stop iteration"
            return p★
        end
        
        # Newton-Raphson iteration
        Δp = - residual / deriv
        p★ = max(p★ + Δp, 1e-14) # TODO: is this correct?
    end

    @warn "Newton-Raphson not converged, returning current p★"
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


function solve_Riemann_problem(W_L::PrimitiveState, W_R::PrimitiveState, eos::PerfectGasEOS)
    p★ = solve_p★(W_L, W_R, eos)
    u★ = calc_u★_from_p★(W_L, W_R, eos, p★)
    wave_structure_L, wave_structure_R =
        calc_wave_structure_from_p★_and_u★(W_L, W_R, eos, p★, u★)

    return ExactRiemannSolution(
        eos, W_L, W_R, p★, u★,
        wave_structure_L.ρ★,
        wave_structure_R.ρ★,

        wave_structure_L.wave_type,
        wave_structure_L.S,
        wave_structure_L.head,
        wave_structure_L.tail,

        wave_structure_R.wave_type,
        wave_structure_R.S,
        wave_structure_R.head,
        wave_structure_R.tail,
    )
end


function sample_solution(x, t, solution::ExactRiemannSolution)
    t <= 0 && throw(ArgumentError("t must be larger than 0"))
    
    ξ = x / t # self similar variable

    γ = solution.eos.γ
    W_L, W_R = solution.W_L, solution.W_R
    ρ_L, u_L, p_L = W_L.ρ, W_L.u, W_L.p
    ρ_R, u_R, p_R = W_R.ρ, W_R.u, W_R.p
    p★, u★ = solution.p★, solution.u★
    ρ★_L = solution.ρ★_L
    ρ★_R = solution.ρ★_R

    # sample solution
    # [reference] RmSv-4.5

    # LEFT OF CONTACT WAVE
    # ahead of left wave (left shock & rarefaction)
    if solution.wave_type_L == Shock # left shock wave
        if ξ < solution.S_L
            return W_L
        end
    else # left rarefaction
        if ξ < solution.head_L # outside of WLfan
            return W_L
        end
    end
    # inside WLfan (if left rarefaction)
    if solution.wave_type_L == Rarefaction && ξ < solution.tail_L
        a_L = √(γ * p_L / ρ_L)
        ρ = ρ_L * (2/(γ+1) + (γ-1)/((γ+1)*a_L) * (u_L - ξ)) ^ (2/(γ-1))
        u = 2/(γ+1) * ((γ-1)/2 * u_L + ξ + a_L)
        p = p_L * (ρ / ρ_L)^γ
        return PrimitiveState(ρ, u, p)
    end
    # left star region
    if ξ < u★ # left star region
        return PrimitiveState(ρ★_L, u★, p★)
    end
    
    # RIGHT OF CONTACT WAVE
    # ahead of right wave (right shock & rarefaction)
    if solution.wave_type_R == Shock # right shock wave
        if ξ > solution.S_R
            return W_R
        end
    else # right rarefaction
        if ξ > solution.head_R # outside of WRfan
            return W_R
        end
    end
    # inside WRfan (if right rarefaction)
    if solution.wave_type_R == Rarefaction && ξ > solution.tail_R
        a_R = √(γ * p_R / ρ_R)
        ρ = ρ_R * (2/(γ+1) + (γ-1)/((γ+1)*a_R) * (ξ - u_R)) ^ (2/(γ-1))
        u = 2/(γ+1) * ((γ-1)/2 * u_R + ξ - a_R)
        p = p_R * (ρ / ρ_R)^γ
        return PrimitiveState(ρ, u, p)
    else # right star region
        return PrimitiveState(ρ★_R, u★, p★)
    end
end