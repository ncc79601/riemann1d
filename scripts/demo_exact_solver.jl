abstract type AbstractState end

struct PrimitiveState{T <: Real} <: AbstractState
    ρ::T
    u::T
    p::T
    function PrimitiveState{T}(ρ::T, u::T, p::T) where T <: Real
        return new{T}(ρ, u, p)
    end
end

function PrimitiveState(ρ::Real, u::Real, p::Real)
    ρ_prom, u_prom, p_prom = promote(ρ, u, p)
    return PrimitiveState{typeof(ρ_prom)}(ρ_prom, u_prom, p_prom)
end
PrimitiveState(; ρ::Real, u::Real, p::Real) = PrimitiveState(ρ, u, p) # kwargs

function PrimitiveState(W::AbstractVector{<:Real})
    length(W) == 3 || throw(ArgumentError("state vector W must only contain 3 elements (ρ, u, p)"))
    return PrimitiveState(W[1], W[2], W[3])
end

# -------------------

abstract type AbstractEOS end

struct PerfectGasEOS{T <: Real} <: AbstractEOS 
    γ::T
    function PerfectGasEOS{T}(γ::T) where T <: Real
        γ <= 1.0 && throw(DomainError(γ, "specific heat ratio γ must be larger than 1.0"))
        return new{T}(γ)
    end
end
# external constructors
PerfectGasEOS(γ::Real) = PerfectGasEOS{typeof(γ)}(γ) # auto type inferring
PerfectGasEOS() = PerfectGasEOS(1.4) # default value
PerfectGasEOS(; γ::Real=1.4) = PerfectGasEOS(γ) # kwargs

# -------------------

function guess_p★(W_L::PrimitiveState, W_R::PrimitiveState)
    # extract primitive variables
    p_L = W_L.p; p_R = W_R.p
    # simple arithmetic mean
    return (p_L + p_R) / 2
end

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

using ForwardDiff

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

@enum NonlinearWaveType Shock Rarefaction

struct NonlinearWaveStructure{T <: Real}
    # type of nonlinear wave
    wave_type ::NonlinearWaveType # Shock or Rarefaction

    # single-side star density
    ρ★  ::T

    # wave velocities
    S   ::T            # shock velocities (NaN if rarefaction)
    head::T            # rarefaction head velocity
    tail::T            # rarefaction tail velocity
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


abstract type RiemannSolution end
struct RiemannExactSolution{T <: Real} <: RiemannSolution
    # equation of state
    eos::PerfectGasEOS

    # initial data
    W_L::PrimitiveState{T}
    W_R::PrimitiveState{T}

    # star values
    p★  ::T
    u★  ::T
    ρ★_L::T
    ρ★_R::T
    
    # left wave structure
    wave_type_L::NonlinearWaveType
    S_L   ::T
    head_L::T
    tail_L::T
    # right wave structure
    wave_type_R::NonlinearWaveType
    S_R   ::T
    head_R::T
    tail_R::T
end

function solve_Riemann_problem(W_L::PrimitiveState, W_R::PrimitiveState, eos::PerfectGasEOS)
    p★ = solve_p★(W_L, W_R, eos)
    u★ = calc_u★_from_p★(W_L, W_R, eos, p★)
    wave_structure_L, wave_structure_R =
        calc_wave_structure_from_p★_and_u★(W_L, W_R, eos, p★, u★)

    return RiemannExactSolution(
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

function sample_solution(x, t, solution::RiemannExactSolution)
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