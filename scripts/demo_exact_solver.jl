abstract type AbstractState end

struct PrimitiveState{T<:Real} <: AbstractState
    ρ::T
    u::T
    p::T
    function PrimitiveState{T}(ρ::T, u::T, p::T) where T<:Real
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
    # Reference: RmSv-4.2
    # extract primitive variables
    ρ_K = W_K.ρ; u_K = W_K.u; p_K = W_K.p
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
    u_L = W_L.u; u_R = W_R.u
    return pressure_function(p, W_L, eos) + pressure_function(p, W_R, eos) + (u_R - u_L)
end

using ForwardDiff

function solve_pressure(W_L::PrimitiveState, W_R::PrimitiveState, eos::PerfectGasEOS;
                        max_iter=50, tol=1e-10)
    # TODO: vacuum detection?

    f(p) = pressure_function(p, W_L, W_R, eos) # currying pressure function
    p★ = guess_p★(W_L, W_R)

    for i in 1:max_iter
        residual = f(p★)
        if abs(residual) < tol # converged
            return p★
        end
        deriv = ForwardDiff.derivative(f, p★)
        if abs(deriv) < 1e-14 # TODO: make it an argument?
            @warn "derivative close to zero, stop iteration"
            return p★
        end
        
        # Newton-Raphson
        Δp = - residual / deriv
        p★ = max(p★ + Δp, 1e-14) # TODO: is this correct?
    end

    @warn "Newton-Raphson not converged, returning current p★"
    return p★
end

# construct left and right data
W_L = PrimitiveState(ρ=1.0, u=0.0, p=1.0)
W_R = PrimitiveState(ρ=0.125, u=0.0, p=0.1)

eos = PerfectGasEOS(γ=1.4)

p★ = solve_pressure(W_L, W_R, eos)
println("p★ = ", p★)