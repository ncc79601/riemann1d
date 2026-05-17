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
PerfectGasEOS(; γ::Real=1.4) = PerfectGasEOS(γ) # kwargs



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



abstract type AbstractRiemannSolution end

struct ExactRiemannSolution{T <: Real} <: AbstractRiemannSolution
    eos::PerfectGasEOS
    W_L::PrimitiveState{T}
    W_R::PrimitiveState{T}
    p★::T
    u★::T
    left_wave::NonlinearWaveStructure{T}
    right_wave::NonlinearWaveStructure{T}
end