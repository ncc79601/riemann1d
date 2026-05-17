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