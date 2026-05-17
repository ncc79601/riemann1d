"""
    AbstractState

Abstract supertype for all thermodynamic state representations.
"""
abstract type AbstractState end

"""
    PrimitiveState{T<:Real} <: AbstractState

Concrete representation of a primitive variable state.

# Fields
- `ρ::T`: density
- `u::T`: velocity
- `p::T`: pressure

# Constructors
- `PrimitiveState(ρ::Real, u::Real, p::Real)`: auto-promotes mixed numeric types
- `PrimitiveState(; ρ::Real, u::Real, p::Real)`: kwarg form
- `PrimitiveState(W::AbstractVector{<:Real})`: from a 3-element vector `[ρ, u, p]`
"""
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


"""
    AbstractEOS

Abstract supertype for equations of state.
"""
abstract type AbstractEOS end

"""
    PerfectGasEOS{T<:Real} <: AbstractEOS

Ideal / perfect gas equation of state.

# Fields
- `γ::T`: ratio of specific heats, must be > 1.0

# Constructors
- `PerfectGasEOS(γ::Real)`: auto-infers type parameter `T`
- `PerfectGasEOS(; γ::Real=1.4)`: kwarg form, defaults to ``\\gamma=1.4`` (air)
"""
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
    AbstractRiemannSolution

Abstract supertype for Riemann problem solutions. Can be sampled at arbitrary ``(x, t)`` via [`sample_solution`](@ref).
"""
abstract type AbstractRiemannSolution end

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
