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
    function PerfectGasEOS{T}(γ::T) where {T <: Real}
        γ <= 1.0 && throw(DomainError(γ, "specific heat ratio γ must be larger than 1.0"))
        return new{T}(γ)
    end
end
# external constructors
function PerfectGasEOS(γ::Real)
    PerfectGasEOS{typeof(γ)}(γ)
end # auto type inferring
function PerfectGasEOS(; γ::Real = 1.4)
    PerfectGasEOS(γ)
end # kwargs
