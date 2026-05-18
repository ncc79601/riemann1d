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