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
    ConservedState{T<:Real} <: AbstractState
Concrete representation of a conserved variable state.
# Fields
- `ρ::T`: density
- `ρu::T`: momentum
- `E ::T`: total energy, ``E = \\rho \\left( e + \\frac{1}{2} u^2 \\right)``
# Constructors
- `ConservedState(ρ::Real, ρu::Real, E::Real)`: auto-promotes mixed numeric types
- `ConservedState(; ρ::Real, ρu::Real, E::Real)`: kwarg form
- `ConservedState(U::AbstractVector{<:Real})`: from a 3-element vector `[ρ, ρu, E]`
"""
struct ConservedState{T <: Real} <: AbstractState
    ρ ::T
    ρu::T
    E ::T
    function ConservedState{T}(ρ::T, ρu::T, E::T) where T <: Real
        return new{T}(ρ, ρu, E)
    end
end

function ConservedState(ρ::Real, ρu::Real, E::Real)
    ρ_prom, ρu_prom, E_prom = promote(ρ, ρu, E)
    return ConservedState{typeof(ρ_prom)}(ρ_prom, ρu_prom, E_prom)
end
ConservedState(; ρ::Real, ρu::Real, E::Real) = ConservedState(ρ, ρu, E) # kwargs
function ConservedState(U::AbstractVector{<:Real})
    length(U) == 3 || throw(ArgumentError("state vector U must only contain 3 elements (ρ, ρu, E)"))
    return ConservedState(U[1], U[2], U[3])
end


# conversion between primitive and conserved variables
"""
    conserved_to_primitive(U::ConservedState, eos::AbstractEOS)

Convert a conserved variable state to a primitive variable state.
"""
function conserved_to_primitive(U::ConservedState, eos::AbstractEOS)
    ρ = U.ρ
    u = U.ρu / ρ
    E = U.E
    p = (eos.γ - 1) * (E - 0.5 * ρ * u^2)
    return PrimitiveState(ρ, u, p)
end

"""
    primitive_to_conserved(::PrimitiveState, eos::AbstractEOS)
Convert a primitive variable state to a conserved variable state.
"""
function primitive_to_conserved(W::PrimitiveState, eos::AbstractEOS)
    ρ = W.ρ
    u = W.u
    p = W.p
    E = p / (eos.γ - 1) + 0.5 * ρ * u^2
    return ConservedState(ρ, ρ * u, E)
end


"""
    sound_speed(W::PrimitiveState, eos::PerfectGasEOS) -> Real

Speed of sound ``a = \\sqrt{\\gamma p / \\rho}`` for a perfect gas.
"""
function sound_speed(W::PrimitiveState, eos::PerfectGasEOS)
    return √(eos.γ * W.p / W.ρ)
end


# flux
"""
    Flux{T<:Real} <: AbstractFlux

Concrete representation of the flux vector for the compressible Euler equations.

# Fields
- `mass::T`: mass flux, ``F_1 = \\rho u``
- `momentum::T`: momentum flux, ``F_2 = \\rho u^2 + p``
- `energy::T`: energy flux, ``F_3 = u \\left( E + p \\right)``

# Constructors
- `Flux(mass::Real, momentum::Real, energy::Real)`: auto-promotes mixed numeric types
- `Flux(; mass::Real, momentum::Real, energy::Real)`: kwarg form
- `Flux(F::AbstractVector{<:Real})
    from a 3-element vector `[F_mass, F_momentum, F_energy]`
- `Flux(W::PrimitiveState, eos::PerfectGasEOS)`: physical flux from primitive state and EOS
- `Flux(U::ConservedState, eos::AbstractEOS)`: physical flux from conserved state and EOS
"""
struct Flux{T <: Real} <: AbstractFlux
    mass::T
    momentum::T
    energy::T
    function Flux{T}(mass::T, momentum::T, energy::T) where T <: Real
        return new{T}(mass, momentum, energy)
    end
end

function Flux(mass::Real, momentum::Real, energy::Real)
    mass_prom, momentum_prom, energy_prom = promote(mass, momentum, energy)
    return Flux{typeof(mass_prom)}(mass_prom, momentum_prom, energy_prom)
end
Flux(; mass::Real, momentum::Real, energy::Real) = Flux(mass, momentum, energy) # kwargs
function Flux(F::AbstractVector{<:Real})
    length(F) == 3 || throw(ArgumentError("flux vector F must only contain 3 elements (mass, momentum, energy)"))
    return Flux(F[1], F[2], F[3])
end

"""
    Flux(W::PrimitiveState, eos::PerfectGasEOS)

Physical flux vector constructed from a primitive variable state and an equation of state.
"""
function Flux(W::PrimitiveState, eos::PerfectGasEOS)
    ρ = W.ρ
    u = W.u
    p = W.p
    E = p / (eos.γ - 1) + 0.5 * ρ * u^2

    F_mass     = ρ * u
    F_momentum = ρ * u^2 + p
    F_energy   = u * (E + p)
    return Flux(F_mass, F_momentum, F_energy)
end

"""
    Flux(U::ConservedState, eos::PerfectGasEOS)
Physical flux vector constructed from a conserved variable state and an equation of state.
"""
function Flux(U::ConservedState, eos::PerfectGasEOS)
    ρ = U.ρ
    ρu = U.ρu
    E = U.E
    u = ρu / ρ
    p = (eos.γ - 1) * (E - 0.5 * ρ * u^2)

    F_mass     = ρu
    F_momentum = ρu^2 / ρ + p
    F_energy   = u * (E + p)
    return Flux(F_mass, F_momentum, F_energy)
end