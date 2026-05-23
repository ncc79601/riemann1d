
"""
    NoLimiter <: AbstractLimiter

Identity limiter: ``\\xi(r) = 1``. Applying no limiting is equivalent to central
differencing, which is non-TVD and will produce oscillations near discontinuities.
"""
struct NoLimiter <: AbstractLimiter end

"""
    SwebyLimiter{T <: Real} <: AbstractLimiter

Sweby limiter: ``\\xi(r) = \\text{max}(0, \\text{min}(1, \\beta r), \\text{min}(r, \\beta))``.
"""
struct SwebyLimiter{T <: Real} <: AbstractLimiter
    β::T
end

function SwebyLimiter(; β::Real = 2.0)
    SwebyLimiter{typeof(β)}(β)
end

# MinBee and SuperBee as special cases of Sweby limiter
"""
    MinBeeLimiter() -> SwebyLimiter

Minbee limiter. Returns a [`SwebyLimiter`](@ref) object with `β = 1.0`.

``\\xi(r) = \\text{max}(0, \\text{min}(1, \\beta r), \\text{min}(r, \\beta))``.
"""
function MinBeeLimiter()
    SwebyLimiter(β = 1.0)
end

"""
    SuperBeeLimiter() -> SwebyLimiter

Superbee limiter. Returns a [`SwebyLimiter`](@ref) object with `β = 2.0`.

``\\xi(r) = \\text{max}(0, \\text{min}(1, 2r), \\text{min}(r, 2))``.
"""
function SuperBeeLimiter()
    SwebyLimiter(β = 2.0)
end

"""
    UltraBeeLimiter <: AbstractLimiter

Ultrabee limiter: ``\\xi(r) = \\text{max}(0, \\text{min}(2r, 2))``.
"""
struct UltraBeeLimiter <: AbstractLimiter end

"""
    vanLeerLimiter <: AbstractLimiter

van Leer limiter: ``\\xi(r) = \\frac{r + |r|}{1 + |r|}``.
"""
struct vanLeerLimiter <: AbstractLimiter end

"""
    MCLimiter <: AbstractLimiter

MC limiter: ``\\xi(r) = \\text{max}(0, \\text{min}(2r, (1 + r) / 2, 2))``.
"""
struct MCLimiter <: AbstractLimiter end

"""
    ξ(r::Real, limiter::AbstractLimiter) -> Real

Calculates the limiter value ``\\xi(r)``.
"""
# similar to CDS, but only use ``\Delta_{i+\frac{1}{2}}`` for ``\Delta_i``
function ξ(r::Real, limiter::NoLimiter)
    return 1.0
end

# Sweby, SuperBee, MinBee
function ξ(r::Real, limiter::SwebyLimiter)
    β = limiter.β
    return max(0, min(1, β * r), min(r, β))
end

# UltraBee
function ξ(r::Real, limiter::UltraBeeLimiter)
    return max(0, min(2r, 2))
end

# vanLeer
function ξ(r::Real, limiter::vanLeerLimiter)
    return (r + abs(r)) / (1 + abs(r))
end

# MC
function ξ(r::Real, limiter::MCLimiter)
    return max(0, min(2r, (1 + r) / 2, 2))
end
