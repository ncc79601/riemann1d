
# TODO: docstring
struct NoLimiter <: AbstractLimiter end

struct SwebyLimiter{T <: Real} <: AbstractLimiter
    β::T
end

SwebyLimiter(; β::Real = 2.0) = SwebyLimiter{typeof(β)}(β)

# MinBee and SuperBee as special cases of Sweby limiter
MinBeeLimiter() = SwebyLimiter(β = 1.0)

SuperBeeLimiter() = SwebyLimiter(β = 2.0)

struct UltraBeeLimiter <: AbstractLimiter end

struct vanLeerLimiter <: AbstractLimiter end

struct MCLimiter <: AbstractLimiter end


# similar to CDS, but only use ``\Delta_{i+\frac{1}{2}}`` for ``\Delta_i``
function ξ(r::Real, limiter::NoLimiter)
    return 1.0
end

# Sweby, SuperBee, MinBee
function ξ(r::Real, limiter::SwebyLimiter)
    β = limiter.β
    return max(0, min(1, β*r), min(r, β))
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
    return max(0, min(2r, (1+r)/2, 2))
end
