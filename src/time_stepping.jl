"""
    max_wave_speed(W::PrimitiveState, eos::PerfectGasEOS) -> Real

Compute the maximum absolute eigenvalue for a single cell:
``\\lambda_\\max = |u| + a`` where ``a = \\sqrt{\\gamma p / \\rho}``.
"""
function max_wave_speed(W::PrimitiveState, eos::PerfectGasEOS)
    a = √(eos.γ * W.p / W.ρ)
    return abs(W.u) + a
end


"""
    max_wave_speed(W_L::PrimitiveState, W_R::PrimitiveState, eos::PerfectGasEOS,
                   method::WaveSpeedMethod) -> Real

Compute the interface maximum wave speed from left and right states using the
specified method.  Used by Riemann solvers that require a two-state estimate.

- `Physical`: ``\\max(|u_L|+a_L,\\, |u_R|+a_R)``
"""
function max_wave_speed(W_L::PrimitiveState, W_R::PrimitiveState, eos::PerfectGasEOS,
                        method::WaveSpeedMethod)
    if method == Physical
        return max(max_wave_speed(W_L, eos), max_wave_speed(W_R, eos))
    else
        throw(ArgumentError("Unsupported wave speed method: $(method)"))
    end
end


"""
    compute_cfl_dt(W_arr, eos::PerfectGasEOS, grid::UniformGrid1D, cfl::Real) -> Real

Compute the maximum time step from the CFL condition.

``\\Delta t = \\text{CFL} \\cdot \\min_i \\frac{\\Delta x}{|u_i| + a_i}``

`W_arr` must be indexed such that physical cells are at `1:grid.N`.
"""
function compute_Δt(W_arr, eos::PerfectGasEOS, grid::UniformGrid1D, cfl::Real)
    λ_max = zero(eltype(W_arr[1].ρ))
    for i in 1:grid.N
        λ_i = max_wave_speed(W_arr[i], eos)
        λ_max = max(λ_max, λ_i)
    end
    return cfl * grid.Δx / λ_max
end


"""
    forward_euler_step!(U, U_new, F, grid::UniformGrid1D, dt::Real)

Advance one time step with the forward Euler method.

``U_i^{n+1} = U_i^n - \\frac{\\Delta t}{\\Delta x} \\left( F_{i+1/2} - F_{i-1/2} \\right)``

- `U` : conserved state array at time ``n`` (indexed `1:grid.N` for physical cells)
- `U_new` : conserved state array at time ``n+1`` (same layout as `U`)
- `F` : numerical fluxes at interfaces (length `grid.N+1`, indexed `1:grid.N+1`)
"""
function forward_euler_step!(U, U_new, F, grid::UniformGrid1D, Δt::Real)
    N = grid.N
    c = Δt / grid.Δx
    for i in 1:N
        Δmass     = F[i+1].mass     - F[i].mass
        Δmomentum = F[i+1].momentum - F[i].momentum
        Δenergy   = F[i+1].energy   - F[i].energy

        U_new[i] = ConservedState(
            U[i].ρ  - c * Δmass,
            U[i].ρu - c * Δmomentum,
            U[i].E  - c * Δenergy,
        )
    end
end
