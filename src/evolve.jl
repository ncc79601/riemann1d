"""
    WaveSpeedMethod

Enumeration of wave speed estimation methods.

- `Physical`: use physical eigenvalues ``\\lambda_\\pm = u \\pm a``
"""
@enum WaveSpeedMethod Physical


"""
    max_wave_speed(W::PrimitiveState, eos::PerfectGasEOS) -> Real

Compute the maximum absolute eigenvalue for a single cell:
``\\lambda_\\max = |u| + a`` where ``a = \\sqrt{\\gamma p / \\rho}``.
"""
function max_wave_speed(W::PrimitiveState, eos::PerfectGasEOS)
    a = sound_speed(W, eos)
    return abs(W.u) + a
end


"""
    max_wave_speed(W_L::PrimitiveState, W_R::PrimitiveState, eos::PerfectGasEOS,
                   method::WaveSpeedMethod) -> Real

Compute the interface maximum wave speed ``S_\\max`` from left and right states using the
specified method.  Used by Riemann solvers that require a two-state estimate.

method:
- `Physical`: ``S_\\max = \\max(|u_L|+a_L,\\, |u_R|+a_R)``
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
    compute_Δt(W_arr, eos::PerfectGasEOS, grid::UniformGrid1D, cfl::Real) -> Real

Compute the maximum time step from the CFL condition.

``\\Delta t = C_\\text{cfl} \\cdot \\frac{\\Delta x}{S_\\max}``

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
    compute_intercell_fluxes!(F, solver, W_L_arr, W_R_arr, eos)

Fill the flux vector `F` (length `N+1`) with numerical fluxes computed from
reconstructed face values `W_L_arr` / `W_R_arr`.
"""
function compute_intercell_fluxes!(
    F::AbstractVector{Flux},
    W_L_arr::AbstractVector{PrimitiveState},
    W_R_arr::AbstractVector{PrimitiveState},
    solver::AbstractRiemannSolver,
    grid::UniformGrid1D,
    eos::PerfectGasEOS,
)
    N = grid.N

    # F[i] -> interface i-1/2
    # W_L_arr[i] -> right of interface i-1/2
    # W_R_Arr[i] -> left of interface i+1/2
    for i in 1:N+1
        F[i] = compute_numerical_flux(
            solver,
            W_R_arr[i-1], # W_R_arr[i-1] as W_L
            W_L_arr[i],   # W_L_arr[i] as W_R
            eos
    )
    end
end


"""
    evaluate_fluxes!(F, U, W, W_padded, W_L, W_R, boundaries, grid, eos, config)

Perform one complete flux-evaluation pipeline for a finite-volume scheme:

1. Convert conserved ``U`` to primitive ``W``
2. Apply boundary conditions to ghost cells in `W_padded`
3. Reconstruct face values ``W_L``, ``W_R``
4. Compute numerical fluxes ``F`` at all ``N+1`` interfaces
"""
function evaluate_fluxes!(
    F         ::AbstractArray{Flux},
    U         ::AbstractArray{ConservedState},
    W         ::AbstractArray{PrimitiveState},
    W_padded  ::AbstractArray{PrimitiveState},
    W_L       ::AbstractArray{PrimitiveState},
    W_R       ::AbstractArray{PrimitiveState},
    boundaries::AbstractArray{BoundaryFace},
    grid      ::UniformGrid1D,
    eos       ::PerfectGasEOS,
    config    ::SolverConfig
)
    N  = grid.N

    # 1. conserved → primitive
    for i in 1:N
        W[i] = conserved_to_primitive(U[i], eos)
    end

    # 2. apply boundary conditions to ghost cells
    apply_bc!(W, W_padded, grid, boundaries)

    # 3. reconstruct face values
    reconstruct!(W_L, W_R, W_padded, grid, config.reconstruction, config.limiter)
    
    # 4. compute numerical fluxes at all N+1 interfaces
    compute_intercell_fluxes!(F, W_L, W_R, config.solver, grid, eos)
end


"""
    forward_euler_step!(U, U_new, F, grid::UniformGrid1D, dt::Real)

Advance one time step with the forward Euler method.

``U_i^{n+1} = U_i^n - \\frac{\\Delta t}{\\Delta x} \\left( F_{i+1/2} - F_{i-1/2} \\right)``

- `U` : conserved state array at time ``n`` (indexed `1:grid.N` for physical cells)
- `U_new` : conserved state array at time ``n+1`` (same layout as `U`)
- `F` : numerical fluxes at interfaces (length `grid.N+1`, indexed `1:grid.N+1`)

Note: allocates one `ConservedState` per cell per step. For zero-allocation
operation, swap to `StructArray{ConservedState}` and write component arrays
directly.
"""
function forward_euler_step!(U, U_new, F, grid::UniformGrid1D, Δt::Real)

    N  = grid.N
    Δx = grid.Δx

    for i in 1:N
        # cell i;  F[i] -> interface i-1/2
        U_new[i] = ConservedState(
            U[i].ρ  + (Δt/Δx) * (F[i].mass     - F[i+1].mass    ),
            U[i].ρu + (Δt/Δx) * (F[i].momentum - F[i+1].momentum),
            U[i].E  + (Δt/Δx) * (F[i].energy   - F[i+1].energy  ),
        )
    end
end


"""
    ExplicitEuler <: AbstractIntegrator

First-order explicit Euler time integrator.
"""
struct ExplicitEuler <: AbstractIntegrator end

"""
    TVDRK2 <: AbstractIntegrator

Second-order TVD Runge–Kutta (SSP-RK2) time integrator.

``U^{(1)}  = U^n + \\Delta t \\, L(U^n)`` \\
``U^{(2)}  = U^{(1)} + \\Delta t \\, L(U^{(1)})`` \\
``U^{n+1}  = \\frac{1}{2} U^n + \\frac{1}{2} U^{(2)}``
"""
struct TVDRK2 <: AbstractIntegrator end


"""
    allocate_work_arrays(U, grid) -> NamedTuple

Pre-allocate all work arrays needed by one `evolve!` call and return them
as a `NamedTuple` with keys `U_new`, `W`, `F`, `W_L`, `W_R`, `boundaries`,
`W_padded`.
"""
function allocate_work_arrays(U, grid::UniformGrid1D)
    N  = grid.N
    ng = grid.ghost_cells

    U_new         = similar(U)
    W             = Vector{PrimitiveState}(undef, N)
    F             = Vector{Flux}(undef, N + 1)
    W_L_data      = Vector{PrimitiveState}(undef, N + 2)
    W_R_data      = Vector{PrimitiveState}(undef, N + 2)
    boundaries    = make_boundary_faces(grid, TransmissiveBC())
    W_padded_data = Vector{PrimitiveState}(undef, N + 2 * ng)

    W_L = OffsetArray(W_L_data, 0 : N + 1)
    W_R = OffsetArray(W_R_data, 0 : N + 1)
    W_padded = OffsetArray(W_padded_data, 1 - ng : N + ng)

    return (; U_new, W, F, W_L, W_R, boundaries, W_padded)
end


"""
    evolve!(U, grid, eos, config)

Run the finite-volume scheme governed by `config.solver` from ``t = 0`` to
`config.max_time`. Dispatches to the integrator specified in `config.integrator`.

``U`` is an array of `ConservedState` indexed `1:grid.N` for physical cells.
It is mutated in-place and contains the final state after the call.

# Returns
- `(t_final, n_steps)`: the final time reached and number of time steps taken
"""
function evolve!(
    U::     AbstractVector{ConservedState},
    grid::  UniformGrid1D,
    eos::   PerfectGasEOS,
    config::SolverConfig,
)
    integrator = config.integrator
    return evolve!(U, grid, eos, config, integrator)
end


"""
    evolve!(U, grid::UniformGrid1D, eos::PerfectGasEOS, config::SolverConfig, integrator::ExplicitEuler)

Use explicit Euler method as the time integrator and evolve the solution.

# Returns
- `(t_final, n_steps)`: the final time reached and number of time steps taken
"""
# Explicit Euler
function evolve!(
    U::AbstractVector{ConservedState},
    grid::UniformGrid1D,
    eos::PerfectGasEOS,
    config::SolverConfig,
    integrator::ExplicitEuler
)
    w = allocate_work_arrays(U, grid)

    t    = 0.0
    step = 0

    while t < config.max_time && step < config.max_steps
        # 1. conserved → primitive
        # 2. apply boundary conditions to ghost cells
        # 3. reconstruct face values
        # 4. compute numerical fluxes at all N+1 interfaces
        evaluate_fluxes!(w.F, U, w.W, w.W_padded, w.W_L, w.W_R,
                         w.boundaries, grid, eos, config)

        # 5. CFL time step (ramp-up: reduced CFL for initial steps)
        cfl_now = step < config.init_steps ? config.init_cfl : config.cfl
        Δt = compute_Δt(w.W_padded, eos, grid, cfl_now)
        Δt = min(Δt, config.max_time - t)

        # 6. forward Euler update
        forward_euler_step!(U, w.U_new, w.F, grid, Δt)

        # 7. swap and advance
        U, w.U_new = w.U_new, U
        t += Δt
        step += 1
    end

    return t, step
end


"""
    evolve!(U, grid::UniformGrid1D, eos::PerfectGasEOS, config::SolverConfig, integrator::TVDRK2)

Use TVD-RK2 as the time integrator and evolve the solution.

# Returns
- `(t_final, n_steps)`: the final time reached and number of time steps taken
"""
function evolve!(
    U::AbstractVector{ConservedState},
    grid::UniformGrid1D,
    eos::PerfectGasEOS,
    config::SolverConfig,
    integrator::TVDRK2
)
    w = allocate_work_arrays(U, grid)
    U_1 = similar(U)  # extra stage-1 buffer for TVDRK2

    t    = 0.0
    step = 0

    while t < config.max_time && step < config.max_steps
        # stage 1: evaluate at tⁿ
        evaluate_fluxes!(w.F, U, w.W, w.W_padded, w.W_L, w.W_R,
                         w.boundaries, grid, eos, config)

        cfl_now = step < config.init_steps ? config.init_cfl : config.cfl
        Δt = compute_Δt(w.W_padded, eos, grid, cfl_now)
        Δt = min(Δt, config.max_time - t)

        # U¹ = Uⁿ + Δt L(Uⁿ)
        forward_euler_step!(U, U_1, w.F, grid, Δt)

        # stage 2: evaluate at tⁿ⁺¹
        evaluate_fluxes!(w.F, U_1, w.W, w.W_padded, w.W_L, w.W_R,
                         w.boundaries, grid, eos, config)

        # U² = U¹ + Δt L(U¹)
        forward_euler_step!(U_1, w.U_new, w.F, grid, Δt)

        # convex average: Uⁿ⁺¹ = 0.5 * (Uⁿ + U²)
        for i in 1:grid.N
            U[i] = ConservedState(
                0.5 * (U[i].ρ  + w.U_new[i].ρ ),
                0.5 * (U[i].ρu + w.U_new[i].ρu),
                0.5 * (U[i].E  + w.U_new[i].E ),
            )
        end

        # advance
        t += Δt
        step += 1
    end

    return t, step
end
