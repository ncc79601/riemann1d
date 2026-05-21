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


#TODO: support switching between explicit Euler and TVD-RK2?


#TODO: docstring
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
    evolve!(U, grid::UniformGrid1D, eos::PerfectGasEOS, config::SolverConfig)

Run the finite-volume scheme governed by `config.solver` from ``t = 0`` to
`config.max_time`.

`U` is an array of `ConservedState` indexed `1:grid.N` for physical cells.
It is mutated in-place and contains the final state after the call.

# Returns
- `(t_final, n_steps)`: the final time reached and number of time steps taken
"""
function evolve!(
    U::AbstractVector{ConservedState},
    grid::UniformGrid1D,
    eos::PerfectGasEOS,
    config::SolverConfig,
)
    N  = grid.N
    ng = grid.ghost_cells

    # --- pre-allocate work arrays ---
    U_new         = similar(U)
    W             = Vector{PrimitiveState}(undef, N)
    F             = Vector{Flux}(undef, N + 1) # F[i] -> interface i-1/2
    W_L_data      = Vector{PrimitiveState}(undef, N + 2) # right of interface i-1/2
    W_R_data      = Vector{PrimitiveState}(undef, N + 2) # left of interface i+1/2
    boundaries    = make_boundary_faces(grid, TransmissiveBC())
    W_padded_data = Vector{PrimitiveState}(undef, N + 2 * ng)

    # offset arrays
    W_L = OffsetArray(W_L_data, 0 : N + 1)
    W_R = OffsetArray(W_R_data, 0 : N + 1)
    W_padded = OffsetArray(W_padded_data, 1 - ng : N + ng)

    #TODO: config needs to add a ReconstructMethod field
    reconstruction = config.reconstruction
    limiter = config.limiter
    
    t    = 0.0
    step = 0

    while t < config.max_time && step < config.max_steps
        #TODO debug
        # println("[timestep $step]")

        # 1. conserved → primitive
        # for i in 1:N
        #     W[i] = conserved_to_primitive(U[i], eos)
        # end

        # 2. apply boundary conditions to ghost cells
        # apply_bc!(W, W_padded, grid, boundaries)

        #TODO debug
        # @warn "applied bc"
        # @show W_padded

        # 3. reconstruct face values
        # reconstruct!(W_L, W_R, W_padded, grid, reconstruction, limiter)
        #TODO debug
        # @show W
        # @show W_L
        # @show W_R
        
        # 4. compute numerical fluxes at all N+1 interfaces
        # compute_intercell_fluxes!(F, W_L, W_R, config.solver, grid, eos)

        evaluate_fluxes!(F, U, W, W_padded, W_L, W_R, boundaries, grid, eos, config)

        # 5. CFL time step (ramp-up: reduced CFL for initial steps)
        cfl_now = step < config.init_steps ? config.init_cfl : config.cfl
        Δt = compute_Δt(W_padded, eos, grid, cfl_now)
        Δt = min(Δt, config.max_time - t)

        # 6. forward Euler update
        forward_euler_step!(U, U_new, F, grid, Δt)

        # 7. swap and advance
        U, U_new = U_new, U
        t += Δt
        step += 1
    end

    return t, step
end
