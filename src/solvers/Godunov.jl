"""
    GodunovSolver <: AbstractRiemannSolver

First-order Godunov finite-volume scheme.  Uses the exact Riemann solver to
compute numerical fluxes at each cell interface.
"""
struct GodunovSolver <: AbstractRiemannSolver end


"""
    compute_numerical_flux(solver::GodunovSolver, W_L, W_R, eos)

Godunov flux: solve the exact Riemann problem and evaluate the physical flux
at ``x/t = 0``.
"""
function compute_numerical_flux(
    solver::GodunovSolver,
    W_L::PrimitiveState,
    W_R::PrimitiveState,
    eos::PerfectGasEOS,
)
    return compute_numerical_flux(ExactSolver(), W_L, W_R, eos)
end


"""
    compute_intercell_fluxes!(F, solver, W_L_arr, W_R_arr, eos)

Fill the flux vector `F` (length `N+1`) with numerical fluxes computed from
reconstructed face values `W_L_arr` / `W_R_arr`.
"""
function compute_intercell_fluxes!(
    F::Vector{Flux},
    solver::AbstractRiemannSolver,
    W_L_arr::Vector{PrimitiveState},
    W_R_arr::Vector{PrimitiveState},
    eos::PerfectGasEOS,
)
    for i in eachindex(F)
        F[i] = compute_numerical_flux(solver, W_L_arr[i], W_R_arr[i], eos)
    end
end


"""
    evolve!(U, grid::UniformGrid1D, eos::PerfectGasEOS, config::SolverConfig)

Run the finite-volume Godunov scheme from ``t = 0`` to `config.max_time`.

`U` is an array of `ConservedState` indexed `1:grid.N` for physical cells.
It is mutated in-place and contains the final state after the call.

# Returns
- `(t_final, n_steps)`: the final time reached and number of time steps taken
"""
function evolve!(
    U,
    grid::UniformGrid1D,
    eos::PerfectGasEOS,
    config::SolverConfig,
)
    N  = grid.N
    ng = grid.ghost_cells

    # --- pre-allocate work arrays ---
    U_new          = similar(U)                              # same shape as U
    W              = Vector{PrimitiveState}(undef, N)        # physical primitive states
    F              = Vector{Flux}(undef, N + 1)              # intercell fluxes
    boundaries     = make_boundary_faces(grid, TransmissiveBC())
    W_padded_data  = Vector{PrimitiveState}(undef, N + 2 * ng)
    W_padded       = OffsetArray(W_padded_data, 1 - ng : N + ng)

    t    = 0.0
    step = 0

    while t < config.max_time && step < config.max_steps
        # 1. conserved → primitive
        for i in 1:N
            W[i] = conserved_to_primitive(U[i], eos)
        end

        # 2. pad with ghost cells + apply BC (reuses pre-allocated W_padded)
        apply_bc!(W, W_padded, grid, boundaries)

        # 3. reconstruct face values
        W_L, W_R = reconstruct_face_values(W_padded, config.limiter, grid)

        # 4. compute numerical fluxes at all N+1 interfaces
        compute_intercell_fluxes!(F, config.solver, W_L, W_R, eos)

        # 5. CFL time step
        Δt = compute_Δt(W_padded, eos, grid, config.cfl)
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
