# Design Document: Approximate Riemann Solvers & MUSCL-Hancock Scheme

> **Project:** Riemann1D  
> **Date:** 2026-05-17  
> **Status:** Draft  

---

## 1. Overview

This document specifies the architecture for extending `Riemann1D` from an exact Riemann solver library to a full 1D finite-volume Euler solver supporting:

- **Approximate Riemann solvers:** HLLC, Roe (with Harten-Hyman entropy fix)
- **MUSCL reconstruction:** piecewise-linear reconstruction with slope limiters
- **Godunov first-order scheme:** baseline comparison
- **Forward Euler time integration**
- **Transmissive boundary conditions** only

---

## 2. Module Organization

```
src/
├── Riemann1D.jl                    # [modify] include new submodules, update exports
├── types.jl                        # [modify] add ConservativeState, Flux, UniformGrid1D, SolverConfig
├── eos.jl                          # [rewrite] EOS utilities: sound speed, internal energy, flux function
├── exact_Riemann_solver.jl         # [no change]
├── limiters.jl                     # [rewrite] slope limiter implementations
├── reconstruction.jl               # [new] MUSCL reconstruction
├── grid.jl                         # [new] 1D uniform grid + transmissive BC
├── time_stepping.jl                # [new] CFL condition + Forward Euler
└── riemann_solvers/
    ├── interface.jl                # [new] AbstractRiemannSolver + riemann_flux dispatch
    ├── exact.jl                    # [rewrite] wrap exact solver as flux interface
    ├── HLLC.jl                     # [rewrite] HLLC approximate solver
    ├── Roe.jl                      # [rewrite] Roe solver with Harten-Hyman entropy fix
    └── Godunov.jl                  # [rewrite] first-order Godunov (no MUSCL)
```

**Loading order** in `Riemann1D.jl`:
1. `using ForwardDiff`
2. `include("types.jl")`
3. `include("eos.jl")`
4. `include("exact_Riemann_solver.jl")`
5. `include("riemann_solvers/interface.jl")`
6. `include("riemann_solvers/exact.jl")`
7. `include("riemann_solvers/HLLC.jl")`
8. `include("riemann_solvers/Roe.jl")`
9. `include("riemann_solvers/Godunov.jl")`
10. `include("limiters.jl")`
11. `include("reconstruction.jl")`
12. `include("grid.jl")`
13. `include("time_stepping.jl")`

---

## 3. New Data Types

### 3.1 `ConservativeState{T}` (`types.jl`)

Conserved variables for the 1D Euler equations.

```julia
struct ConservativeState{T}
    ρ::T    # density
    ρu::T   # momentum
    ρE::T   # total energy per unit volume, ρE = p/(γ-1) + ½ρu²
end
```

### 3.2 `Flux{T}` (`types.jl`)

Numerical flux vector (same components as the physical flux for Euler).

```julia
struct Flux{T}
    mass::T      # ρu
    momentum::T  # ρu² + p
    energy::T    # u(ρE + p)
end
```

### 3.3 `UniformGrid1D{T}` (`grid.jl`)

```julia
struct UniformGrid1D{T}
    xmin::T
    xmax::T
    N::Int          # number of cells
    Δx::T           # cell width
    xc::Vector{T}   # cell centre coordinates (length N)
    xf::Vector{T}   # cell face coordinates  (length N+1)
end
```

### 3.4 `SolverConfig` (`types.jl`)

Aggregate simulation parameters.

```julia
struct SolverConfig{S<:AbstractRiemannSolver, L<:AbstractLimiter, T}
    solver::S
    limiter::L
    cfl::T
    max_time::T
    max_steps::Int
    use_muscl::Bool      # true = 2nd-order MUSCL, false = 1st-order Godunov
end
```

---

## 4. EOS Utilities (`eos.jl`)

Functions that operate on `PrimitiveState` and `PerfectGasEOS`:

```julia
"""
    sound_speed(W::PrimitiveState, eos::PerfectGasEOS) -> Float64

Returns `a = sqrt(γ * p / ρ)`.
"""
function sound_speed end

"""
    internal_energy(W::PrimitiveState, eos::PerfectGasEOS) -> Float64

Returns `e = p / (ρ * (γ - 1))`.
"""
function internal_energy end

"""
    total_energy(W::PrimitiveState, eos::PerfectGasEOS) -> Float64

Returns `E = internal_energy(W, eos) + 0.5 * u²`.
"""
function total_energy end

"""
    total_enthalpy(W::PrimitiveState, eos::PerfectGasEOS) -> Float64

Returns `H = (ρ*E + p) / ρ`.
"""
function total_enthalpy end

"""
    euler_flux(W::PrimitiveState) -> Flux

Physical flux `F(W) = (ρu, ρu² + p, u(ρE + p))`.
"""
function euler_flux end

"""
    primitive_to_conservative(W::PrimitiveState, eos::PerfectGasEOS) -> ConservativeState
"""
function primitive_to_conservative end

"""
    conservative_to_primitive(U::ConservativeState, eos::PerfectGasEOS) -> PrimitiveState
"""
function conservative_to_primitive end
```

---

## 5. Riemann Solver Interface (`riemann_solvers/interface.jl`)

### 5.1 Abstract Type

```julia
abstract type AbstractRiemannSolver end
```

### 5.2 Core Dispatch Function

```julia
"""
    riemann_flux(solver::AbstractRiemannSolver, W_L::PrimitiveState, W_R::PrimitiveState, eos::PerfectGasEOS) -> Flux

Compute the Godunov numerical flux at an interface separating states `W_L` (left)
and `W_R` (right). Each concrete solver implements this method via dispatch.
"""
function riemann_flux end
```

### 5.3 Concrete Solvers

| Solver struct | File | Notes |
|---------------|------|-------|
| `HLLCSolver` | `riemann_solvers/HLLC.jl` | Einfeldt wave-speed estimates |
| `RoeSolver` | `riemann_solvers/Roe.jl` | Roe average + Harten-Hyman entropy fix (`δ = 0.1`) |
| `ExactSolver` | `riemann_solvers/exact.jl` | Calls `solve_Riemann_problem` then `sample_solution(0, 1, sol)` |
| `GodunovSolver` | `riemann_solvers/Godunov.jl` | Alias or thin wrapper — same as exact but implies first-order context |

---

## 6. HLLC Solver (`riemann_solvers/HLLC.jl`)

### 6.1 Algorithm

Given left/right states `W_L`, `W_R`:

1. **Wave speed estimates** (Einfeldt / Davis):
   ```
   a_L = sound_speed(W_L, eos),  a_R = sound_speed(W_R, eos)
   u_tilde = (√ρ_L * u_L + √ρ_R * u_R) / (√ρ_L + √ρ_R)   (Roe average velocity)
   a_tilde = sqrt( ... )                                   (Roe average sound speed)
   S_L = min(u_L - a_L, u_tilde - a_tilde)
   S_R = max(u_R + a_R, u_tilde + a_tilde)
   ```

2. **Contact wave speed**:
   ```
   S_star = [p_R - p_L + ρ_L u_L (S_L - u_L) - ρ_R u_R (S_R - u_R)]
          / [ρ_L (S_L - u_L) - ρ_R (S_R - u_R)]
   ```

3. **Star-state conservative variables** (for K = L, R):
   ```
   U_star_K = ρ_K * (S_K - u_K) / (S_K - S_star) *
              [ 1, S_star, E_K + (S_star - u_K)(S_star + p_K / (ρ_K (S_K - u_K))) ]^T
   ```

4. **Flux selection**:
   ```
   if S_L >= 0:         F_HLLC = F(W_L)
   elseif S_L < 0 ≤ S_star:  F_HLLC = F(W_L) + S_L * (U_star_L - U(W_L))
   elseif S_star < 0 ≤ S_R:  F_HLLC = F(W_R) + S_R * (U_star_R - U(W_R))
   else:                 F_HLLC = F(W_R)
   ```

### 6.2 Implementation Notes

- Use Einfeldt's estimates for `S_L`, `S_R` rather than Davis's simpler (but more diffusive) `min(u_L-a_L, u_R-a_R)`.
- `S_star` formula comes from Batten et al. (1997), Eq. (21).
- Internal helpers are prefixed with underscore: `_wave_speeds_HLLC`, `_hllc_flux`, `_star_state_HLLC`.

---

## 7. Roe Solver (`riemann_solvers/Roe.jl`)

### 7.1 Algorithm

1. **Roe average state** `W_tilde = (ρ_tilde, u_tilde, H_tilde)`:
   ```
   R = sqrt(ρ_R / ρ_L)
   ρ_tilde = R * ρ_L
   u_tilde = (R * u_R + u_L) / (R + 1)
   H_tilde = (R * H_R + H_L) / (R + 1)    # total enthalpy H = (ρE + p) / ρ
   a_tilde = sqrt((γ - 1) * (H_tilde - 0.5 * u_tilde²))
   ```

2. **Eigenvalues** of the Roe matrix:
   ```
   λ₁ = u_tilde - a_tilde
   λ₂ = u_tilde
   λ₃ = u_tilde + a_tilde
   ```

3. **Right eigenvectors** `r₁, r₂, r₃` (columns of the eigenvector matrix):
   ```
   r₁ = (1, u_tilde - a_tilde, H_tilde - u_tilde*a_tilde)^T
   r₂ = (1, u_tilde, 0.5*u_tilde²)^T
   r₃ = (1, u_tilde + a_tilde, H_tilde + u_tilde*a_tilde)^T
   ```

4. **Wave strengths** `α₁, α₂, α₃` from `ΔU = U_R - U_L`:
   ```
   α₁ = (Δρ - Δp/a_tilde²) / 2
   α₂ = Δρ - α₁ - α₃       # or directly: Δρ - Δp/a_tilde²
   α₃ = (Δρ + Δp/a_tilde²) / 2
   ```

5. **Harten-Hyman entropy fix** on each eigenvalue:
   ```
   δ = 0.1 * a_tilde                # default threshold
   if |λ| < δ:
       |λ| ← (λ² + δ²) / (2δ)
   else:
       |λ| ← |λ|
   ```

6. **Roe flux**:
   ```
   F_Roe = ½ [F(W_L) + F(W_R)] - ½ Σ_{k=1}^{3} α_k |λ_k| r_k
   ```

### 7.2 Implementation Notes

- The entropy fix is applied element-wise on each of the three eigenvalues.
- Internal helpers: `_roe_average`, `_roe_eigenvalues`, `_roe_right_eigenvectors`, `_roe_wave_strengths`, `_entropy_fix`.

---

## 8. Slope Limiters (`limiters.jl`)

### 8.1 Abstract Type

```julia
abstract type AbstractLimiter end
```

### 8.2 Interface

```julia
"""
    limiter_value(lim::AbstractLimiter, r::Real) -> Real

Compute the limiter function `φ(r)`.

# Arguments
- `r`: slope ratio, `r = (u_i - u_{i-1}) / (u_{i+1} - u_i)`.
"""
function limiter_value end
```

### 8.3 Implemented Limiters

| Struct | `φ(r)` formula | Symmetry |
|--------|---------------|----------|
| `MinMod` | max(0, min(1, r)) | symmetric |
| `VanLeer` | (r + |r|) / (1 + |r|) | symmetric |
| `SuperBee` | max(0, min(2r, 1), min(r, 2)) | symmetric |
| `MC` | max(0, min(2r, 0.5*(1+r), 2)) | symmetric |
| `VanAlbada` | (r² + r) / (1 + r²) | symmetric |

All return 0 for `r ≤ 0` (extrema detection — slope set to zero at discontinuity).

---

## 9. MUSCL Reconstruction (`reconstruction.jl`)

### 9.1 Core Function

```julia
"""
    reconstruct(lim::AbstractLimiter, u_im1::T, u_i::T, u_ip1::T) -> (u_L, u_R)

Perform MUSCL reconstruction for cell `i` given its value and its two neighbours.
Returns the reconstructed values at the left and right faces of cell `i`.

# Formulae
    Δ_im1 = u_i - u_im1
    Δ_ip1 = u_ip1 - u_i
    r_i = Δ_im1 / Δ_ip1  (with guard for denominator = 0)
    φ_i = limiter_value(lim, r_i)
    Δ_i = φ_i * (u_i - u_im1)  [upwind-biased] or ½ φ_i * (u_ip1 - u_im1) [centred]

    u_L (right side of left face) = u_i - ½ φ_i * (u_ip1 - u_im1)
    u_R (left side of right face)  = u_i + ½ φ_i * (u_ip1 - u_im1)
""";
function reconstruct end
```

### 9.2 Field-Wise Reconstruction

```julia
"""
    muscl_reconstruct_all(W::Vector{PrimitiveState{T}}, lim::AbstractLimiter, grid::UniformGrid1D{T};
                          variables::Symbol=:primitive) -> (W_face_L::Vector{PrimitiveState{T}}, W_face_R::Vector{PrimitiveState{T}})

Reconstruct face values for the entire domain.
Returns two `Vector{PrimitiveState}` of length `N+1` (one per face).
`W_face_L[i]` is the state immediately left of face `i`, `W_face_R[i]` is right of face `i`.

The `variables` keyword selects:
- `:primitive` — reconstruct `ρ, u, p` individually (default, simplest, adequate)
- `:conservative` — reconstruct `ρ, ρu, ρE`
- `:characteristic` — (future) reconstruct in characteristic variables
"""
function muscl_reconstruct_all end
```

### 9.3 Boundary Handling

Ghost cells for transmissive BC (`W[0] = W[1]`, `W[N+1] = W[N]`) are populated **before** reconstruction, so the limiter at boundary cells uses `r = 1` → `φ = 1` (no limiting at physical boundaries).

---

## 10. Grid & Boundary Conditions (`grid.jl`)

### 10.1 Constructor

```julia
"""
    UniformGrid1D(xmin::Real, xmax::Real, N::Integer)

Create a uniform 1D grid with `N` cells spanning `[xmin, xmax]`.
"""
function UniformGrid1D(xmin::T, xmax::T, N::Integer) where {T}
    Δx = (xmax - xmin) / N
    xc = [xmin + (i - 0.5) * Δx for i in 1:N]
    xf = [xmin + (i - 1) * Δx for i in 1:(N+1)]
    return UniformGrid1D(promote(xmin, xmax, Δx)..., N, Δx, xc, xf)
end
```

### 10.2 Transmissive Boundary Conditions

```julia
"""
    apply_transmissive_bc!(W::Vector{PrimitiveState{T}})

Apply transmissive (zero-gradient) boundary conditions in-place.
Sets `W[0] = W[1]` and `W[N+1] = W[N]` on the ghost cells.
"""
function apply_transmissive_bc! end
```

Ghost cells are stored as `W[0]` and `W[N+1]` in a zero-indexed vector, or
handled in-line during the reconstruction and flux loops.

---

## 11. Time Stepping (`time_stepping.jl`)

### 11.1 CFL Time Step

```julia
"""
    compute_cfl_dt(W::Vector{PrimitiveState{T}}, eos::PerfectGasEOS, grid::UniformGrid1D{T}, cfl::Real) -> T

Compute the maximum stable time step from the CFL condition:
    Δt = CFL * min_i(Δx / (|u_i| + a_i))
"""
function compute_cfl_dt end
```

### 11.2 Forward Euler

```julia
"""
    forward_euler_step!(U::Vector{ConservativeState{T}}, U_new::Vector{ConservativeState{T}},
                        flux::Vector{Flux{T}}, grid::UniformGrid1D{T}, Δt::T)

Update `U_new = U - (Δt / Δx) * ΔF` where `ΔF_i = flux[i+1] - flux[i]`.
"""
function forward_euler_step! end
```

### 11.3 Main Time Loop

```julia
"""
    evolve!(U::Vector{ConservativeState{T}}, grid::UniformGrid1D{T},
            eos::PerfectGasEOS, config::SolverConfig)

Run the full finite-volume evolution to `config.max_time`.
"""
function evolve! end
```

The loop body per step:
1. `W = conservative_to_primitive.(U, eos)`
2. `apply_transmissive_bc!(W)`
3. `W_L, W_R = muscl_reconstruct_all(W, config.limiter, grid)` (skip if `!config.use_muscl`)
4. `flux[i] = riemann_flux(config.solver, W_R[i-1], W_L[i], eos)` for `i in 1:N+1`
5. `Δt = compute_cfl_dt(W, eos, grid, config.cfl)`
6. `forward_euler_step!(U, U_new, flux, grid, Δt)`
7. `U .= U_new`

---

## 12. Exports (`Riemann1D.jl`)

```julia
# Existing (unchanged)
export AbstractState, PrimitiveState, AbstractEOS, PerfectGasEOS,
       NonlinearWaveStructure, AbstractRiemannSolution, ExactRiemannSolution,
       PressureGuessMethod, PV, TR, TS,
       solve_Riemann_problem, sample_solution

# New types
export ConservativeState, Flux, UniformGrid1D, SolverConfig

# EOS utilities
export sound_speed, internal_energy, total_energy, total_enthalpy, euler_flux,
       primitive_to_conservative, conservative_to_primitive

# Riemann solvers
export AbstractRiemannSolver, riemann_flux,
       HLLCSolver, RoeSolver, ExactSolver, GodunovSolver

# Limiters
export AbstractLimiter, limiter_value,
       MinMod, VanLeer, SuperBee, MC, VanAlbada

# Reconstruction
export reconstruct, muscl_reconstruct_all

# Grid & BC
export apply_transmissive_bc!

# Time stepping
export compute_cfl_dt, forward_euler_step!, evolve!
```

---

## 13. Dependencies

No new dependencies beyond the existing `ForwardDiff.jl`.

If `StaticArrays.jl` is desired for `ConservativeState` and `Flux` performance (recommended but optional), add:
```toml
[deps]
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
StaticArrays = "90137ffa-7385-5640-81b9-520cb181641f"
```

Decision: **do not add StaticArrays** for now. Use standard `struct` with manual arithmetic overloads if needed. Keeps the dependency footprint minimal.

---

## 14. Testing Strategy (Future)

When tests are implemented, they should cover:

| Test | What it verifies |
|------|-----------------|
| `test_conservation` | `riemann_flux` is consistent: `F_HLLC`, `F_Roe`, `F_Exact` all match for uniform flow |
| `test_sod_exact` | Exact Riemann solver reproduces analytical Sod solution |
| `test_sod_hllc` | HLLC mesh convergence on Sod problem |
| `test_sod_roe` | Roe mesh convergence on Sod problem (watch for entropy fix on sonic point) |
| `test_limiter_symmetry` | Each limiter satisfies `φ(r)/r = φ(1/r)` |
| `test_reconstruction_constant` | MUSCL recovers constant field exactly |
| `test_reconstruction_linear` | MUSCL recovers linear field exactly (no limiting at smooth extrema) |
| `test_cfl_stability` | Forward Euler blows up for CFL > 1, stable for CFL < 1 |

Model problems:
- **Sod shock tube:** `(ρ,u,p)_L = (1,0,1)`, `(ρ,u,p)_R = (0.125,0,0.1)`, `γ = 1.4`
- **Lax shock tube:** `(ρ,u,p)_L = (0.445,0.698,3.528)`, `(ρ,u,p)_R = (0.5,0,0.571)`, `γ = 1.4`
- **Shu-Osher entropy wave:** `(ρ,u,p)_L = (3.857143,2.629369,10.33333)`, `(ρ,u,p)_R = (1+0.2sin(5x),0,1)`

---

## 15. References

1. **Toro, E.F.** *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed., Springer, 2009.
   - §10.4: HLLC approximate Riemann solver
   - §11.3: The Roe approximate Riemann solver
   - §13.5: MUSCL-Hancock scheme
2. **Batten, P., Clarke, N., Lambert, C., Causon, D.M.** "On the Choice of Wavespeeds for the HLLC Riemann Solver," *SIAM J. Sci. Comput.*, 18(6), 1997.
3. **Harten, A., Hyman, J.M.** "Self-Adjusting Grid Methods for One-Dimensional Hyperbolic Conservation Laws," *J. Comput. Phys.*, 50, 1983.
4. **LeVeque, R.J.** *Finite Volume Methods for Hyperbolic Problems*, Cambridge, 2002.
