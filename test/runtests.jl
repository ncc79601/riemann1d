using Riemann1D
using Test
using Aqua
using JET

@testset "Riemann1D" begin
    @testset "module loads and exports" begin
        @test isdefined(Riemann1D, :PrimitiveState)
        @test isdefined(Riemann1D, :ConservedState)
        @test isdefined(Riemann1D, :PerfectGasEOS)
        @test isdefined(Riemann1D, :sound_speed)
        @test isdefined(Riemann1D, :conserved_to_primitive)
        @test isdefined(Riemann1D, :primitive_to_conserved)
    end

    @testset "constructors" begin
        W = PrimitiveState(1.0, 0.0, 1.0)
        @test W isa AbstractState
        @test W.ρ == 1.0
        @test W.u == 0.0
        @test W.p == 1.0

        U = ConservedState(1.0, 0.0, 2.5)
        @test U isa AbstractState
        @test U.ρ == 1.0
        @test U.ρu == 0.0
        @test U.E == 2.5

        eos = PerfectGasEOS(1.4)
        @test eos isa AbstractEOS
        @test eos.γ == 1.4

        @test_throws DomainError PerfectGasEOS(0.5)
    end

    @testset "sound_speed" begin
        eos = PerfectGasEOS(1.4)
        W = PrimitiveState(1.4, 0.0, 1.0)
        @test sound_speed(W, eos) ≈ 1.0
    end

    @testset "conversion round-trip" begin
        eos = PerfectGasEOS(1.4)
        W = PrimitiveState(1.5, 2.0, 3.0)
        U = primitive_to_conserved(W, eos)
        W_back = conserved_to_primitive(U, eos)
        @test W_back.ρ ≈ W.ρ
        @test W_back.u ≈ W.u
        @test W_back.p ≈ W.p
    end

    @testset "arithmetic" begin
        U1 = ConservedState(1.0, 0.5, 2.0)
        U2 = ConservedState(2.0, 1.0, 3.0)

        # +
        U3 = U1 + U2
        @test U3.ρ == 3.0
        @test U3.ρu == 1.5
        @test U3.E == 5.0

        # -
        U4 = U2 - U1
        @test U4.ρ == 1.0
        @test U4.ρu == 0.5
        @test U4.E == 1.0

        # scalar *
        U5 = 2.0 * U1
        @test U5.ρ == 2.0
        @test U5.ρu == 1.0
        @test U5.E == 4.0
    end

    @testset "Sod problem — consistency check" begin
        W_L = PrimitiveState(1.0, 0.0, 1.0)
        W_R = PrimitiveState(0.125, 0.0, 0.1)
        eos = PerfectGasEOS(1.4)

        # Exact solver runs without error
        sol = solve_Riemann_problem_exact(W_L, W_R, eos)
        @test sol isa ExactRiemannSolution
        @test sol.p★ > 0.0
        @test sol.u★ > 0.0

        # Sample solution at x=0 gives sensible density
        state = sample_exact_solution(0.0, 0.2, sol)
        @test state.ρ > 0.0
        @test state.p > 0.0
    end

    @testset "Sod problem — uniform flow (all solvers agree)" begin
        W = PrimitiveState(1.0, 0.5, 1.0)
        eos = PerfectGasEOS(1.4)
        F_ref = Flux(W, eos)

        solvers = [
            GodunovSolver(),
            PVRS(),
            TRRS(),
            TSRS(),
            AIRS(),
            ANRS(),
            HLLC(),
            RoeSolver()
        ]

        for solver in solvers
            F_num = compute_numerical_flux(solver, W, W, eos)
            @test F_num.mass ≈ F_ref.mass
            @test F_num.momentum ≈ F_ref.momentum
            @test F_num.energy ≈ F_ref.energy
        end
    end

    @testset "end-to-end: Sod shock tube (1st order Godunov)" begin
        grid = UniformGrid1D(-0.5, 0.5, 100; ghost_cells = 2)
        problem = SodProblem()
        eos = PerfectGasEOS(1.4)

        U = init_simulation(problem, grid, eos)
        config = SolverConfig(GodunovSolver(), 0.9, 0.25, 1000;
            reconstruction = NoReconstruct(),
            limiter = NoLimiter(),
            integrator = ExplicitEuler(),
            init_steps = 5,
            init_cfl = 0.2
        )
        _, n_steps, _ = run_simulation!(U, grid, eos, config)
        W = [conserved_to_primitive(U[i], eos) for i in 1:(grid.N)]
        @test n_steps > 0
        @test all(w -> w.ρ > 0, W)
        @test all(w -> w.p > 0, W)
    end

    @testset "end-to-end: Sod shock tube (HLLC + vanLeer + TVDRK2)" begin
        grid = UniformGrid1D(-0.5, 0.5, 100; ghost_cells = 2)
        problem = SodProblem()
        eos = PerfectGasEOS(1.4)

        U = init_simulation(problem, grid, eos)
        config = SolverConfig(HLLC(), 0.4, 0.25, 1000;
            reconstruction = SecondOrderReconstruct(),
            limiter = vanLeerLimiter(),
            integrator = TVDRK2(),
            init_steps = 5,
            init_cfl = 0.2
        )
        _, n_steps, _ = run_simulation!(U, grid, eos, config)
        W = [conserved_to_primitive(U[i], eos) for i in 1:(grid.N)]
        @test n_steps > 0
        @test all(w -> w.ρ > 0, W)
        @test all(w -> w.p > 0, W)
    end

    @testset "Aqua check" begin
        Aqua.test_all(
            Riemann1D;
            ambiguities = false,
            piracies = true,
            stale_deps = true,
            unbound_args = true,
            undefined_exports = true,
            project_extras = true
        )
    end

    @testset "JET check" begin
        JET.test_package(Riemann1D; target_modules = (Riemann1D,))
    end
end
