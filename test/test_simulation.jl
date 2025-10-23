@testset "Simulation integration tests" begin
    @testset "Basic simulation returns Monte Carlo estimate" begin
        process = WienerProcess(drift=0.1, volatility=0.2)
        config = PathConfig(n_steps=100, dt=0.01, store_path=false)

        result = simulate(
            process,
            x -> x,  # Identity functional
            1000,
            config;
            initial_state=1.0,
            sampling=StandardSampling(),
            parallel=:serial,
            rng_seed=42
        )

        @test result isa MonteCarloEstimate
        @test result.n_samples == 1000
        @test isfinite(result.mean)
        @test isfinite(result.std_dev)
        @test result.std_err > 0
    end

    @testset "Serial vs parallel execution consistency" begin
        process = WienerProcess(drift=0.1, volatility=0.2)
        config = PathConfig(n_steps=100, dt=0.01, store_path=false)

        result_serial = simulate(
            process,
            x -> x,
            10000,
            config;
            initial_state=1.0,
            sampling=StandardSampling(),
            parallel=:serial,
            rng_seed=42
        )

        result_parallel = simulate(
            process,
            x -> x,
            10000,
            config;
            initial_state=1.0,
            sampling=StandardSampling(),
            parallel=:threads,
            rng_seed=42
        )

        # Results should be within 3 standard errors of each other
        diff = abs(result_serial.mean - result_parallel.mean)
        combined_stderr = sqrt(result_serial.std_err^2 + result_parallel.std_err^2)
        @test diff < 3 * combined_stderr
    end

    @testset "Wiener process analytical validation via simulate" begin
        # E[X_T] = x0 + drift*T
        process = WienerProcess(drift=0.1, volatility=0.2)
        x0 = 1.0
        T = 1.0
        config = PathConfig(n_steps=100, dt=T / 100, store_path=false)

        result = simulate(
            process,
            x -> x,
            10000,
            config;
            initial_state=x0,
            sampling=StandardSampling(),
            parallel=:serial,
            rng_seed=42
        )

        expected_mean = x0 + 0.1 * T
        @test abs(result.mean - expected_mean) < 3 * result.std_err
    end

    @testset "GBM analytical validation via simulate" begin
        # E[X_T] = x0 * exp(drift*T)
        process = GeometricBrownianMotion(drift=0.05, volatility=0.2)
        x0 = 1.0
        T = 1.0
        config = PathConfig(n_steps=100, dt=T / 100, store_path=false)

        result = simulate(
            process,
            x -> x,
            10000,
            config;
            initial_state=x0,
            sampling=StandardSampling(),
            parallel=:serial,
            rng_seed=123
        )

        expected_mean = x0 * exp(0.05 * T)
        @test abs(result.mean - expected_mean) < 3 * result.std_err
    end

    @testset "Nonlinear functional" begin
        # Test E[X_T^2]
        process = WienerProcess(drift=0.0, volatility=1.0)
        x0 = 0.0
        T = 1.0
        config = PathConfig(n_steps=100, dt=T / 100, store_path=false)

        result = simulate(
            process,
            x -> x^2,
            10000,
            config;
            initial_state=x0,
            sampling=StandardSampling(),
            parallel=:serial,
            rng_seed=456
        )

        # E[X_T^2] = Var[X_T] = sigma^2 * T = 1.0 * 1.0 = 1.0
        expected_mean = 1.0
        @test abs(result.mean - expected_mean) < 3 * result.std_err
    end

    @testset "Path-dependent functional" begin
        # Test maximum of path
        process = WienerProcess(drift=0.0, volatility=0.2)
        config = PathConfig(n_steps=100, dt=0.01, store_path=true)

        result = simulate(
            process,
            path -> maximum(path),
            1000,
            config;
            initial_state=1.0,
            sampling=StandardSampling(),
            parallel=:serial,
            rng_seed=789
        )

        # Maximum should be >= initial value
        @test result.mean >= 1.0
    end

    @testset "Antithetic sampling reduces variance" begin
        process = WienerProcess(drift=0.0, volatility=1.0)
        config = PathConfig(n_steps=100, dt=0.01, store_path=false)

        result_standard = simulate(
            process,
            x -> x,
            10000,
            config;
            initial_state=0.0,
            sampling=StandardSampling(),
            parallel=:serial,
            rng_seed=42
        )

        result_antithetic = simulate(
            process,
            x -> x,
            10000,
            config;
            initial_state=0.0,
            sampling=AntitheticSampling(),
            parallel=:serial,
            rng_seed=42
        )

        # Antithetic should have lower standard error
        @test result_antithetic.std_err < result_standard.std_err
    end

    @testset "Confidence intervals contain true mean" begin
        # For Wiener process, we know the exact mean
        process = WienerProcess(drift=0.1, volatility=0.2)
        x0 = 1.0
        T = 1.0
        config = PathConfig(n_steps=100, dt=T / 100, store_path=false)

        # Run multiple times and check coverage
        true_mean = x0 + 0.1 * T
        coverage_count = 0
        n_trials = 100

        for seed in 1:n_trials
            result = simulate(
                process,
                x -> x,
                1000,
                config;
                initial_state=x0,
                sampling=StandardSampling(),
                parallel=:serial,
                rng_seed=seed
            )

            ci_lower, ci_upper = result.confidence_interval
            if ci_lower <= true_mean <= ci_upper
                coverage_count += 1
            end
        end

        # 95% CI should contain true mean about 95% of the time
        # Allow some slack (e.g., 90-100% coverage)
        coverage_rate = coverage_count / n_trials
        @test coverage_rate > 0.90
    end

    @testset "Edge case: single path" begin
        process = WienerProcess(drift=0.0, volatility=1.0)
        config = PathConfig(n_steps=10, dt=0.1, store_path=false)

        result = simulate(
            process,
            x -> x,
            1,
            config;
            initial_state=0.0,
            sampling=StandardSampling(),
            parallel=:serial,
            rng_seed=42
        )

        @test result.n_samples == 1
        @test isfinite(result.mean)
    end

    @testset "Edge case: single time step" begin
        process = WienerProcess(drift=0.0, volatility=1.0)
        config = PathConfig(n_steps=1, dt=1.0, store_path=false)

        result = simulate(
            process,
            x -> x,
            100,
            config;
            initial_state=0.0,
            sampling=StandardSampling(),
            parallel=:serial,
            rng_seed=42
        )

        @test result.n_samples == 100
        @test isfinite(result.mean)
    end
end
