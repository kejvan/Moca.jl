@testset "Process tests" begin
    @testset "Wiener process step" begin
        process = WienerProcess(drift=0.1, volatility=0.2)
        x = 1.0
        dt = 0.01
        Z = 0.5  # Standard normal random variable

        # Euler-Maruyama: x_new = x + drift*dt + volatility*sqrt(dt)*Z
        x_new = step(process, x, dt, Z)
        expected = x + 0.1 * dt + 0.2 * sqrt(dt) * Z

        @test x_new ≈ expected
    end

    @testset "Geometric Brownian motion step" begin
        process = GeometricBrownianMotion(drift=0.05, volatility=0.3)
        x = 2.0
        dt = 0.01
        Z = 0.5  # Standard normal random variable

        # GBM: x_new = x + drift*x*dt + volatility*x*sqrt(dt)*Z
        x_new = step(process, x, dt, Z)
        expected = x + 0.05 * x * dt + 0.3 * x * sqrt(dt) * Z

        @test x_new ≈ expected
    end

    @testset "Wiener process analytical validation" begin
        # E[X_T] = x0 + drift*T
        # Var[X_T] = volatility^2 * T
        process = WienerProcess(drift=0.1, volatility=0.2)
        x0 = 1.0
        T = 1.0
        n_paths = 10000

        rng = StandardRNG(42)
        config = PathConfig(n_steps=100, dt=T / 100, store_path=false)

        terminal_values = Float64[]
        for _ in 1:n_paths
            path = generate_path(process, x0, config, rng, sampling=StandardSampling())
            push!(terminal_values, path)
        end

        # Check mean
        sample_mean = sum(terminal_values) / length(terminal_values)
        expected_mean = x0 + 0.1 * T
        @test abs(sample_mean - expected_mean) < 0.01

        # Check variance
        sample_var = sum((x - sample_mean)^2 for x in terminal_values) / (length(terminal_values) - 1)
        expected_var = 0.2^2 * T
        @test abs(sample_var - expected_var) < 0.01
    end

    @testset "Geometric Brownian motion analytical validation" begin
        # E[X_T] = x0 * exp(drift*T)
        process = GeometricBrownianMotion(drift=0.05, volatility=0.2)
        x0 = 1.0
        T = 1.0
        n_paths = 10000

        rng = StandardRNG(123)
        config = PathConfig(n_steps=100, dt=T / 100, store_path=false)

        terminal_values = Float64[]
        for _ in 1:n_paths
            path = generate_path(process, x0, config, rng, sampling=StandardSampling())
            push!(terminal_values, path)
        end

        # Check mean
        sample_mean = sum(terminal_values) / length(terminal_values)
        expected_mean = x0 * exp(0.05 * T)
        @test abs(sample_mean - expected_mean) / expected_mean < 0.02  # Within 2%
    end
end
