@testset "Path generation tests" begin
    @testset "Streaming mode returns terminal value" begin
        process = WienerProcess(drift=0.1, volatility=0.2)
        rng = StandardRNG(42)
        config = PathConfig(n_steps=100, dt=0.01, store_path=false)

        result = generate_path(process, 1.0, config, rng, sampling=StandardSampling())

        @test result isa Float64
        @test !isnan(result)
        @test isfinite(result)
    end

    @testset "Stored mode returns full path" begin
        process = WienerProcess(drift=0.1, volatility=0.2)
        rng = StandardRNG(42)
        config = PathConfig(n_steps=100, dt=0.01, store_path=true)

        result = generate_path(process, 1.0, config, rng, sampling=StandardSampling())

        @test result isa Vector{Float64}
        @test length(result) == 101  # n_steps + 1 (including initial value)
        @test result[1] == 1.0  # Initial value
    end

    @testset "Path starts at initial state" begin
        process = WienerProcess(drift=0.0, volatility=1.0)
        rng = StandardRNG(123)
        config = PathConfig(n_steps=10, dt=0.1, store_path=true)

        x0 = 5.0
        path = generate_path(process, x0, config, rng, sampling=StandardSampling())

        @test path[1] == x0
    end

    @testset "Path length matches configuration" begin
        process = GeometricBrownianMotion(drift=0.05, volatility=0.2)
        rng = StandardRNG(456)

        for n_steps in [10, 50, 100]
            config = PathConfig(n_steps=n_steps, dt=0.01, store_path=true)
            path = generate_path(process, 1.0, config, rng, sampling=StandardSampling())
            @test length(path) == n_steps + 1
        end
    end

    @testset "Different RNGs produce different paths" begin
        process = WienerProcess(drift=0.0, volatility=1.0)
        config = PathConfig(n_steps=100, dt=0.01, store_path=false)

        rng1 = StandardRNG(111)
        rng2 = StandardRNG(222)

        path1 = generate_path(process, 0.0, config, rng1, sampling=StandardSampling())
        path2 = generate_path(process, 0.0, config, rng2, sampling=StandardSampling())

        @test path1 != path2
    end

    @testset "Same RNG seed produces identical paths" begin
        process = WienerProcess(drift=0.1, volatility=0.2)
        config = PathConfig(n_steps=100, dt=0.01, store_path=true)

        rng1 = StandardRNG(42)
        rng2 = StandardRNG(42)

        path1 = generate_path(process, 1.0, config, rng1, sampling=StandardSampling())
        path2 = generate_path(process, 1.0, config, rng2, sampling=StandardSampling())

        @test path1 == path2
    end
end
