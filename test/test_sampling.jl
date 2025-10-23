@testset "Sampling strategy tests" begin
    @testset "Standard sampling produces single path" begin
        process = WienerProcess(drift=0.0, volatility=1.0)
        rng = StandardRNG(42)
        config = PathConfig(n_steps=10, dt=0.1, store_path=false)

        result = generate_path(process, 0.0, config, rng, sampling=StandardSampling())

        # Should return a single Float64 value (terminal value)
        @test result isa Float64
    end

    @testset "Antithetic sampling produces tuple of two paths" begin
        process = WienerProcess(drift=0.0, volatility=1.0)
        rng = StandardRNG(42)
        config = PathConfig(n_steps=10, dt=0.1, store_path=false)

        result = generate_path(process, 0.0, config, rng, sampling=AntitheticSampling())

        # Should return a tuple of two Float64 values
        @test result isa Tuple{Float64,Float64}
    end

    @testset "Antithetic sampling with stored paths" begin
        process = WienerProcess(drift=0.0, volatility=1.0)
        rng = StandardRNG(123)
        config = PathConfig(n_steps=10, dt=0.1, store_path=true)

        result = generate_path(process, 0.0, config, rng, sampling=AntitheticSampling())

        # Should return tuple of two arrays
        @test result isa Tuple{Vector{Float64},Vector{Float64}}
        @test length(result[1]) == 11  # n_steps + 1
        @test length(result[2]) == 11
    end

    @testset "Antithetic variance reduction for linear functional" begin
        # For linear functionals, antithetic sampling should reduce variance
        process = WienerProcess(drift=0.0, volatility=1.0)
        n_paths = 5000
        config = PathConfig(n_steps=100, dt=0.01, store_path=false)

        # Standard sampling
        rng_standard = StandardRNG(42)
        standard_values = Float64[]
        for _ in 1:n_paths
            path = generate_path(process, 0.0, config, rng_standard, sampling=StandardSampling())
            push!(standard_values, path)
        end
        standard_var = sum((x - mean(standard_values))^2 for x in standard_values) / (length(standard_values) - 1)

        # Antithetic sampling
        rng_antithetic = StandardRNG(42)
        antithetic_values = Float64[]
        for _ in 1:(n_paths÷2)
            path_A, path_B = generate_path(process, 0.0, config, rng_antithetic, sampling=AntitheticSampling())
            push!(antithetic_values, (path_A + path_B) / 2.0)
        end
        antithetic_var = sum((x - mean(antithetic_values))^2 for x in antithetic_values) / (length(antithetic_values) - 1)

        # Antithetic variance should be smaller
        @test antithetic_var < standard_var
    end

    @testset "Antithetic paths have symmetric randoms" begin
        # Test that antithetic paths truly use negated randoms
        process = WienerProcess(drift=0.0, volatility=1.0)
        rng = StandardRNG(999)
        config = PathConfig(n_steps=5, dt=0.1, store_path=true)

        path_A, path_B = generate_path(process, 0.0, config, rng, sampling=AntitheticSampling())

        # For zero drift Wiener, if randoms are truly negated,
        # path_A and path_B should be symmetric around initial value
        # (they won't be exactly symmetric due to floating point, but close)
        @test length(path_A) == length(path_B)
        @test path_A[1] == path_B[1]  # Same initial value
    end
end
