@testset "RNG tests" begin
    @testset "Reproducability" begin
        rng1 = StandardRNG(42)
        rng2 = StandardRNG(42)

        samples1 = [rand_normal(rng1) for _ in 1:10]
        samples2 = [rand_normal(rng2) for _ in 1:10]

        @test samples1 == samples2
    end

    @testset "RNG splitting" begin
        parent_rng = StandardRNG(123)
        child_rngs = split_rng(parent_rng, 3)

        # Each child should produce different values
        samples = [rand_normal(child_rngs[i]) for i in 1:3]
        @test length(unique(samples)) == 3  # All different
    end

    @testset "Normal distribution" begin
        rng = StandardRNG(456)
        samples = [rand_normal(rng; mean=0.0, std_dev=1.0) for _ in 1:1000]

        # Check sample mean and std are close to theoretical values
        sample_mean = sum(samples) / length(samples)
        sample_std = sqrt(sum((x - sample_mean)^2 for x in samples) / (length(samples) - 1))

        @test abs(sample_mean) < 0.1  # Should be close to 0
        @test abs(sample_std - 1.0) < 0.1  # Should be close to 1
    end

    @testset "Uniform distribution" begin
        rng = StandardRNG(789)
        samples = [rand_uniform(rng; left=0.0, right=1.0) for _ in 1:1000]

        @test all(0.0 .<= samples .<= 1.0)

        # Check mean is close to 0.5
        sample_mean = sum(samples) / length(samples)
        @test abs(sample_mean - 0.5) < 0.05
    end
end