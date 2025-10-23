using Statistics: mean, std
using Moca: variance, std_dev, std_err, confidence_interval
import Moca: finalize  # Use import to avoid conflict with Base.finalize

@testset "Statistics tests" begin
    @testset "Online statistics initialization" begin
        stats = OnlineStatistics()
        @test stats.n == 0
        @test stats.mean == 0.0
        @test stats.squared_diffs == 0.0
    end

    @testset "Welford algorithm correctness" begin
        stats = OnlineStatistics()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        for v in values
            update!(stats, v)
        end

        @test stats.n == 5
        @test stats.mean ≈ 3.0
        @test variance(stats) ≈ 2.5  # Sample variance with Bessel's correction
        @test std_dev(stats) ≈ sqrt(2.5)
    end

    @testset "Online statistics vs batch statistics" begin
        # Generate random data
        rng = StandardRNG(42)
        values = [rand_normal(rng) for _ in 1:1000]

        # Compute using OnlineStatistics
        stats = OnlineStatistics()
        for v in values
            update!(stats, v)
        end

        # Compute using Julia's built-in functions
        batch_mean = mean(values)
        batch_std = std(values)

        @test stats.mean ≈ batch_mean
        @test std_dev(stats) ≈ batch_std
    end

    @testset "Standard error computation" begin
        stats = OnlineStatistics()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        for v in values
            update!(stats, v)
        end

        # stderr = std / sqrt(n)
        expected_stderr = sqrt(2.5) / sqrt(5)
        @test std_err(stats) ≈ expected_stderr
    end

    @testset "Confidence interval" begin
        stats = OnlineStatistics()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        for v in values
            update!(stats, v)
        end

        ci = confidence_interval(stats; confidence_level=0.95)

        # Check that CI is a tuple
        @test ci isa Tuple{Float64,Float64}
        # Check that lower < mean < upper
        @test ci[1] < stats.mean < ci[2]
        # Check symmetry around mean
        @test abs(ci[1] - stats.mean) ≈ abs(ci[2] - stats.mean)
    end

    @testset "Chan's parallel aggregation algorithm" begin
        # Create two separate statistics accumulators
        stats1 = OnlineStatistics()
        stats2 = OnlineStatistics()

        values1 = [1.0, 2.0, 3.0]
        values2 = [4.0, 5.0, 6.0]

        for v in values1
            update!(stats1, v)
        end
        for v in values2
            update!(stats2, v)
        end

        # Aggregate
        combined = aggregate([stats1, stats2])

        # Should match statistics computed on all values
        all_values = [values1; values2]
        @test combined.mean ≈ mean(all_values)
        @test variance(combined) ≈ var(all_values; corrected=true)
    end

    @testset "Finalize converts to Monte Carlo estimate" begin
        stats = OnlineStatistics()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        for v in values
            update!(stats, v)
        end

        result = finalize(stats)

        @test result isa MonteCarloEstimate
        @test result.mean == stats.mean
        @test result.std_dev ≈ std_dev(stats)
        @test result.std_err ≈ std_err(stats)
        @test result.n_samples == 5
    end

    @testset "Aggregate empty list throws error" begin
        @test_throws ErrorException aggregate(OnlineStatistics[])
    end

    @testset "Aggregate single statistics returns itself" begin
        stats = OnlineStatistics()
        update!(stats, 5.0)
        update!(stats, 10.0)

        result = aggregate([stats])
        @test result.n == stats.n
        @test result.mean == stats.mean
        @test result.squared_diffs == stats.squared_diffs
    end
end
