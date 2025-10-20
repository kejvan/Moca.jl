using Distributions

"""
    MonteCarloEstimate

Final Monte Carlo simulation result.
"""
struct MonteCarloEstimate
    mean::Float64
    std_dev::Float64
    std_err::Float64
    confidence_interval::Tuple{Float64,Float64}
    n_samples::Int
end

"""
    OnlineStatistics

Mutable accumulator for computing statistics incrementally.
"""
mutable struct OnlineStatistics
    n::Int
    mean::Float64
    squared_diffs::Float64  # For Welford's algorithm
end

function OnlineStatistics()
    return OnlineStatistics(0, 0.0, 0.0)
end

"""
    update!(stats::OnlineStatistics, value)

Update statistics with new sample using Welford's algorithm.
"""
function update!(stats::OnlineStatistics, value::Float64)::OnlineStatistics
    stats.n += 1
    delta_i = value - stats.mean
    stats.mean += delta_i / stats.n
    delta_f = value - stats.mean
    stats.squared_diffs += delta_i * delta_f
    return stats
end

function variance(stats::OnlineStatistics)::Float64
    if stats.n < 2
        return 0.0
    end
    return stats.squared_diffs / (stats.n - 1)  # Bessel's correction
end

function std_dev(stats::OnlineStatistics)::Float64
    return stats |> variance |> sqrt
end

function std_err(stats::OnlineStatistics)::Float64
    if stats.n < 2
        return Inf
    end
    return std_dev(stats) / sqrt(stats.n)
end

"""
    confidence_interval(stats; confidence_level=0.95)

Compute confidence interval using normal approximation.
"""
function confidence_interval(
    stats::OnlineStatistics;
    confidence_level=0.95
)::Tuple{Float64,Float64}
    if stats.n < 2
        return (-Inf, Inf)
    end
    alpha = (1 - confidence_level) / 2
    z_score = quantile(Normal(0, 1), 1 - alpha)

    margin = z_score * std_err(stats)
    return (stats.mean - margin, stats.mean + margin)
end

"""
    finalize(stats; confidence_level=0.95)

Convert accumulator to final MonteCarloEstimate.
"""
function finalize(
    stats::OnlineStatistics;
    confidence_level=0.95
)::MonteCarloEstimate
    estimate = MonteCarloEstimate(
        stats.mean,
        std_dev(stats),
        std_err(stats),
        confidence_interval(stats; confidence_level=confidence_level),
        stats.n
    )
    return estimate
end

"""
    aggregate(stats_list)

Combine statistics from multiple workers using Chan's parallel variance algorithm.
"""
function aggregate(stats_list::Vector{OnlineStatistics})::OnlineStatistics
    l = length(stats_list)
    l == 0 && error("Cannot aggregate empty statistics list")

    stats_b = stats_list[1]
    combined = OnlineStatistics(
        stats_b.n,
        stats_b.mean,
        stats_b.squared_diffs
    )

    for i = 2:l
        stats_b = stats_list[i]

        n_a = combined.n
        n_b = stats_b.n
        n_tot = n_a + n_b

        if n_tot == 0
            continue
        end

        delta = stats_b.mean - combined.mean
        combined.squared_diffs = combined.squared_diffs + stats_b.squared_diffs + delta^2 * n_a * n_b / n_tot

        combined.mean = (n_a * combined.mean + n_b * stats_b.mean) / n_tot

        combined.n = n_tot
    end

    return combined
end