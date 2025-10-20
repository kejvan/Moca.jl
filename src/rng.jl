using Random

abstract type RNGStrategy end

"""
    StandardRNG <: RNGStrategy

Immutable RNG wrapper for thread-safe parallel execution.
"""
struct StandardRNG <: RNGStrategy
    rng::Random.AbstractRNG
    seed::Int
end

function StandardRNG(seed::Int)::StandardRNG
    rng = Random.MersenneTwister(seed)
    return StandardRNG(rng, seed)
end

"""
    split_rng(rng::StandardRNG, n::Int=2)

Create n independent RNG streams for parallel execution.
"""
function split_rng(rng::StandardRNG, n::Int=2)::Vector{StandardRNG}
    return map(1:n) do _
        rand(rng.rng, 1:typemax(Int)) |> StandardRNG
    end
end

"""
    rand_normal(rng::StandardRNG; mean=0.0, std_dev=1.0)

Sample from normal distribution with given mean and standard deviation.
"""
function rand_normal(
    rng::StandardRNG;
    mean::Float64=0.0,
    std_dev::Float64=1.0
)::Float64
    z = randn(rng.rng)
    return mean + std_dev * z
end

"""
    rand_uniform(rng::StandardRNG; left=0.0, right=1.0)

Sample from uniform distribution on [left, right].
"""
function rand_uniform(
    rng::StandardRNG;
    left::Float64=0.0,
    right::Float64=1.0
)::Float64
    z = rand(rng.rng)
    return left + (right - left) * z
end