module Moca

include("rng.jl")
include("processes.jl")
include("paths.jl")
include("statistics.jl")
include("simulation.jl")

export RNGStrategy, StandardRNG, split_rng, rand_normal, rand_uniform
export StochasticProcess, WienerProcess, GeometricBrownianMotion, step
export PathConfig, generate_path
export MonteCarloEstimate, OnlineStatistics, update!, finalize, aggregate
export simulate

end