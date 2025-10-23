module Moca

include("rng.jl")
export RNGStrategy, StandardRNG, split_rng, rand_normal, rand_uniform

include("processes.jl")
export StochasticProcess, WienerProcess, GeometricBrownianMotion, OrnsteinUhlenbeckProcess, step

include("sampling.jl")
export SamplingStrategy, StandardSampling, AntitheticSampling

include("paths.jl")
export PathConfig, generate_path

include("statistics.jl")
export MonteCarloEstimate, OnlineStatistics, update!, finalize, aggregate

include("simulation.jl")
export simulate

end