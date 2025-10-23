using Test
using Moca
using Moca: step  # Explicitly import step to resolve ambiguity with Base.step
using Statistics  # For mean, std functions used in tests

@testset "Moca.jl tests" begin
    include("test_rng.jl")
    include("test_processes.jl")
    include("test_sampling.jl")
    include("test_paths.jl")
    include("test_statistics.jl")
    include("test_simulation.jl")
end