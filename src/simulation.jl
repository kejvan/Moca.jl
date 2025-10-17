"""
    simulate(process, functional, n_paths; initial_state=0.0, config, rng_seed=42)

Run Monte Carlo simulation by generating n_paths trajectories and computing statistics.
"""
function simulate(
    process::StochasticProcess,
    functional::Function,
    n_paths::Int;
    initial_state::Float64=0.0,
    config::PathConfig,
    rng_seed::Int=42
)::MonteCarloEstimate
    stats = OnlineStatistics()
    rng = StandardRNG(rng_seed)
    rng_list = split_rng(rng, n_paths)
    for rng in rng_list
        path_or_value = generate_path(process, initial_state, config, rng)
        result = functional(path_or_value)
        update!(stats, result)
    end
    return finalize(stats)
end