"""
    simulate(process, functional, n_paths; initial_state=0.0, config, rng_seed=42, parallel=:serial)

Run Monte Carlo simulation by generating n_paths trajectories and computing statistics. Supports serial (:serial) and multi-threaded (:threads) execution.
"""
function simulate(
    process::StochasticProcess,
    functional::Function,
    n_paths::Int;
    initial_state::Float64=0.0,
    config::PathConfig,
    rng_seed::Int=42,
    parallel::Symbol=:serial
)::MonteCarloEstimate
    if parallel == :serial
        rng = StandardRNG(rng_seed)
        rng_list = split_rng(rng, n_paths)

        stats = OnlineStatistics()

        for rng in rng_list
            path_or_value = generate_path(process, initial_state, config, rng)
            result = functional(path_or_value)
            update!(stats, result)
        end

        return finalize(stats)

    elseif parallel == :threads
        rng = StandardRNG(rng_seed)
        n_threads = Threads.nthreads()
        thread_rngs = split_rng(rng, n_threads)
        thread_stats = [OnlineStatistics() for _ in 1:n_threads]

        # Use :static scheduler to prevent task migration
        Threads.@threads :static for i in 1:n_paths
            thread_idx = mod1(Threads.threadid(), n_threads)
            path_or_value = generate_path(
                process,
                initial_state,
                config,
                thread_rngs[thread_idx]
            )
            result = functional(path_or_value)
            update!(thread_stats[thread_idx], result)
        end

        combined_stats = aggregate(thread_stats)
        return finalize(combined_stats)
    else
        error("Invalid parallel option: $parallel. Use :serial or :threads")
    end
end