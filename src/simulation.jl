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
        n_chunks = Threads.nthreads()
        chunk_rngs = split_rng(rng, n_chunks)

        paths_per_chunk = div(n_paths, n_chunks)

        tasks = map(1:n_chunks) do chunk_id
            Threads.@spawn begin
                local_rng = chunk_rngs[chunk_id]
                local_stats = OnlineStatistics()

                start_idx = (chunk_id - 1) * paths_per_chunk + 1
                end_idx = chunk_id == n_chunks ? n_paths : chunk_id * paths_per_chunk

                for i in start_idx:end_idx
                    path_or_value = generate_path(
                        process,
                        initial_state,
                        config,
                        local_rng
                    )
                    result = functional(path_or_value)
                    update!(local_stats, result)
                end
                local_stats
            end
        end

        chunk_stats = fetch.(tasks)
        combined_stats = aggregate(chunk_stats)
        return finalize(combined_stats)
    else
        error("Invalid parallel option: $parallel. Use :serial or :threads")
    end
end