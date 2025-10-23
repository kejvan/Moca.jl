"""
    simulate(process, functional, n_paths; initial_state=0.0, config, rng_seed=42, sampling=StandardSampling(), parallel=:serial)

Run Monte Carlo simulation by generating n_paths trajectories and computing statistics. Supports serial (:serial) and multi-threaded (:threads) execution. Use sampling parameter to specify variance reduction strategy.
"""
function simulate(
    process::StochasticProcess,
    functional::Function,
    n_paths::Int,
    config::PathConfig;
    initial_state::Float64=0.0,
    rng_seed::Int=42,
    sampling::SamplingStrategy=StandardSampling(),
    parallel::Symbol=:serial
)::MonteCarloEstimate
    if parallel == :serial
        return simulate_serial(
            process,
            functional,
            n_paths,
            config;
            initial_state=initial_state,
            rng_seed=rng_seed,
            sampling=sampling,
        )
    elseif parallel == :threads
        return simulate_parallel(
            process,
            functional,
            n_paths,
            config;
            initial_state=initial_state,
            rng_seed=rng_seed,
            sampling=sampling,
        )
    else
        error("Invalid parallel option: $parallel. Use :serial or :threads")
    end
end


function simulate_serial(
    process::StochasticProcess,
    functional::Function,
    n_paths::Int,
    config::PathConfig;
    initial_state::Float64=0.0,
    rng_seed::Int=42,
    sampling::SamplingStrategy=StandardSampling(),
)::MonteCarloEstimate
    rng = StandardRNG(rng_seed)
    rng_list = split_rng(rng, n_paths)

    stats = OnlineStatistics()

    if sampling isa StandardSampling
        for rng in rng_list
            result = generate_path(
                process,
                initial_state,
                config,
                rng;
                sampling=sampling
            ) |> functional
            update!(stats, result)
        end
    elseif sampling isa AntitheticSampling
        for rng in rng_list
            result_A, result_B = generate_path(
                process,
                initial_state,
                config,
                rng;
                sampling=sampling
            )
            result = (functional(result_A) + functional(result_B)) / 2.0
            update!(stats, result)
        end
    end

    return finalize(stats)
end


function simulate_parallel(
    process::StochasticProcess,
    functional::Function,
    n_paths::Int,
    config::PathConfig;
    initial_state::Float64=0.0,
    rng_seed::Int=42,
    sampling::SamplingStrategy=StandardSampling(),
)::MonteCarloEstimate
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

            if sampling isa StandardSampling
                for i in start_idx:end_idx
                    result = generate_path(
                        process,
                        initial_state,
                        config,
                        local_rng;
                        sampling=sampling
                    ) |> functional
                    update!(local_stats, result)
                end
            elseif sampling isa AntitheticSampling
                for i in start_idx:end_idx
                    result_A, result_B = generate_path(
                        process,
                        initial_state,
                        config,
                        local_rng;
                        sampling=sampling
                    )
                    result = (functional(result_A) + functional(result_B)) / 2
                    update!(local_stats, result)
                end
            end
            local_stats
        end
    end

    chunk_stats = fetch.(tasks)
    return chunk_stats |> aggregate |> finalize
end