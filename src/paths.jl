"""
    PathConfig

Configuration for path generation.
"""
struct PathConfig
    n_steps::Int
    dt::Float64
    store_path::Bool  # If false, only terminal value is returned
end

"""
    generate_path(process, initial_state, config, rng)

Generate a single trajectory. Returns Vector{Float64} if store_path=true, Float64 otherwise.
"""
function generate_path(
    process::StochasticProcess,
    initial_state::Float64,
    config::PathConfig,
    rng::StandardRNG
)::Union{Vector{Float64},Float64}
    state = initial_state
    if config.store_path
        states = Float64[initial_state]
        for _ = 1:config.n_steps
            state = step(process, state, config.dt, rng)
            push!(states, state)
        end
        return states
    else
        for _ = 1:config.n_steps
            state = step(process, state, config.dt, rng)
        end
        return state
    end
end