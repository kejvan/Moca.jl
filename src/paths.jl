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
    generate_path(process, initial_state, config, rng, sampling=StandardSampling())

Generate a single trajectory. Returns Vector{Float64} if store_path=true, Float64 otherwise.
Dispatches to appropriate implementation based on sampling strategy.
"""
function generate_path(
    process::StochasticProcess,
    initial_state::Float64,
    config::PathConfig,
    rng::StandardRNG,
    sampling::SamplingStrategy=StandardSampling(),
)::Union{Vector{Float64},Float64}
    return _generate_path(sampling, process, initial_state, config, rng)
end

# Standard sampling implementation
function _generate_path(
    ::StandardSampling,
    process::StochasticProcess,
    initial_state::Float64,
    config::PathConfig,
    rng::StandardRNG,
)::Union{Vector{Float64},Float64}
    state = initial_state
    if config.store_path
        states = Float64[initial_state]
        for _ = 1:config.n_steps
            Z = rand_normal(rng)
            state = step(process, state, config.dt, Z)
            push!(states, state)
        end
        return states
    else
        for _ = 1:config.n_steps
            Z = rand_normal(rng)
            state = step(process, state, config.dt, Z)
        end
        return state
    end
end

# Antithetic sampling implementation
function _generate_path(
    ::AntitheticSampling,
    process::StochasticProcess,
    initial_state::Float64,
    config::PathConfig,
    rng::StandardRNG,
)::Union{Vector{Float64},Float64}
    state_A = initial_state
    state_B = initial_state
    if config.store_path
        states_A = Float64[initial_state]
        states_B = Float64[initial_state]
        for _ = 1:config.n_steps
            Z = rand_normal(rng)
            state_A = step(process, state_A, config.dt, Z)
            state_B = step(process, state_B, config.dt, -Z)
            push!(states_A, state_A)
            push!(states_B, state_B)
        end
        return (states_A .+ states_B) ./ 2.0
    else
        for _ = 1:config.n_steps
            Z = rand_normal(rng)
            state_A = step(process, state_A, config.dt, Z)
            state_B = step(process, state_B, config.dt, -Z)
        end
        return (state_A + state_B) / 2.0
    end
end