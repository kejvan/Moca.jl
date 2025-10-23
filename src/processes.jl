abstract type StochasticProcess end

"""
Brownian motion with drift: dX = drift * dt + volatility * dW
"""
@kwdef struct WienerProcess <: StochasticProcess
    drift::Float64
    volatility::Float64
end

"""
    step(process::WienerProcess, current_state, dt, Z)

Advance process by dt using Euler-Maruyama discretization. Z is a standard normal random variable.
"""
function step(
    process::WienerProcess,
    current_state::Float64,
    dt::Float64,
    Z::Float64
)::Float64
    dX = process.drift * dt + process.volatility * sqrt(dt) * Z
    return current_state + dX
end

"""
Geometric Brownian motion: dS = drift * S * dt + volatility * S * dW
"""
@kwdef struct GeometricBrownianMotion <: StochasticProcess
    drift::Float64
    volatility::Float64
end

"""
    step(process::GeometricBrownianMotion, current_state, dt, Z)

Advance process by dt using Euler-Maruyama discretization. Z is a standard normal random variable.
"""
function step(
    process::GeometricBrownianMotion,
    current_state::Float64,
    dt::Float64,
    Z::Float64
)::Float64
    dS =
        process.drift * current_state * dt +
        process.volatility * current_state * sqrt(dt) * Z
    return current_state + dS
end

"""
Mean-reverting process: dX = mean_reversion_speed * (long_term_mean - X) * dt + volatility * dW
"""
@kwdef struct OrnsteinUhlenbeckProcess <: StochasticProcess
    mean_reversion_speed::Float64
    long_term_mean::Float64
    volatility::Float64
end

"""
    step(process::OrnsteinUhlenbeckProcess, current_state, dt, Z)

Advance process by dt using Euler-Maruyama discretization. Z is a standard normal random variable.
"""
function step(
    process::OrnsteinUhlenbeckProcess,
    current_state::Float64,
    dt::Float64,
    Z::Float64
)::Float64
    dX =
        process.mean_reversion_speed * (process.long_term_mean - current_state) * dt + process.volatility * sqrt(dt) * Z
    return current_state + dX
end
