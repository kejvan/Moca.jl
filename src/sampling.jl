abstract type SamplingStrategy end

struct StandardSampling <: SamplingStrategy end
struct AntitheticSampling <: SamplingStrategy end