# DenseMaskLayer.jl, for pruning of neural network parameters with Lux.jl

using Lux, Random, Zygote

struct DenseMaskLayer <: Lux.AbstractExplicitLayer
    init_weight
    init_bias
    init_W_mask
    init_b_mask
    activation
end
function DenseMaskLayer(weight::AbstractArray, bias::AbstractArray, W_mask::AbstractArray, b_mask::AbstractArray, activation::Function)
    return DenseMaskLayer(() -> copy(weight), () -> copy(bias), () -> copy(W_mask), () -> copy(b_mask), activation)
end

# weight and bias are parameters
Lux.initialparameters(::AbstractRNG, layer::DenseMaskLayer) = (weight=layer.init_weight(), bias=layer.init_bias(),)
# W_mask and b_mask are states
Lux.initialstates(::AbstractRNG, layer::DenseMaskLayer) = (W_mask=layer.init_W_mask(), b_mask=layer.init_b_mask(),)
(l::DenseMaskLayer)(x, ps, st) = l.activation.((st.W_mask .* ps.weight * x) .+ (st.b_mask .* ps.bias)), st

rng = Random.default_rng()

function DenseMaskLayer(in_dims, out_dims; activation = identity)
    layer = DenseMaskLayer(randn(rng, out_dims, in_dims), randn(rng, out_dims), ones(out_dims, in_dims), ones(out_dims), activation)
    return layer
end
