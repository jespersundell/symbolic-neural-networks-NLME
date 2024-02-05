# Functions to fit covariate model in form of a symbolic regression network to data

using Lux, Statistics, SymbolicUtils, Zygote

##############################################################################
#### Read and prepare data
##############################################################################

# Struct for input data
struct InputData
    covariates # vector or real
    infusionrate::Vector # infusion rate
    t::Vector # time vector
    hs::Vector # time differences between rate changes. Δ(t+1)-Δ(t) = hs
    y_obs::Vector # observations, remove? (already in y_data)
    t_obs::Vector # time vector for observations
    idx_meas::Vector # indices of y_obs where we have observations
    covariate_normalization::Vector # conversion of covariates to 0->1
    pkparam_normalization::Vector # conversion of volumes and clearances to 0->1
end

function prepare_data(df_inf, df_cov, df_pkparams_eleveld)

    # Find max predictions of V1, V2 etc from the Eleveld model, to use for scaling
    maxV1 = maximum(df_pkparams_eleveld[!, :V1]) * 1000 # [ml]
    maxV2 = maximum(df_pkparams_eleveld[!, :V2]) * 1000
    maxV3 = maximum(df_pkparams_eleveld[!, :V3]) * 1000
    maxCL = maximum(df_pkparams_eleveld[!, :CL]) * 1000 / 60 # [ml/s]
    maxQ2 = maximum(df_pkparams_eleveld[!, :Q2]) * 1000 / 60
    maxQ3 = maximum(df_pkparams_eleveld[!, :Q3]) * 1000 / 60

    normalizationpkparams = [maxV1, maxV2, maxV3, maxCL, maxQ2, maxQ3] # Normalization factor for PK parameters (max values in Eleveld data set)

    n_pat = size(df_cov)[1] # Number of patients

    x_data = Vector{InputData}(undef, n_pat) # To save input data: covariates, infusion rates, time vector, measurements and scaling
    y_data = Vector{Vector{Float32}}(undef, n_pat)

    input_normalization = [maximum(df_cov.AGE), maximum(df_cov.WGT), 0] # [88,100,0.5] # Normalization factor for AGE and WGT (max value in dataset)

    for i = 1:n_pat
        subject = df_inf[in([i]).(df_inf.ID), :] # Get all infusions and observations for a patient

        gdrval = df_cov.GDR[i] == 1 ? -0.5f0 : 0.5f0 # male = -0.5, female = 0.5
        idx_meas = findall(x -> !iszero(x), subject.Measurement)
        hs = diff(subject.Time)
        hs = [hs; hs[end]] # Add last hs
        x_data[i] = InputData(
            Float32.([df_cov.AGE[i] / input_normalization[1], df_cov.WGT[i] / input_normalization[2], gdrval]), Float32.(subject.InfusionRate),
            Float32.(subject.Time),
            Float32.(hs),
            Float32.(exp.(subject.Measurement[idx_meas])), # remove?
            Float32.(subject.Time[idx_meas]),
            Int32.(idx_meas),
            Float32.(input_normalization),
            Float32.(normalizationpkparams))

        y_data[i] = exp.(subject.Measurement[idx_meas]) # Measurements are logarithmic in data set.
    end

    return x_data, y_data
end

##############################################################################
#### Create and train symbolic regression network
##############################################################################

function createSNN()
    snn = Chain(
        DenseMaskLayer(3, 5),
        x -> activation_1(x),
        DenseMaskLayer(3, 5),
        x -> activation_2(x),
        DenseMaskLayer(3, 1, activation=abs)
    )
    return snn
end

function oneparamnn()
    nn = DenseMaskLayer(3, 1, activation=abs)
    return nn #, params, masks
end



# Function to create symbolic regression network with two PK parameters in a parallel structure
function setupmodel(; rng=Random.default_rng())
    modelV1 = createSNN()
    modelCL = createSNN()
    modelV2 = oneparamnn()
    modelV3 = oneparamnn()
    modelQ2 = oneparamnn()
    modelQ3 = oneparamnn()
    model = Parallel(nothing, modelV1, modelV2, modelV3, modelCL, modelQ2, modelQ3)
    model = f32(model) # Convert to Float32
    params, masks = Lux.setup(rng, model)

    # Set weight parameters to zero for V2, V3, Q2 and Q3
    params[2].weight .= 0.0f0
    masks[2].W_mask .= 0.0f0 
    params[3].weight .= 0.0f0
    masks[3].W_mask .= 0.0f0
    params[5].weight .= 0.0f0
    masks[5].W_mask .= 0.0f0
    params[6].weight .= 0.0f0
    masks[6].W_mask .= 0.0f0

    # Initiate power function. Exponent in a^b only takes numerical values, means only bias term. Remove weight so that we cannot get x^x and only x^a.
    params[1][1].weight[end, :] .= 0.0
    masks[1][1].W_mask[end, :] .= 0.0
    params[4][1].weight[end, :] .= 0.0
    masks[4][1].W_mask[end, :] .= 0.0

    return model, params, masks
end


"""
    activation_1(x)

Activation function for first layer. Assumes input is of size 7.

# Arguments:
- `x`: Input vector (or matrix).
"""
function activation_1(x)
    return [x[1, :] (x[2, :] .* x[3, :]) powerfunction.(x[4, :], x[5, :])]'
end


"""
    activation_2(x)

Activation function for second layer. Assumes input is of size 5.

# Arguments:
- `x`: Input vector (or matrix).
"""
function activation_2(x)
    return [x[1, :] (x[2, :] .* x[3, :]) (x[4, :] ./ (x[5, :] .+ one(eltype(x))))]'
end

# Special power function called by activation functions.
function powerfunction(x1, x2)
    z = zero(eltype(x1))
    if x1 ≈ z
        return z # Return zero if base is zero.
    else
        return abs.(x1) .^ x2 # |a|^b, to avoid for example (-0.5)^1.2
    end
end

# Computes loss (ALE) over infusion data for one patient for our given model.
function loss(params, masks, model, x::InputData, y)
    u = x.infusionrate # Infusion rates
    v = zeros(eltype(u), length(u)) # Bolus doses # FIXME: remove these later
    # v = x.v # Bolus doses
    hs = x.hs # Time differences between rate changes. Δ(t+1)-Δ(t) = hs

    θ_scaled, _ = model(x.covariates, params, masks) # Scaled PK parameters (V1 and CL) from (0,1)
    θ = reduce(vcat, θ_scaled).* x.pkparam_normalization # Scale predicted PK parameters from (0,1) to range (0,maxval in dataset), [ml] for volumes and [ml/s] for clearances
    V1,V2,V3,CL,Q2,Q3 = θ

    # Convert from parameterization in volumes and clearances to parameterization in rates
    k10 = CL / V1 # [1/s]
    k12 = Q2 / V1 # [1/s]
    k13 = Q3 / V1
    k21 = Q2 / V2
    k31 = Q3 / V3
    θ = [k10, k12, k13, k21, k31, V1] # PK parameter vector, units [1/s] and [ml]

    V1inv, λ, λinv, R = PK3(θ) # Create necessary matrices for simulation of 3rd order compartment model

    totalloss = 0.0f0 # squared error
    j = 1 # counter to keep track of next free spot in y
    x_state = zeros(eltype(params[1][1][1][1]), 3)  # initial state (make sure that we can run ForwardDiff through it)
    for i in eachindex(hs) # Iterate through all time samples
        if i in x.idx_meas # if we want to compute output
            x_state, yi = @inbounds updatestateoutput(x_state, hs[i], V1inv, λ, λinv, R, u[i], v[i]) # update state and compute output
            totalloss += abs.(log(y[j] / abs(yi))) # ALE
            j += 1
        else
            x_state = @inbounds updatestate(x_state, hs[i], λ, λinv, u[i], v[i]) # update state
        end
    end
    return totalloss # Loss over all time samples

end

# Computes loss over infusion data for all patients for our given model.
function loss(params, masks, model, x_data::Vector{InputData}, y_data)
    totalloss = zero(eltype(params[1][1][1][1]))
    for i in eachindex(x_data)
        totalloss += loss(params, masks, model, x_data[i], y_data[i]) # Compute loss for each patient
    end
    return totalloss / length(x_data) # Mean loss
end

# Training function 
function train!(x_data, y_data, epochs, params, masks, model, opt_state)
    training_loss = zeros(Float32, epochs)
    l = loss(params, masks, model, x_data, y_data)
    println("Loss before training: $l")

    for epoch in 1:epochs

        l, back = pullback(p -> loss(p, masks, model, x_data, y_data), params)
        gs = back(one(l))[1]
        opt_state, params = Optimisers.update(opt_state, params, gs)

        training_loss[epoch] = l
        # epoch % 100 == 0 && println("Epoch: $(epoch) | Loss: $(l)")

        if isnan(l)
            println("Loss is NaN, breaking training.")
            break
        end
    end
    finalloss = training_loss[epochs]
    println("Loss after training: $finalloss")
    return params, training_loss # masks not updated during training
end


##############################################################################
## Hessian-based pruning of parameters and covariates
#############################################################################

## function to set least non-zero salience parameter and corresponding layer state to zero
## returns parameters and layer masks
function zero_param_smallest_salience(params, masks, model, x_data, y_data)
    # set lowest salience parameter to 0.0 and freeze it
    params_flat, rebuild_params = destructure(params)
    masks_flat, rebuild_masks = destructure(masks)
    param_salience = salience_params(params_flat, rebuild_params, masks, model, x_data, y_data)

    index_vec = Vector(1:length(param_salience)) # Only find min of those who are trainable (not yet pruned)

    # Split into n_snns vectors, and remove the smallest of each subnetwork.
    # + remove those for V2, V3, Q2 and Q3, since these should not be part of the pruning
    # n_snn = length(params) # nbr of pk parameters to estimate
    n_params_snn = 44 # getnbrparams(params[1]) # nbr of parameters in each SNN (V1 and CL)
    # N = Int32(length(params_flat) / n_snn) # nbr of parameters in each SNN

    # For V1 and CL
    range_V1 = 1:n_params_snn
    range_CL = n_params_snn+4+4+1:2*n_params_snn + 4+4
    # indices_smallest_salience = zeros(Int32, 2) # indices of smallest salience parameters for V1 and CL
    index_paramsleft = findall(x -> !iszero(x), masks_flat[range_V1]) # V1, find all non-zero masks
    indextemp = index_vec[range_V1][index_paramsleft]
    V1_index_smallest_salience = indextemp[findmin(param_salience[range_V1][index_paramsleft])[2]]

    index_paramsleft = findall(x -> !iszero(x), masks_flat[range_CL]) # CL, find all non-zero masks
    indextemp = index_vec[range_CL][index_paramsleft]
    CL_index_smallest_salience = indextemp[findmin(param_salience[range_CL][index_paramsleft])[2]]
    
    # Set parameter and mask to zero for the pruned parameter(s)
    params_flat[V1_index_smallest_salience] = 0.0
    masks_flat[V1_index_smallest_salience] = 0.0
    params_flat[CL_index_smallest_salience] = 0.0
    masks_flat[CL_index_smallest_salience] = 0.0

    # Rebuild and return parameters and masks
    params = rebuild_params(params_flat)
    masks = rebuild_masks(masks_flat)

    return params, masks, [V1_index_smallest_salience, CL_index_smallest_salience]
end

function getnbrparams(masks)
    masks_flat, _ = destructure(masks)
    index_paramsleft = findall(x -> !iszero(x), masks_flat)
    return length(index_paramsleft)
end

# Compute the loss of the model using a flat representation of the parameters, for computation of the diagonal hessian with respect to the parameters.
function loss_diaghess(params_flat::Vector, rebuild_params, masks, model, x_data, y_data)
    params = rebuild_params(params_flat)
    totalloss = loss(params, masks, model, x_data, y_data)
    return totalloss
end

# Calculate diagonal hessian with respect to model parameters
function diaghessian_params(params_flat, rebuild_params, masks, model, x_data::InputData, y_data)
    sumdiaghessian = diaghessian(p -> loss_diaghess(p, rebuild_params, masks, model, x_data, y_data), params_flat)[1]
    return sumdiaghessian
end

# Calculate salience of model parameters: θ^2 * diagonal element of hessian
function salience_params(params_flat, rebuild_params, masks, model, x_data::Vector, y_data) # compute salience of all parameters
    sumdiaghessian = zeros(eltype(params_flat[1]), length(params_flat))
    for i in eachindex(x_data)
        sumdiaghessian += diaghessian_params(params_flat, rebuild_params, masks, model, x_data[i], y_data[i])
    end
    param_salience = (params_flat .^ 2) .* abs.(sumdiaghessian) # Compute salience
    return param_salience
end

##############################################################################
## Simulation function
##############################################################################
function simulate(params, masks, model, x)
    u = x.infusionrate # Infusion rates
    v = zeros(eltype(u), length(u)) # Bolus doses # FIXME: remove these later
    hs = x.hs # Time differences between rate changes. Δ(t+1)-Δ(t) = hs

    θ_scaled, _ = model(x.covariates, params, masks) # Scaled PK parameters (V1 and CL) from (0,1)
    θ = reduce(vcat, θ_scaled) .* x.pkparam_normalization # Scale predicted PK parameters from (0,1) to range (0,maxval in dataset), [ml] for volumes and [ml/s] for clearances
    V1, V2, V3, CL, Q2, Q3 = θ

    # Convert from parameterization in volumes and clearances to parameterization in rates
    k10 = CL / V1 # [1/s]
    k12 = Q2 / V1 # [1/s]
    k13 = Q3 / V1
    k21 = Q2 / V2
    k31 = Q3 / V3
    θ = [k10, k12, k13, k21, k31, V1] # PK parameter vector, units [1/s] and [ml]

    # totalloss = 0.0f0 # squared error
    ys = pk3sim(θ, u, v, hs, x.idx_meas) # Simulate output

    return ys
end

function simulatewhite(x)
    fCLmale(age, wgt) = (26.88 - 0.029 * age) * wgt / 60 # [ml/s]
    fCLfemale(age, wgt) = (37.87 - 0.198 * age) * wgt / 60 # [ml/s]
    fV1male(age, wgt) = (175.5 + 0.056 * age) * wgt # [ml]
    fV1female(age, wgt) = (191.8 - 0.669 * age) * wgt # [ml]

    u = x.infusionrate # Infusion rates
    v = zeros(eltype(u), length(u)) # Bolus doses # FIXME: remove these later
    hs = x.hs # Time differences between rate changes. Δ(t+1)-Δ(t) = hs

    age = x.covariates[1] * x.covariate_normalization[1] # [years]
    wgt = x.covariates[2] * x.covariate_normalization[2] # [kg]
    CL = 0.0f0 # [ml/s]
    V1 = 0.0f0 # [ml]
    if x.covariates[3] == -0.5 # male
        CL = fCLmale(age, wgt)
        V1 = fV1male(age, wgt)
    else # female
        CL = fCLfemale(age, wgt)
        V1 = fV1female(age, wgt)
    end

    k10 = CL / V1 # [1/s]
    k12 = 0.114f0 / 60 # [1/s]
    k13 = 0.042f0 / 60
    k21 = 0.055f0 / 60
    k31 = 0.0033f0 / 60
    θ = [k10, k12, k13, k21, k31, V1] # PK parameter vector, units [1/s] and [ml]

    ys = pk3sim(θ, u, v, hs, x.idx_meas) # Simulate output

    return ys
end

# Plot predictions vs observations in log-log scale
function plotpredictions(x_data, y_data, params, masks, model)
    p = scatter(aspect_ratio=:equal, ylims=(-0.1, Inf), xlims=(-0.1, Inf))
    for i in eachindex(x_data)
        y = simulate(params, masks, model, x_data[i])
        # scatter!(p3, y, y_data[i], label="", xlabel="Predictions", ylabel="Observations")
        # plot in log log scale 
        scatter!(p, log.(y), log.(y_data[i]), label="", xlabel="Predictions", ylabel="Observations")
    end
    return p
end

##############################################################################
## Function for prediction errors
##############################################################################

# Function for computing prediction errors

function MSE(y_meas, y_pred)
    mse = sum(abs2.(y_meas .- y_pred)) ./ length(y_pred)
    return mse
end

function MdAPE(y_meas, y_pred) # Median Absolute Prediction Error: median(abs((C_observed - C_predicted)/C_predicted* 100 %))
    mdape = 0.0
    ind = findall(x -> x = !isapprox(x, 0.0, atol=1e-6), y_pred)
    if length(ind) > 0
        mdape = 100 * median(abs.((y_meas[ind] .- y_pred[ind]) ./ y_pred[ind])) # %
    end
    return mdape
end

function MdPE(y_meas, y_pred) # Absolute Prediction Error: abs((C_observed - C_predicted)/C_predicted* 100 %))
    ape = 0.0
    ind = findall(x -> x = !isapprox(x, 0.0, atol=1e-6), y_pred)
    if length(ind) > 0
        ape = median(100 * (y_meas[ind] .- y_pred[ind]) ./ y_pred[ind]) # %
    end
    return ape
end

function MdALE(y_meas, y_pred) # Median Absolute Logarithmic Error: median(abs(log(C_observed/C_predicted)))
    mdale = 0.0
    ind = findall(x -> x = !isapprox(x, 0.0, atol=1e-6), y_pred)
    if length(ind) > 0
        mdale = median(abs.(log.(y_meas[ind] ./ abs.(y_pred[ind]))))
    end
    return mdale
end


function MdLE(y_meas, y_pred)
    le = 0.0
    ind = findall(x -> x = !isapprox(x, 0.0, atol=1e-6), y_pred)
    if length(ind) > 0
        le = median(log.(y_meas[ind] ./ abs.(y_pred[ind])))
    end
    return le
end

function getpredictionerrors(x_data, y_data,params, masks, model)

    # Compute MSE loss for the White model
    # Compute bias and precision
    # Compute mean(MdLE), mean(MdALE), mean(MdPE) and mean(MdAPE)

    summdle_white = 0.0
    summdale_white = 0.0
    summdpe_white = 0.0
    summdape_white = 0.0
    summse_white = 0.0
    for i in eachindex(x_data)
        ysim = simulatewhite(x_data[i])
        summdle_white += MdLE(y_data[i], ysim)
        summdale_white += MdALE(y_data[i], ysim)
        summdpe_white += MdPE(y_data[i], ysim)
        summdape_white += MdAPE(y_data[i], ysim)
        summse_white += MSE(y_data[i], ysim)
    end
    println("White model, MSE: ", summse_white) # 130
    println("White model, MeanMdLE: ", summdle_white / length(x_data))
    println("White model, MeanMdALE: ", summdale_white / length(x_data))
    println("White model, MeanMdPE: ", summdpe_white / length(x_data))
    println("White model, MeanMdAPE: ", summdape_white / length(x_data))

    # Compute other prediction errors for our model
    summdle = 0.0
    summdale = 0.0
    summdpe = 0.0
    summdape = 0.0
    summse = 0.0
    for i in eachindex(x_data)
        ysim = simulate(params, masks, model, x_data[i])
        summdle += MdLE(y_data[i], ysim)
        summdale += MdALE(y_data[i], ysim)
        summdpe += MdPE(y_data[i], ysim)
        summdape += MdAPE(y_data[i], ysim)
        summse += MSE(y_data[i], ysim)
    end
    loss(params, masks, model, x_data, y_data)
    println("Our model, MSE: ", summse)
    println("Our model, MeanMdLE: ", summdle / length(x_data))
    println("Our model, MeanMdALE: ", summdale / length(x_data))
    println("Our model, MeanMdPE: ", summdpe / length(x_data))
    println("Our model, MeanMdAPE: ", summdape / length(x_data))

end
##############################################################################
## Functions for reading equations from network structure
##############################################################################

# Get equations in readable form from network structure.

# using SymbolicUtils

"""
     layer2string(nn, layer, input, [rounding=false])

Converts neural network to readable string. Is called for each layer on the network with the specific input (also treating activation functions as specific layers). There are no explicit checks on the size of the input.

Matches the following network structure
Unit, Power function, division
    activation_1(x) = [x[1, :] (x[2, :] .* x[3, :]) powerfunction.(x[4, :], x[5, :])]'
    activation_2(x) = [x[1, :] (x[2, :] .* x[3, :]) (x[4, :] ./ (x[5, :] .+ one(eltype(x))))]'

    snn = Chain(
        DenseMaskLayer(3, 5),
        x -> activation_1(x),
        DenseMaskLayer(3, 5),
        x -> activation_2(x),
        DenseMaskLayer(3, 1, activation=abs)
    )
# Arguments:
- `nn`: Neural network with structure as given above.
- `layer`: Int, which layer we are considering.
- `input`: Vector, input to this layer.
- `rounding`: Bool. If parameter values should be rounded for visibility. Default: rounding = false.

Returns a readable String from this layer. (In a vector if the layer has several outputs).
"""
function layer2string(params, masks, layer, input, roundvals=false)
    p = params
    m = masks

    if layer in [1 3 5] # Dense layers
        W = p.weight .* m.W_mask  # Parameter value of weight with mask
        B = p.bias .* m.b_mask   # Parameter value of bias with mask

        if roundvals # rounding for visibility
            W = round.(W, digits=2) # round for visibility
            B = round.(B, digits=2)
        end

        n_outputs = size(W, 1)
        n_inputs = size(W, 2)

        l_str = String[]
        for j = 1:n_outputs
            push!(l_str, "")
            for k = 1:n_inputs
                w0 = W[j, k]
                if w0 != 0 # weights
                    if w0 < 0
                        l_str[j] = "$(l_str[j])+($(w0))*($(input[k]))"
                    else
                        l_str[j] = "$(l_str[j])+$(w0)*($(input[k]))"
                    end
                end
            end
            b0 = B[j]
            if b0 != 0 # bias
                if b0 < 0
                    l_str[j] = "$(l_str[j])+($(b0))"
                else
                    l_str[j] = "$(l_str[j])+$(b0)"
                end
            end
            if layer == 5 # put abs on output
                l_str[j] = "(abs($(l_str[j])))"
            end
        end
    elseif layer == 2 # Activation functions (here treated as specific layers)
        l_str = String[]
        for i in eachindex(input)
            if i == 2 # Multiplication
                if isempty(input[i]) || isempty(input[i+1]) || input[i] == "0" || input[i+1] == "0" # if either of the inputs are 0                    push!(l_str, "0")
                else
                    push!(l_str, "($(input[i]))*($(input[i+1]))")
                end
            elseif isempty(input[i])
                if i == 5 # power function exponent = 0
                    push!(l_str, "1") # x^0 = 1
                else
                    push!(l_str, "0")
                end
            elseif i == 1 # passthrough
                push!(l_str, "($(input[i]))")
            elseif i == 5 # power function
                push!(l_str, "(abs($(input[4])))^($(input[i]))")
            end
        end
    elseif layer == 4 # Activation functions (here treated as specific layers)
        l_str = String[]
        for i in eachindex(input)
            if i == 2 # Multiplication
                if isempty(input[i]) || isempty(input[i+1]) || input[i] == "0" || input[i+1] == "0" # if either of the inputs are 0
                    push!(l_str, "0")
                else
                    push!(l_str, "($(input[i]))*($(input[i+1]))")
                end
            elseif isempty(input[i])
                if i != 3
                    push!(l_str, "0")
                end
            elseif i == 1 # passthrough
                push!(l_str, "($(input[i]))")
            elseif i == 4 # division function
                push!(l_str, "($(input[i]))/($(input[i+1]) + 1)")
            end
        end
    end
    return l_str
end


"""
    get_fctsfrommodel(model)

Converts Flux model to callable functions or readable expressions (with readable=true), one for each subnetwork.
"""
function getexpressions(params, masks; roundvals=false)
    n_networks = length(params) # V1 and CL
    fcts_expr = Vector{Function}(undef, n_networks)
    readable_expr = Vector{String}(undef, n_networks)
    input = ["x_age"; "x_wgt"; "x_gdr"] # or ["age"; "wgt"; "x_gdr"]?
    pkparam_normalization = x_data[1].pkparam_normalization

    for i in [1,4] # loop over params for the two PK parameters
        p = params[i]
        m = masks[i]

        Y1 = layer2string(p[1], m[1], 1, input, roundvals)
        Y2 = layer2string(p[2], m[2], 2, Y1, roundvals)
        Y3 = layer2string(p[3], m[3], 3, Y2, roundvals)
        Y4 = layer2string(p[4], m[4], 4, Y3, roundvals)
        Y = layer2string(p[5], m[5], 5, Y4, roundvals)[1]
        Yscaled = Y # placeholder, to initialize Yscaled
        if i == 1
            Yscaled = "$(pkparam_normalization[1])" * Y # scale V1 params back to normal
        elseif i == 2
            Yscaled = "$(pkparam_normalization[4])" * Y # scale CL params back to normal
        end

        readable_expr[i] = string(expand(eval(Meta.parse(Yscaled)))) # simple readable expression
        fcts_expr[i] = eval(Meta.parse("(x_age,x_wgt,x_gdr) -> " * Yscaled)) # parse into functions
    end

    # V2, V3, Q2 and Q3
    for i in [2,3,5,6]
        readable_expr[i] = string(abs(params[i].bias[1])*pkparam_normalization[i]) # simple readable expression
    end

    fcts_expr[2] = eval(Meta.parse("(x_age,x_wgt,x_gdr) -> " * readable_expr[2])) # parse into functions
    fcts_expr[3] = eval(Meta.parse("(x_age,x_wgt,x_gdr) -> " * readable_expr[3])) # parse into functions
    fcts_expr[5] = eval(Meta.parse("(x_age,x_wgt,x_gdr) -> " * readable_expr[5])) # parse into functions
    fcts_expr[6] = eval(Meta.parse("(x_age,x_wgt,x_gdr) -> " * readable_expr[6])) # parse into functions

    println("V1 (inputs scaled 0->1, output correctly scaled) = \n", readable_expr[1], "\n",
    "V2 = ", readable_expr[2], "\n",
    "V3 = ", readable_expr[3], "\n",
    "CL (inputs scaled 0->1, output correctly scaled) = \n", readable_expr[4], "\n",
    "Q2 = ", readable_expr[5], "\n",
    "Q3 = ", readable_expr[6]
    )

    return fcts_expr, readable_expr
end

function custom_argmin(arr::Vector)
    min_val = Inf  # Initialize with positive infinity
    min_index = 0

    for (index, value) in enumerate(arr)
        if value != 0 && value < min_val
            min_val = value
            min_index = index
        end
    end
    return min_index, min_val
end

##############################################################################
## Functions for training and pruning of SNN
##############################################################################
"""
covmodeling(x_data, y_data, seed, opt, n_epochs_train, n_epochs_prune, n_prune_init, n_prune_byone)

    symbolic_regression_nn(i, x, y, opt, seed, nepochs_init, nepochs_prune)

Training and pruning of a symbolic regression regression network on data.

# Arguments:
- `x_data`: Vector{InputData}. Training x data with length(x) == nbr of patients.
- `y_data`: Vector{Vector{Number}}. Training y data with length(y) == nbr of patients.
- `seed`: Int. Seed for reproducibility.
- `opt`: Optimizer object, for example Adam(0.0001).
- `n_epochs_train`: Int. Nbr of epochs to train before start of pruning.
- `n_epochs_prune`: Int. Nbr of epochs to train between parameter pruning steps.
- `n_prune_init`: Int. Nbr of parameters to prune at once, before pruning one parameter at a time.
- `n_prune_byone`: Int. Nbr of parameters to prune one at a time, after pruning several parameters at once.

Returns params, masks, model, training_loss, final_loss
"""
function covmodeling(x_data, y_data, seed, opt, n_epochs_train, n_epochs_prune, n_prune_init, n_prune_byone)
    Random.seed!(rng, seed)

    model, params, masks = setupmodel() # Setup model for 6 PK parameters
    opt_state = Optimisers.setup(opt, params) # Optimiser state

    params, training_loss = train!(x_data, y_data, n_epochs_train, params, masks, model, opt_state) # Train

    # Parameter pruning, several at once
    println("Pruning ", n_prune_init, " parameters at once.")
    for i = 1:n_prune_init
        # println("Parameter # to be removed: ", i)
        params, masks, removed_indices = zero_param_smallest_salience(params, masks, model, x_data, y_data)
        # println("Removed parameter index: ", removed_indices[1], " and ", removed_indices[2])
        if isnan(loss(params, masks, model, x_data, y_data))
            println("Loss is NaN, breaking training.")
            break
        end
    end

    params, _ = train!(x_data, y_data, n_epochs_train, params, masks, model, opt_state) # Train

    # println("Number of (trainable) parameters left: ", getnbrparams(masks))
    # Parameter pruning, one at a time
    println("Pruning ", n_prune_byone, " parameters one at a time.")
    for i = 1:n_prune_byone
        # println("Parameter # to be removed: ", n_prune_init + i)
        params, masks, removed_indices = zero_param_smallest_salience(params, masks, model, x_data, y_data)
        # println("Removed parameter index: ", removed_indices[1], " and ", removed_indices[2])
        if isnan(loss(params, masks, model, x_data, y_data))
            println("Loss is NaN, breaking training.")
            break
        end
        params, _ = train!(x_data, y_data, n_epochs_prune, params, masks, model, opt_state)
    end

    println("Number of parameters left for V1 and CL, respectively: ", getnbrparams(masks[1]))

    final_loss = loss(params, masks, model, x_data, y_data)
    println("Final loss after training and pruning of parameters: ", final_loss)
    return params, masks, model, training_loss, final_loss
end