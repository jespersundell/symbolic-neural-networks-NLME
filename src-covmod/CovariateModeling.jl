# Covariate model for White data set, using the same covariates as in the original publication.
# Date: 2024-01-30
# Three covariates. 2 PK parameters (CL and V1) that we seek covariate expressions for. The other 4 (V2,V3,Q2,Q3), we seek a constant.

# Training data:
# x_data: struct that contains covariates (normalized), infusion data (time, infusions), observation data (time, observations), and scaling for PK parameters. 
# y_data: vector with measurements for each individual

# Training:
# Epochs = 5000, learning rate = 0.005, Optimizer Adam, 8 inits, 10 parameters removed at once, 21 parameters removed one at a time. Resulting in 10 parameters left for V1 and CL, respectively.
# Loss function Mean(ALE)
# Input to simulation should be in 1/seconds and in ml for V1.

# Pruning:
# Pruning 1 parameter means pruning 1 from each PK parameter network with covariates (V1 and CL).
# Pruning a total of 31 parameters results in 10 parameters left for V1 and CL, respectively.

# Run the SNN:
# julia
# include("CovModelingWhite.jl")

# If you run the code directly from the command line, julia CovModelingWhite.jl, you may get the following error:
# NOTE: SymbolicUtils is not thread-safe, so if the code is run with multiple threads, it may crash if the training finishes at the same time. This may happen for a few training epochs.

using Pkg
cd(@__DIR__)
Pkg.activate("..")

using CSV, DataFrames, JLD2, Lux, Optimisers, Plots, Random, StaticArrays, Zygote

include("DenseMaskLayer.jl") # Custom masked layer
include("covmodelfunctions.jl") # Functions for covariate modeling
include("fastpksim.jl") # Simulator for PK model

############################################################################################################

# Read data
df_inf = CSV.read("csv/infusiondata_white.csv", DataFrame) # Infusion data for White data set
df_cov = CSV.read("csv/covariatedata_white.csv", DataFrame) # Covariate data for White data set
df_pkparams_eleveld = CSV.read("csv/pkparams_eleveld.csv", DataFrame) # PK parameters for Eleveld data set, for scaling purposes only

rng = Random.default_rng()
Random.seed!(rng, 1) # seed

x_data, y_data = prepare_data(df_inf, df_cov, df_pkparams_eleveld) # Read training data

# Epochs, learning rate and optimizer
n_epochs_train = 5000 # Nbr of epochs to train at start + after pruning of several parameters at once
n_epochs_prune = 5000 # Nbr of epochs to train after pruning of one parameter
n_prune_init = 10 # First remove several parameters at once, to save time.
n_prune_byone = 21 # Prune one parameter at a time, for n_prune_byone times

n_init = 8 # Number of initialisations to run
opt = Adam(0.005f0) # Optimizer

@syms x_age x_wgt x_gdr # Symbolics, matches inputs. For printing readable expressions

"""
function main(; n_epochs_train=n_epochs_train, n_epochs_prune=n_epochs_prune, n_prune_init=n_prune_init, n_prune_byone=n_prune_byone, n_init=n_init, seed=1)

# Arguments:
No arguments are needed, but the following can be specified:
- `n_epochs_train`: Int. Nbr of epochs to train before start of pruning.
- `n_epochs_prune`: Int. Nbr of epochs to train between parameter pruning steps.
- `n_prune_init`: Int. Nbr of parameters to prune at once, before pruning one parameter at a time.
- `n_prune_byone`: Int. Nbr of parameters to prune one at a time, after pruning several parameters at once.
- `n_init`: Int. Nbr of initialisations to run.
- `seed`: Int. Seed for random number generator.

Returns params, masks, model, training_loss, final_loss

Writes final SNNs with parameters and masks to file "jld2/covmodel_threadi.jld2" with name `model`.

"""
function main(; n_epochs_train=n_epochs_train, n_epochs_prune=n_epochs_prune, n_prune_init=n_prune_init, n_prune_byone=n_prune_byone, n_init=n_init, seed=1)

    Threads.@threads for i = 1:n_init
        println("Initialisation #", i)
        params, masks, model, _, final_loss = @time covmodeling(x_data, y_data, seed + i, opt, n_epochs_train, n_epochs_prune, n_prune_init, n_prune_byone) # Find covariate expressions

        # Save result to jld2 file 
        filename = "jld2/covmodel_thread$i.jld2"
        jldsave(filename; params, masks, model, final_loss, i)

        @syms x_age x_wgt x_gdr # Symbolics, matches inputs.

        # Print result as readable expression
        _, _ = getexpressions(params, masks) # Print readable expressions, covariates inputs are scaled to be 0->1, outputs V1 and CL have correct scaling
        # getpredictionerrors(x_data, y_data, params, masks, model) # Get prediction errors, compared to the White model
        # plotpredictions(x_data, y_data, best_params, best_masks, best_model) # Plot predictions and observations in log-log scale
    end
end

# main(n_epochs_train=10, n_epochs_prune=1, n_prune_init=1, n_prune_byone=1, n_init=2) # For testing

main()
