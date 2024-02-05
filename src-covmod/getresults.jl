# Load jld2 files and analyze result

# male = -0.5, female = 0.5
# Scaling factors for age and weight are 88 years and 100 kg, respectively.

# Resulting expressions for V1 and CL has a maximum of 10 parameter each

using Pkg
cd(@__DIR__)
Pkg.activate("..")

using CSV, DataFrames, JLD2, Lux, Optimisers, Plots, Polynomials, Random, StaticArrays, Zygote

include("DenseMaskLayer.jl") # Custom masked layer
include("covmodelfunctions.jl") # Functions for covariate modeling
include("fastpksim.jl") # Simulator for PK model

############################################################################################################

# Read data
df_inf = CSV.read("csv/infusiondata_white.csv", DataFrame) # Infusion data for White data set
df_cov = CSV.read("csv/covariatedata_white.csv", DataFrame) # Covariate data for White data set
df_pkparams_eleveld = CSV.read("csv/pkparams_eleveld.csv", DataFrame) # PK parameters for Eleveld data set, for scaling purposes only

## Get data
rng = Random.default_rng()
Random.seed!(rng, 1) # seed

x_data, y_data = prepare_data(df_inf, df_cov, df_pkparams_eleveld)


## Open file and read result
f = jldopen("jld2/covmodel_final.jld2", "r+") # Open file

params = f["params"]
masks = f["masks"]
# model = f["model"] # Do not use this, use model structure directly from setupmodel()
final_loss = f["final_loss"]
close(f)
model, _, _ = setupmodel() # Setup model for 6 PK parameters

## Get expressions and prediction errors
@syms x_age x_wgt x_gdr # Symbolics, matches inputs.
fcts_expr, readable_expr = getexpressions(params, masks) # Get readable expressions, covariates inputs are scaled to be 0->1, outputs V1 and CL have correct scaling

getpredictionerrors(x_data, y_data, params, masks, model) # Get prediction errors, compared to the White model

plotpredictions(x_data, y_data, params, masks, model) # Plot predictions and observations in log-log scale

## Get final expressions
V2s = fcts_expr[2](1, 1, 1) ./ 1000 # [litres]
V3s = fcts_expr[3](1, 1, 1) ./ 1000 # [litres]
Q2s = fcts_expr[5](1, 1, 1) ./ 1000 * 60 # [l/min]
Q3s = fcts_expr[6](1, 1, 1) ./ 1000 * 60 # [l/min]

# Check V1_male and V1_female. Note that input is scaled to 0->1!
x_wgts = LinRange(0.01, 1, 100)
v1fs = fcts_expr[1].(ones(length(x_wgts)),x_wgts, 0.5*ones(length(x_wgts)))./1000
plot(x_wgts, v1fs)
v1ms = fcts_expr[1].(ones(length(x_wgts)),x_wgts, -0.5*ones(length(x_wgts)))./1000
plot!(x_wgts, v1ms)

# After simplification in Maple, we get the correctly scaled model (inputs and outputs are correctly scaled)
wgts = LinRange(40, 160, 100)
v1male(wgt) = (-22.61503780 - 0.09422565851 * wgt + 0.00007926406026 * wgt^2) / (-3.041537594 + 0.002608768274 * wgt)
plot(wgts, v1male.(wgts),label="Male", xlabel="Weight [kg]", ylabel="V1 volume [litres]")
v1female(wgt) = (22.61503780 - 0.09422565851 * wgt - 0.00007926406026 * wgt^2) / (3.041537594 + 0.002608768274 * wgt)
plot!(wgts, v1female.(wgts), label="Female", xlabel="Weight [kg]", ylabel="V1 volume [litres]")


## Check CL
# Two separate plot where we vary age and wgt. Note that input is scaled to 0->1!

# First, fix age to 0.5 and vary weight. Plot
x_age = 0.5
x_wgts = LinRange(0.01, 1, 100)
clmale_1 = fcts_expr[4].(x_age, x_wgts, 0.5) ./ 1000 * 60
clfemale_1 = fcts_expr[4].(x_age, x_wgts, -0.5)./1000 * 60
plot(x_wgts, clmale_1)
plot!(x_wgts, clfemale_1)

# Then, fix weight to 0.5 and vary age. Plot
x_wgt = 0.5
x_ages = LinRange(0.01, 1, 100)
clmale_2 = fcts_expr[4].(x_ages, x_wgt, 0.5) ./ 1000 * 60
clfemale_2 = fcts_expr[4].(x_ages, x_wgt, -0.5)./1000 * 60
plot(x_ages, clmale_2)
plot!(x_ages, clfemale_2)