# Study demographics of the data set

using Pkg
cd(@__DIR__)
Pkg.activate(".")

using CSV, DataFrames

df_cov = CSV.read("csv/covariatedata_white.csv", DataFrame)

ages = df_cov.AGE
weights = df_cov.WGT
gdrs = df_cov.GDR # Male 1, Female 2

min_age = minimum(ages)
max_age = maximum(ages)
min_weight = minimum(weights)
max_weight = maximum(weights)
nbr_male = sum(gdrs .== 1)
nbr_female = sum(gdrs .== 2)