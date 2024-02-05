## Pumas model codes

using DataFrames, CSV
using Random
using Pumas
using PumasUtilities
using DataFramesMeta
using CairoMakie

## Read data set
moddat = CSV.read("white_data.csv", DataFrame)

## Convert data set to object which is readable to Pumas
pop_SNN = read_pumas(
  moddat;
  id = :ID,
  time = :TIME,
  rate = :RATE,
  evid = :EVID,
  covariates = [:AGE, :WGT, :M1F2],
  observations = [:DV],
  amt = :AMT,
  cmt = :CMT,
)

## Base model
basemod = @model begin
    @param begin
      ## Fixed effects ##################################
      θCl ∈ RealDomain(lower = 0.0)
      θVc ∈ RealDomain(lower = 0.0)
      θQ2 ∈ RealDomain(lower=0.0)
      θV2 ∈ RealDomain(lower=0.0)
      θQ3 ∈ RealDomain(lower=0.0)
      θV3 ∈ RealDomain(lower=0.0)
      σ ∈ RealDomain(lower = 0.0)
      ## Random effects ##################################
      ωCl ∈ RealDomain(lower = 0)
      ωVc ∈ RealDomain(lower = 0)
      ωQ2 ∈ RealDomain(lower = 0)
      ωV2 ∈ RealDomain(lower = 0)
    end
    @random begin
      ηCl ~ Normal(0.0, ωCl)
      ηVc ~ Normal(0.0, ωVc)
      ηQ2 ~ Normal(0.0, ωQ2)
      ηV2 ~ Normal(0.0, ωV2)
    end
    @covariates begin
      AGE
      WGT
      M1F2
    end
    @pre begin
      Vc_cov = θVc
      Vc  = Vc_cov * exp(ηVc)

      Cl_cov = θCl
      Cl = Cl_cov * exp(ηCl)

      Q2_cov = θQ2
      Q2 = Q2_cov * exp(ηQ2)

      V2_cov = θV2
      V2 = V2_cov * exp(ηV2)
  
      Q3_cov = θQ3
      Q3  = Q3_cov
  
      V3_cov = θV3
      V3 = V3_cov

      k12 = Q2/Vc
      k21 = Q2/V2
      k13 = Q3/Vc
      k31 = Q3/V3
    end
    @dynamics begin
      Central'     = -((Cl/Vc)+k12+k13)*Central + k21*Peripheral + k31*Peripheral2
      Peripheral'  = k12*Central - k21*Peripheral
      Peripheral2' = k13*Central - k31*Peripheral2
    end
    @derived begin
      conc = @. Central / Vc
      DV ~ @. Normal(conc, (conc*σ))
    end
end

## Initial estimates base model
param_base = (
## Fixed effects
   θCl = 1.7,
   θVc = 7,
   θQ2 = 1.2,
   θV2 = 7,
   θQ3 = 0.7,
   θV3 = 80,
   σ = 0.14,
## random effects
    ωVc = 0.6,  
    ωCl = 0.3,
    ωQ2 = 0.6,
    ωV2 = 0.9
)
## Optimize parameters of base model
fit_base = fit(basemod, pop_SNN, param_base, Pumas.FOCEI())

### White Covariate model
covmod = @model begin
    @param begin
      ## Fixed effects ##########################################
      θ1 ∈ RealDomain(lower=0.0)
      θ2 ∈ RealDomain(lower=-5.0, upper=5.0)
      θ3 ∈ RealDomain(lower=0.0)
      θ4 ∈ RealDomain(lower=-5.0, upper=5.0)
      θ5 ∈ RealDomain(lower=0.0)
      θ6 ∈ RealDomain(lower=-5.0, upper=5.0)
      θ7 ∈ RealDomain(lower=0.0)
      θ8 ∈ RealDomain(lower=-5.0, upper=5.0)
      θQ2 ∈ RealDomain(lower=0.0)
      θV2 ∈ RealDomain(lower=0.0)
      θQ3 ∈ RealDomain(lower=0.0)
      θV3 ∈ RealDomain(lower=0.0)
      σ ∈ RealDomain(lower = 0.0)
      ## Random effects ######################################
      ωCl ∈ RealDomain(lower = 0)
      ωVc ∈ RealDomain(lower = 0)
      ωQ2 ∈ RealDomain(lower = 0)
      ωV2 ∈ RealDomain(lower = 0)
    end
    @random begin
      ηCl ~ Normal(0.0, ωCl)
      ηVc ~ Normal(0.0, ωVc)
      ηQ2 ~ Normal(0.0, ωQ2)
      ηV2 ~ Normal(0.0, ωV2)
    end
    @covariates begin
      AGE
      WGT
      M1F2
    end
    @pre begin
      Clm = (θ1 + (θ2 *AGE) ) * WGT
      Clf = (θ3 + (θ4 *AGE) ) * WGT
      Cl_cov = ifelse(M1F2 == 1, Clm, Clf)
      Cl = Cl_cov * exp(ηCl)
  
      Vcm = (θ5 + (θ6 *AGE) ) * WGT
      Vcf = (θ7 + (θ8 *AGE) ) * WGT
      Vc_cov = ifelse(M1F2 == 1, Vcm, Vcf)
      Vc  = Vc_cov * exp(ηVc)
  
      Q2_cov = θQ2
      Q2 = Q2_cov * exp(ηQ2)
  
      V2_cov = θV2
      V2 = V2_cov * exp(ηV2)
  
      Q3_cov = θQ3
      Q3  = Q3_cov
  
      V3_cov = θV3
      V3 = V3_cov
  
      k12 = Q2/Vc
      k21 = Q2/V2
      k13 = Q3/Vc
      k31 = Q3/V3
    end
    @dynamics begin
      Central'     = -((Cl/Vc)+k12+k13)*Central + k21*Peripheral + k31*Peripheral2
      Peripheral'  = k12*Central - k21*Peripheral
      Peripheral2' = k13*Central - k31*Peripheral2
    end
    @derived begin
      conc = @. Central / Vc
      DV ~ @. Normal(conc, (conc*σ))
    end
end

## Initial estimates White model
param_cov = (
## Fixed effects
  θ1 = 0.02,
  θ2 = 9.42E-06,
  θ3 = 0.03,
  θ4 = 0.0001,
  θ5 = 0.1,
  θ6 = 0.0001,
  θ7 = 0.09,
  θ8 = -1.03E-05,
  θQ2 = 1.2,
  θV2 = 8,
  θQ3 = 1,
  θV3 = 100,
  σ = 0.15,
## random effects
  ωVc = 0.5,  
  ωCl = 0.2,
  ωQ2 = 0.5,
  ωV2 = 0.8
)
## Optimize parameters of White model
fit_cov = fit(covmod, pop_SNN, param_cov, Pumas.FOCEI())

## SNN model
SNN_mod = @model begin
    @param begin
      ## Fixed effects ##########################################
      θ1 ∈ RealDomain()
      θ2 ∈ RealDomain()
      θ3 ∈ RealDomain(lower= 0.0)
      θ4 ∈ RealDomain()
      θ5 ∈ RealDomain()
      θQ2 ∈ RealDomain(lower=0.0)
      θV2 ∈ RealDomain(lower=0.0)
      θQ3 ∈ RealDomain(lower=0.0)
      θV3 ∈ RealDomain(lower=0.0)
      σ ∈ RealDomain(lower = 0.0)
      ## Random effects ######################################
      ωCl ∈ RealDomain(lower = 0)
      ωVc ∈ RealDomain(lower = 0)
      ωQ2 ∈ RealDomain(lower = 0)
      ωV2 ∈ RealDomain(lower = 0)
    end
    @random begin
      ηCl ~ Normal(0.0, ωCl)
      ηVc ~ Normal(0.0, ωVc)
      ηQ2 ~ Normal(0.0, ωQ2)
      ηV2 ~ Normal(0.0, ωV2)
    end
    @covariates begin
      AGE
      WGT
      M1F2
    end
    @pre begin
      Vcm = θ1 * WGT
      Vcf = θ2 * WGT
      Vc_cov = ifelse(M1F2 == 1, Vcm, Vcf)
      Vc  = Vc_cov * exp(ηVc)
  
      Cl_cov = θ3 + (θ4 * AGE) + (θ5 * WGT)
      Cl = Cl_cov * exp(ηCl)
  
      Q2_cov = θQ2
      Q2 = Q2_cov * exp(ηQ2)
  
      V2_cov = θV2
      V2 = V2_cov *exp(ηV2)
  
      Q3_cov = θQ3
      Q3  = Q3_cov
  
      V3_cov = θV3
      V3 = V3_cov
  
      k12 = Q2/Vc
      k21 = Q2/V2
      k13 = Q3/Vc
      k31 = Q3/V3
    end
    @dynamics begin
      Central'     = -((Cl/Vc)+k12+k13)*Central + k21*Peripheral + k31*Peripheral2
      Peripheral'  = k12*Central - k21*Peripheral
      Peripheral2' = k13*Central - k31*Peripheral2
    end
    @derived begin
      conc = @. Central / Vc
      DV ~ @. Normal(conc, (conc*σ))
    end
end

## Initial estimates SNN model  
pSNN = (
  ## Fixed effects
  θ1 = 0.09,
  θ2 = 0.06,
  θ3 = 0.01,
  θ4 = 0.001,
  θ5 = 0.02,
  θQ2 = 1,
  θV2 = 8,
  θQ3 = 1.2,
  θV3 = 100,
  σ = 0.12,
  ## random effects
  ωVc = 0.5,  
  ωCl = 0.3,
  ωQ2 = 0.5,
  ωV2 = 0.5 
)
## Optimize parameters of SNN model
fit_COV_mod2 = fit(SNN_mod, pop_SNN, pSNN, Pumas.FOCEI())
