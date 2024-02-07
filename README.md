# symbolic-neural-networks-NLME
Code for paper "Symbolic neural networks for automated covariate modeling in a mixed-effects framework", written by Jesper Sundell, Ylva Wahlquist and Kristian Soltesz.

## Run the code
The code is written in Julia and the NLME modeling was done with Pumas.jl.

Start by cloning the repo
```
git clone https://github.com/jespersundell/symbolic-neural-networks-NLME/tree/main
```

### Covariate modeling
Install julia and instantiate the environment by running `julia` and in the Julia REPL, run:
```julia 
include("src-covmod/CovModelingWhite.jl")
```
and you will obtain the results from the covariate modeling that was done with Lux.jl.

### NLME modeling
Use Pumas.jl and in the Julia REPL, run:
```julia 
include("NLME-models/NLME_models.jl")
```
and you will obtain the results from the NLME modeling.

