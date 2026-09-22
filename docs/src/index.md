# PartiallyObservedMarkovProcesses.jl

The package is a Julia implementation of the [pomp package for R](https://kingaa.github.io/pomp/).

## Package Features

- [Implementation of POMP models](@ref)
- [Simulation](@ref)
- [Particle filter](@ref)
- [Iterated filtering](@ref)
- [Likelihood slices and profiles](@ref)
- [Plotting](@ref)
- [Trajectory matching](@ref)
- [Workhorses](@ref) (low-level interface to basic model components)
- [Helper functions](@ref)
- [Reproducibility tools](@ref)

## Function Documentation

### Implementation of POMP models

#### Basic constructor

```@docs
pomp
```

#### `rprocess` plugins

```@docs
euler
discrete_time
onestep
vectorfield
```

### Simulation

```@docs
simulate
simulate_array
```

### Particle filter

```@docs
pfilter
```

### Iterated filtering

```@docs
mif2
traces
geometric_cooling
```

### Likelihood slices and profiles

The workflow follows R `pomp` and R `phylopomp`: build a design with
`slice_design` or `profile_design`, evaluate it with `slice` or
`profile` (which run `pfilter`, or `mif2` followed by `pfilter`, at
every row), and summarize a profile with `mcap`.

```@docs
slice_design
profile_design
runif_design
sobol_design
pfilter_loglik
slice
profile
mcap
MCAP
```

### Plotting

Plot recipes are provided through `RecipesBase`; load `Plots` to use
them. `plot(mf)` on a `mif2` result (or on a vector of them) draws the
convergence diagnostics that R `pomp`'s `plot` method draws: effective
sample size and conditional log likelihood over time for the last
iteration, then the log likelihood and each perturbed parameter against
the iteration number. Pass `pars = (:a, :b)` to select parameters.
`sliceplot(df)` plots the output of `slice`, one panel per sliced
parameter. `mcapplot(m)` plots an `MCAP`: the points, the smooth, the
quadratic fit, the point estimate, the confidence interval, and the
cutoff.

### Trajectory matching

```@docs
traj_match_objfun
```

### Workhorses

```@docs
rinit
rinit!
```

```@docs
rprocess
rprocess!
```

```@docs
rmeasure
```

```@docs
logdmeasure
logdmeasure!
```

```@docs
logdprior
logdprior!
```

```@docs
rprior
```

### Helper functions

```@docs
coef
obs
states
init_state
times
timezero
melt
logmeanexp
```

### Reproducibility tools

```@docs
@freeze
@bake
```
