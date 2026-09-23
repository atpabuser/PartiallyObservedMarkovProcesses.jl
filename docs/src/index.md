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
mif
@perturbn
@ivp
geometric_cooling
hyperbolic_cooling
```

### Likelihood slices and profiles

The workflow follows R `pomp`: build a design with `slice_design` or
`profile_design`, evaluate it with `slice` (fixed-parameter particle
filters at every row) or `profile` (`mif` from every row, with the
profiled parameters held fixed, then fresh fixed-parameter particle
filters at the estimate), and summarize a profile with `mcap`.
`monitor` evaluates the unperturbed likelihood along a `mif` trace, as
a diagnostic of the fit.

```@docs
slice_design
profile_design
runif_design
sobol_design
pfilter_loglik
slice
profile
mcap
monitor
resampled
```

### Plotting

Plot recipes are provided through `RecipesBase`; load `Plots` to use
them. `plot(mf)` on a `mif` result (or on a vector of them) draws the
convergence diagnostics that R `pomp`'s `plot` method draws: effective
sample size and conditional log likelihood over time for the last
iteration, then the log likelihood and each perturbed parameter against
the iteration number. Pass `pars = (:a, :b)` to select parameters, and
`monitor = df` (the output of `monitor`) to add the unperturbed log
likelihood. `sliceplot(df)` plots the output of `slice`, one panel per
sliced parameter. `mcapplot(m)` plots an `MCAP`: the points, the
smooth, the quadratic fit, the point estimate, the confidence interval,
and the cutoff.

```@docs
sliceplot
mcapplot
```

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
traces
```

### Reproducibility tools

```@docs
@freeze
@bake
```
