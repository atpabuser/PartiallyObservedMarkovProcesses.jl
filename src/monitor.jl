import DataFrames: DataFrame, metadata!, nrow
import Random: default_rng, seed!

## Additions on top of `mif`: a hyperbolic cooling schedule, and a
## monitor that evaluates the unperturbed likelihood along an
## iterated-filtering trace. Neither changes `mif` itself.

"""
    hyperbolic_cooling(frac; start = 0)

Returns a hyperbolic cooling schedule, for use as the `cooling`
argument of [`mif`](@ref), under which the perturbations are at a
fraction `frac` of their original magnitude after 50 iterations. At
iteration `i` (counting from `i = 0`, as `mif` does), the perturbation
scale is
```math
c(i) = \\frac{a}{a+i}, \\qquad a = \\frac{50\\,\\mathrm{frac}}{1-\\mathrm{frac}},
```
so that ``c(0) = 1`` and ``c(50) = \\mathrm{frac}``; `frac = 1` gives
no cooling. The scale decays like ``1/i`` rather than geometrically,
so late iterations keep more of their perturbation than under
[`geometric_cooling`](@ref).

The scale changes only between iterations. R `pomp`'s hyperbolic
cooling also cools within an iteration, observation by observation, so
the two schedules agree in their end points but are not identical.

**Continuation.** A `mif` computation that is continued (by calling
`mif` on its result) calls the schedule from `i = 0` again. To
continue a computation that has already performed `k` iterations on
the same schedule, pass `start = k`: the schedule then returns
``c(i+k)``. This lines up the cooling only. The continued computation
still restarts its particle swarm from the previous point estimate, so
`k` iterations followed by `m` more is not the same computation as one
run of `k+m` iterations.
"""
hyperbolic_cooling(
    frac::AbstractFloat;
    start::Integer = 0,
) = begin
    frac = Float64(frac)
    @assert 0 < frac ≤ 1 "`frac` must be ∈ (0,1]"
    @assert start ≥ 0 "`start` must be non-negative"
    if frac == 1
        n -> 1.0
    else
        a = 50*frac/(1-frac)
        n -> a/(a+n+start)
    end
end

"""
    resampled(object)

For the result of a [`pfilter`](@ref) or [`mif`](@ref) computation,
returns a vector with one entry per observation time, `true` where
the particles were resampled after that observation.
"""
resampled(object::PfilterdPompObject) = object.resample
resampled(object::MifdPompObject) = resampled(object.pfobj)

"""
    monitor(runs; Np, seed, nreps = 1, every = 1)

The unperturbed log likelihood along an iterated-filtering trace, for
use as a diagnostic. `runs` is the result of a [`mif`](@ref)
computation, or a vector of results in which each continues the one
before it. At every `every`-th iteration of the combined trace, and at
the last, the log likelihood at that iteration's point estimate is
estimated afresh by [`pfilter_loglik`](@ref) with `Np` particles and
`nreps` replicates.

This is a different quantity from the `logLik` column of
[`traces`](@ref), which is the likelihood of the perturbed model that
`mif` filters. It is computed after the fact, from the recorded point
estimates, so it cannot change the fit. It uses its own random numbers,
generated from `seed`, and restores the state of the default
random-number generator on exit, so calling it does not change any
later computation either.

Returns a `DataFrame` with columns `iteration` (numbered as in
[`traces`](@ref), where row 1 is the starting point, and counted on
across all the runs), `loglik`, `se`, and `ess`, followed by the parameters at which
each evaluation was made. The settings `Np`, `nreps`, `every`, and
`seed` are recorded in the data frame's metadata.

This is a diagnostic of the fitting process. The log likelihood
reported for a fitted point (for example by [`profile`](@ref)) should
come from its own replicated evaluation, not from this trace.
"""
monitor(
    runs::AbstractVector{<:MifdPompObject};
    Np::Integer,
    seed::Integer,
    nreps::Integer = 1,
    every::Integer = 1,
) = begin
    isempty(runs) && error("`monitor`: no `mif` results given.")
    every ≥ 1 || error("`monitor`: `every` must be positive.")
    pnames = keys(coef(runs[1]))
    ## Combined trace: each continuation starts from the final estimate
    ## of the run before it. Its iteration-0 row is the mean of a freshly
    ## perturbed swarm around that estimate, not a completed iteration,
    ## so it is dropped and the iteration count runs on.
    points = NamedTuple[]
    iters = Int[]
    for (k,m) ∈ enumerate(runs)
        tr = traces(m)
        for j ∈ (k == 1 ? 1 : 2):nrow(tr)
            push!(points,NamedTuple{pnames}(Tuple(tr[j,p] for p ∈ pnames)))
            push!(iters,isempty(iters) ? 1 : iters[end]+1)
        end
    end
    keep = [i for i ∈ eachindex(iters) if (iters[i]-1) % every == 0 || i == lastindex(iters)]
    object = pomp(runs[end])
    rng = default_rng()
    saved = copy(rng)
    res = try
        seed!(seed)
        [pfilter_loglik(object;Np,nreps,params=points[i]) for i ∈ keep]
    finally
        copy!(rng,saved)
    end
    out = DataFrame(
        iteration=iters[keep],
        loglik=[r.loglik for r ∈ res],
        se=[r.se for r ∈ res],
        ess=[r.ess for r ∈ res],
    )
    for p ∈ pnames
        out[!,p] = [points[i][p] for i ∈ keep]
    end
    for (k,v) ∈ pairs((Np=Np,nreps=nreps,every=every,seed=seed))
        metadata!(out,string(k),v,style=:note)
    end
    out
end

monitor(run::MifdPompObject; kwargs...) = monitor([run]; kwargs...)
