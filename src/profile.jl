import DataFrames: DataFrame, AbstractDataFrame, DataFrameRow, nrow, metadata, propertynames
import Random: default_rng

## Replicated likelihood evaluation, likelihood slices, and profile
## likelihood computations, following the pattern of R `phylopomp`'s
## `mtbd2_loglik` and `mtbd2_slice` and the R `pomp` profile workflow
## (`profile_design`, `mif`, replicated `pfilter`, `mcap`).

const LoglikSummary = NamedTuple{(:loglik,:se,:ess),NTuple{3,Float64}}

"""
    pfilter_loglik(object; Np, nreps = 1, params = coef(object), kwargs...)

Estimate the log likelihood of `object` at `params` by `nreps`
independent [`pfilter`](@ref) runs with `Np` particles each, combined
on the likelihood scale by [`logmeanexp`](@ref). Additional keyword
arguments are passed to `pfilter`.

Returns `(loglik, se, ess)`: the log-mean-exp of the replicate
estimates, its jack-knife standard error, and the effective sample size
of the replicate ensemble. Replicates whose estimate is not finite are
dropped; if none is finite, the result is `(loglik=-Inf, se=NaN,
ess=0)`. With a single finite replicate `se` is `NaN`.

This follows R `phylopomp`'s `mtbd2_loglik`. A small `ess` relative to
`nreps`, or a large `se`, indicates that `Np` is too small.
"""
pfilter_loglik(
    object::AbstractPompObject;
    Np::Integer,
    nreps::Integer = 1,
    params::NamedTuple = coef(object),
    kwargs...,
) = begin
    nreps ≥ 1 || error("`pfilter_loglik`: `nreps` must be positive.")
    lls = [logLik(pfilter(object;Np,params,kwargs...)) for _ ∈ 1:nreps]
    summarize_loglik(lls)
end

summarize_loglik(lls::AbstractVector{<:Real})::LoglikSummary = begin
    finite = filter(isfinite,lls)
    if isempty(finite)
        (loglik=-Inf,se=NaN,ess=0.0)
    elseif length(finite) == 1
        (loglik=Float64(finite[1]),se=NaN,ess=1.0)
    else
        est = logmeanexp(finite;se=true,ess=true)
        (loglik=est.est,se=est.se,ess=est.ess)
    end
end

## The parameter set for one row of a design: the row's parameter
## columns laid over the object's own parameters. Non-parameter columns
## other than `slice` are an error, so that a typo in a design is not
## silently ignored. An object that carries no parameters (e.g.
## `gompertz()`) cannot be checked: its design must then supply every
## parameter, and its columns are taken as given.
design_params(object::AbstractPompObject, row::DataFrameRow) = begin
    base = coef(object)
    cols = filter(c -> c != :slice, propertynames(row))
    if !isempty(base)
        bad = filter(c -> c ∉ keys(base), cols)
        isempty(bad) ||
            error("design column(s) $(join(map(string,bad),", ")) are not parameters of the model.")
    end
    merge(base,NamedTuple{Tuple(cols)}(Tuple(Float64(row[c]) for c ∈ cols)))
end

with_loglik(design::AbstractDataFrame, res::AbstractVector{LoglikSummary}) = begin
    out = DataFrame(design;copycols=true)
    out.loglik = [r.loglik for r ∈ res]
    out.se = [r.se for r ∈ res]
    out.ess = [r.ess for r ∈ res]
    out
end

"""
    slice(object, design; Np, nreps = 1, kwargs...)

Evaluate the log likelihood of `object` at every row of `design`,
holding all parameters at the row's values, by
[`pfilter_loglik`](@ref) with `Np` particles and `nreps` replicates.
`design` is typically the output of [`slice_design`](@ref); its
parameter columns override the corresponding parameters of `object`,
and a `slice` column, if present, is carried through. Rows are
evaluated in parallel when Julia has more than one thread.

Returns `design` with `loglik`, `se`, and `ess` columns appended. This
is a likelihood slice, not a profile: the other parameters are held
fixed, not re-maximized. Additional keyword arguments are passed to
`pfilter`.
"""
slice(
    object::AbstractPompObject,
    design::AbstractDataFrame;
    Np::Integer,
    nreps::Integer = 1,
    kwargs...,
) = begin
    n = nrow(design)
    n > 0 || error("`slice`: `design` has no rows.")
    ps = [design_params(object,design[i,:]) for i ∈ 1:n]
    res = Vector{LoglikSummary}(undef,n)
    flexmap!(1:n) do i
        res[i] = pfilter_loglik(object;Np,nreps,params=ps[i],kwargs...)
    end
    with_loglik(design,res)
end


## The names of the parameters that a `mif` perturbations function
## perturbs, found by calling it once at the zero-time (lag 0) and once
## after the first observation (lag 1) and reading the names it
## returns. The calls consume random numbers, so the state of the
## default random-number generator is saved and restored: finding out
## which parameters are perturbed must not change the fit.
perturbed_names(perturbations::Function, params::NamedTuple) = begin
    rng = default_rng()
    saved = copy(rng)
    names = try
        union(
            keys(perturbations(1.0,0;params...)),
            keys(perturbations(1.0,1;params...)),
        )
    finally
        copy!(rng,saved)
    end
    Tuple(names)
end

"""
    profile(object, design; Nmif, Np, perturbations, cooling,
            nreps = 1, Np_eval = Np, kwargs...)

Profile likelihood computation. For every row of `design` (typically
the output of [`profile_design`](@ref)), run [`mif`](@ref) from that
row's parameter values with the given `Nmif`, `Np`, `perturbations`,
and `cooling`, then estimate the log likelihood at the resulting point
estimate by [`pfilter_loglik`](@ref) with `Np_eval` particles and
`nreps` replicates. Remaining keyword arguments (`avfun`, `trigger`,
`target`, and so on) are passed to `mif`. Rows are run in parallel
when Julia has more than one thread.

The parameters estimated at each row are those that `perturbations`
perturbs; every other parameter keeps its exact value from the row. In
particular, the profiled parameters are held fixed by leaving them out
of `perturbations`. If `design` carries the `"profiled"` metadata
written by `profile_design`, a `perturbations` function that perturbs
one of them is an error.

The log likelihood reported for a row is always a fresh evaluation, by
unperturbed particle filters, at exactly the parameter vector reported
in that row. The likelihood that `mif` computes internally is that of
a perturbed model and is not used.

Returns a `DataFrame` with one column per model parameter, followed by
`loglik`, `se`, and `ess` columns. Feed the profiled column and
`loglik` to [`mcap`](@ref) to obtain the Monte Carlo adjusted profile.

**Note.** `mif`'s default point estimate is the geometric mean of the
particle swarm, taken over every parameter; it fails if any parameter
is negative. Pass `avfun = mean` in that case.
"""
profile(
    object::AbstractPompObject,
    design::AbstractDataFrame;
    Nmif::Integer,
    Np::Integer,
    perturbations::Function,
    cooling::Function,
    nreps::Integer = 1,
    Np_eval::Integer = Np,
    kwargs...,
) = begin
    n = nrow(design)
    n > 0 || error("`profile`: `design` has no rows.")
    ps = [design_params(object,design[i,:]) for i ∈ 1:n]
    perturbed = perturbed_names(perturbations,ps[1])
    md = metadata(design)
    if haskey(md,"profiled")
        for p ∈ md["profiled"]
            p ∈ perturbed &&
                error("`profile`: profiled parameter `$p` must not be perturbed; remove it from `perturbations`.")
        end
    end
    ests = Vector{NamedTuple}(undef,n)
    res = Vector{LoglikSummary}(undef,n)
    flexmap!(1:n) do i
        mf = mif(object;Nmif,Np,perturbations,cooling,params=ps[i],kwargs...)
        ## Perturbed parameters take their estimates. Every other
        ## parameter keeps its exact starting value: the swarm average
        ## of a constant need not return that constant exactly, and the
        ## likelihood must be evaluated at the point that is reported.
        est = coef(mf)
        θ = merge(ps[i],NamedTuple{perturbed}(Tuple(getfield(est,p) for p ∈ perturbed)))
        ests[i] = θ
        res[i] = pfilter_loglik(object;Np=Np_eval,nreps,params=θ)
    end
    out = DataFrame(design;copycols=true)
    for p ∈ keys(ps[1])
        if p ∈ perturbed || p ∉ propertynames(out)
            out[!,p] = [Float64(getfield(ests[i],p)) for i ∈ 1:n]
        end
    end
    with_loglik(out,res)
end
