import DataFrames: DataFrame, AbstractDataFrame, DataFrameRow, nrow, metadata, propertynames

## Replicated likelihood evaluation, likelihood slices, and profile
## likelihood computations, following the pattern of R `phylopomp`'s
## `mtbd2_loglik` and `mtbd2_slice` and the R `pomp` profile workflow
## (`profile_design`, `mif2`, replicated `pfilter`, `mcap`).

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
## silently ignored.
design_params(object::AbstractPompObject, row::DataFrameRow) = begin
    base = coef(object)
    pnames = keys(base)
    cols = filter(c -> c != :slice, propertynames(row))
    bad = filter(c -> c ∉ pnames, cols)
    isempty(bad) ||
        error("design column(s) $(join(map(string,bad),", ")) are not parameters of the model.")
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

"""
    profile(object, design; Nmif, Np, rw_sd, nreps = 1, Np_eval = Np, kwargs...)

Profile likelihood computation. For every row of `design` (typically
the output of [`profile_design`](@ref)), run [`mif2`](@ref) from that
row's parameter values with the given `Nmif`, `Np`, and `rw_sd`, then
estimate the log likelihood at the resulting point estimate by
[`pfilter_loglik`](@ref) with `Np_eval` particles and `nreps`
replicates. The profiled parameters are held fixed by leaving them out
of `rw_sd`; if `design` carries the `"profiled"` metadata written by
`profile_design`, naming one of them in `rw_sd` or `rw_sd_init` is an
error. Remaining keyword arguments (`transform`, `cooling_fraction_50`,
`rw_sd_init`, and so on) are passed to `mif2`. Rows are run in parallel
when Julia has more than one thread.

Returns a `DataFrame` with one column per model parameter: the
perturbed parameters carry the point estimates from each run, the
others their values from `design` (or from `object`, for parameters the
design does not mention), followed by `loglik`, `se`, and `ess`
columns. Feed the profiled column and `loglik` to
[`mcap`](@ref) to obtain the Monte Carlo adjusted profile.
"""
profile(
    object::AbstractPompObject,
    design::AbstractDataFrame;
    Nmif::Integer,
    Np::Integer,
    rw_sd::NamedTuple,
    nreps::Integer = 1,
    Np_eval::Integer = Np,
    kwargs...,
) = begin
    n = nrow(design)
    n > 0 || error("`profile`: `design` has no rows.")
    perturbed = keys(rw_sd)
    if haskey(kwargs,:rw_sd_init)
        perturbed = (perturbed...,keys(kwargs[:rw_sd_init])...)
    end
    md = metadata(design)
    if haskey(md,"profiled")
        for p ∈ md["profiled"]
            p ∈ perturbed &&
                error("`profile`: profiled parameter `$p` must not be perturbed; remove it from `rw_sd`.")
        end
    end
    ps = [design_params(object,design[i,:]) for i ∈ 1:n]
    ests = Vector{NamedTuple}(undef,n)
    res = Vector{LoglikSummary}(undef,n)
    flexmap!(1:n) do i
        mf = mif2(object;Nmif,Np,rw_sd,params=ps[i],kwargs...)
        θ = coef(mf)
        ests[i] = θ
        res[i] = pfilter_loglik(pomp(mf);Np=Np_eval,nreps,params=θ)
    end
    ## Perturbed parameters take their estimates; every other parameter
    ## is unchanged by construction and keeps its exact starting value
    ## (the cloud average would otherwise leave floating-point residue).
    out = DataFrame(design;copycols=true)
    for p ∈ keys(coef(object))
        if p ∈ perturbed
            out[!,p] = [Float64(getfield(ests[i],p)) for i ∈ 1:n]
        elseif p ∉ propertynames(out)
            out[!,p] = [Float64(getfield(ps[i],p)) for i ∈ 1:n]
        end
    end
    with_loglik(out,res)
end
