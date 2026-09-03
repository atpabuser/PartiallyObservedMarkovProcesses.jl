import DataFrames: DataFrame

melt(x::Nothing) = DataFrame()

melt(x::NamedTuple; id...) = DataFrame(;merge(NamedTuple(id),x)...)

melt(x::Any; id...) = melt((value=x,);id...)

"""
    melt(x::AbstractArray, names::Symbol...)

Convert an array into a data frame, using the variables named in `names` as identifiers.  If `x` is an array of `NamedTuple`s, the tuples will be arranged into columns of the resulting data frame.
"""
melt(
    x::AbstractArray{T,N},
    names::Vararg{Symbol,N},
) where {T,N} = begin
    c = NamedTuple{Tuple(names)}.(Tuple.(CartesianIndices(axes(x))))
    reduce(vcat,[melt(x[i];c[i]...) for i ∈ eachindex(x)])
end

melt(x::AbstractArray{T,N}) where {T,N} =
    melt(x,(Symbol("id",i) for i ∈ 1:N)...)

"""
    melt(x::AbstractPompObject; id...)

Convert an `AbstractPompObject` to a data frame, with columns for time, observables, and latent states, if present.
"""
melt(x::AbstractPompObject; id...) = begin
    timevar = pomp(x).timevar
    t = NamedTuple{(timevar,)}.(times(x))
    n = length(t)
    s = isnothing(states(x)) ? fill((;),n) : states(x)
    d = isnothing(obs(x)) ? fill((;),n) : obs(x)
    DataFrame(
        map(
            (t,d,s) -> merge((;id...),t,d,s),
            t,d,s
        )
    )
end

"""
    melt(x::PfilterdPompObject; id...)

Convert a `PfilterdPompObject` to a data frame, with columns for time, observables, and latent states, if present, as well as effective sample size and conditional log likelihood.
"""
melt(x::PfilterdPompObject; id...) =
    hcat(
        melt(pomp(x);id...),
        DataFrame(
            ess=x.eff_sample_size,
            cond_logLik=x.cond_logLik
        )
    )

"""
    melt(x::WpfilterdPompObject; id...)

Convert a `WpfilterdPompObject` to a data frame, with columns for time,
observables, and latent states, if present, as well as effective sample
size, conditional log likelihood, and whether resampling occurred.
"""
melt(x::WpfilterdPompObject; id...) =
    hcat(
        melt(pomp(x);id...),
        DataFrame(
            ess=x.eff_sample_size,
            cond_logLik=x.cond_logLik,
            resampled=x.resampled
        )
    )

"""
    melt(x::Mif2dPompObject; id...)

Convert a `Mif2dPompObject` to a data frame giving the iterated-filtering
trace: one row per IF2 iteration (including iteration 0, the starting
point), with columns for the iteration number, the perturbed-model log
likelihood, and the parameter estimate.
"""
melt(x::Mif2dPompObject; id...) =
    DataFrame(map(r -> merge(NamedTuple(id),r), traces(x)))
