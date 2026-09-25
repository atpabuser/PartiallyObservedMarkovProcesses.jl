module PartiallyObservedMarkovProcessesAoGExt

using PartiallyObservedMarkovProcesses
import PartiallyObservedMarkovProcesses as POMP
import PartiallyObservedMarkovProcesses: sliceplot, mcapplot, traceplot
using AlgebraOfGraphics
import AlgebraOfGraphics: Scatter, Lines, Errorbars, VLines, HLines
import DataFrames: DataFrame, AbstractDataFrame, propertynames, nrow

## default `draw` options, overridden by the caller's
drawit(spec, defaults, kwargs) = draw(spec; merge(defaults,values(kwargs))...)

sliceplot(df::AbstractDataFrame; kwargs...) = begin
    @assert :slice ∈ propertynames(df) && :loglik ∈ propertynames(df) "the data frame needs `slice` and `loglik` columns"
    se = :se ∈ propertynames(df)
    d = DataFrame(
        parameter=[string(r.slice) for r ∈ eachrow(df)],
        value=[Float64(r[r.slice]) for r ∈ eachrow(df)],
        loglik=df.loglik,
        se=se ? df.se : fill(NaN,nrow(df)),
    )
    x = :value => "parameter value"
    y = :loglik => "log likelihood"
    spec = data(d) * mapping(x,y,layout=:parameter) * visual(Scatter)
    if se && any(isfinite,d.se)
        e = d[isfinite.(d.se),:]
        spec += data(e) * mapping(x,y,:se,layout=:parameter) * visual(Errorbars)
    end
    drawit(spec,(facet=(;linkxaxes=:none),),kwargs)
end

sliceplot(_...; __...) = error("`sliceplot` takes the data frame returned by `slice`.")

mcapplot(m::MCAP; kwargs...) = begin
    ll = :loglik => "log likelihood"
    spec = data((parameter=m.parameter,loglik=m.logLik)) *
        mapping(:parameter,ll) * visual(Scatter)
    spec += data(m.fit) *
        mapping(:parameter,:smoothed => "log likelihood",color=direct("smoothed")) *
        visual(Lines)
    spec += data(m.fit) *
        mapping(:parameter,:quadratic => "log likelihood",color=direct("quadratic")) *
        visual(Lines;linestyle=:dash)
    spec += mapping([m.mle]) * visual(VLines)
    if all(isfinite,m.ci)
        spec += mapping(collect(m.ci)) * visual(VLines;linestyle=:dot)
        spec += mapping([maximum(m.fit.smoothed)-m.delta]) * visual(HLines;linestyle=:dot)
    end
    drawit(spec,(;),kwargs)
end

mcapplot(_...; __...) = error("`mcapplot` takes an `MCAP` object.")

## traces in long format: one row per run, iteration, and panel
trace_data(mfs, pars, monitors) = begin
    @assert !isempty(mfs) "no `mif` results to plot"
    m1 = mfs[1]
    pnames = isnothing(pars) ?
        collect(POMP.perturbed_names(m1.perturbations,[coef(m1)],length(times(m1)))) :
        collect(Symbol,pars)
    for p ∈ pnames
        @assert p ∈ keys(coef(m1)) "`$p` is not a parameter of the model"
    end
    if !isnothing(monitors)
        @assert length(monitors) == length(mfs) "give one `monitor` data frame per `mif` result"
    end
    run = String[]; iteration = Int[]; variable = String[]; value = Float64[]
    add!(k,its,v,vals) = for (i,x) ∈ zip(its,vals)
        ismissing(x) && continue
        push!(run,string(k)); push!(iteration,i); push!(variable,v); push!(value,x)
    end
    for (k,m) ∈ enumerate(mfs)
        t = traces(m)
        add!(k,t.iteration,"logLik",t.logLik)
        isnothing(monitors) || add!(k,monitors[k].iteration,"monitor logLik",monitors[k].loglik)
        for p ∈ pnames
            add!(k,t.iteration,string(p),t[!,p])
        end
    end
    order = ["logLik"; isnothing(monitors) ? String[] : ["monitor logLik"]; string.(pnames)]
    DataFrame(;run,iteration,variable,value), order
end

traceplot(
    mfs::AbstractVector{<:POMP.MifdPompObject};
    pars = nothing,
    monitor = nothing,
    kwargs...,
) = begin
    monitors = monitor isa AbstractDataFrame ? [monitor] : monitor
    d, order = trace_data(mfs,pars,monitors)
    spec = data(d) *
        mapping(:iteration,:value => "",color=:run,layout=:variable => sorter(order)) *
        visual(Lines)
    drawit(spec,(facet=(;linkyaxes=:none),),kwargs)
end

traceplot(mf::POMP.MifdPompObject; kwargs...) = traceplot([mf]; kwargs...)

traceplot(_...; __...) = error("`traceplot` takes the result of `mif`, or a vector of them.")

end
