import RecipesBase: @recipe, @series, @userplot
import DataFrames: AbstractDataFrame, propertynames

## Plot recipes. Load `Plots` (or another RecipesBase consumer) to use
## them:
##   plot(mf)               trace plot of a `mif2` result, as R `pomp`'s
##                          `plot` method for `mif2d_pomp`
##   plot([mf1, mf2, ...])  the same, one line per object
##   sliceplot(df)          the output of `slice`
##   mcapplot(m)            an `MCAP`

mif2_panels(
    mfs::AbstractVector{<:Mif2dPompObject},
    pars,
) = begin
    isempty(mfs) && error("no `mif2` results to plot.")
    m1 = mfs[1]
    pnames = pars === nothing ?
        collect(keys(merge(m1.rw_sd,m1.rw_sd_init))) :
        collect(Symbol,pars)
    for p ∈ pnames
        p ∈ keys(coef(m1)) || error("`$p` is not a parameter of the model.")
    end
    panels = Vector{Tuple{String,Vector{Vector{Float64}},Vector{Vector{Float64}},String,Symbol}}()
    push!(panels,(
        "eff.sample.size",
        [collect(Float64,times(m)) for m ∈ mfs],
        [collect(Float64,eff_sample_size(m)) for m ∈ mfs],
        "time",:log10,
    ))
    push!(panels,(
        "cond.logLik",
        [collect(Float64,times(m)) for m ∈ mfs],
        [collect(Float64,cond_logLik(m)) for m ∈ mfs],
        "time",:identity,
    ))
    its = [[Float64(t.iteration) for t ∈ traces(m)] for m ∈ mfs]
    push!(panels,(
        "logLik",its,
        [[Float64(t.logLik) for t ∈ traces(m)] for m ∈ mfs],
        "MIF iteration",:identity,
    ))
    if any(m -> m.Nmonitor > 0,mfs)
        push!(panels,(
            "monitor_logLik",its,
            [[Float64(t.monitor_logLik) for t ∈ traces(m)] for m ∈ mfs],
            "MIF iteration",:identity,
        ))
    end
    for p ∈ pnames
        push!(panels,(
            string(p),its,
            [[Float64(getfield(t,p)) for t ∈ traces(m)] for m ∈ mfs],
            "MIF iteration",:identity,
        ))
    end
    panels
end

@recipe function f(mf::Mif2dPompObject; pars = nothing)
    panels = mif2_panels([mf],pars)
    np = length(panels)
    nc = np ≤ 4 ? 1 : 2
    nr = ceil(Int,np/nc)
    layout --> (nr,nc)
    size --> (500*nc,180*nr)
    legend --> false
    for (i,(title,xs,ys,xl,ysc)) ∈ enumerate(panels)
        for (x,y) ∈ zip(xs,ys)
            @series begin
                subplot := i
                seriestype := :path
                ylabel := title
                xlabel := xl
                yscale := ysc
                x, y
            end
        end
    end
end

@recipe function f(mfs::AbstractVector{<:Mif2dPompObject}; pars = nothing)
    panels = mif2_panels(mfs,pars)
    np = length(panels)
    nc = np ≤ 4 ? 1 : 2
    nr = ceil(Int,np/nc)
    layout --> (nr,nc)
    size --> (500*nc,180*nr)
    legend --> false
    for (i,(title,xs,ys,xl,ysc)) ∈ enumerate(panels)
        for (x,y) ∈ zip(xs,ys)
            @series begin
                subplot := i
                seriestype := :path
                ylabel := title
                xlabel := xl
                yscale := ysc
                x, y
            end
        end
    end
end

"""
    sliceplot(df)

Plot the output of [`slice`](@ref): one panel per sliced parameter,
showing the log-likelihood estimates against the parameter, with error
bars from the `se` column when present. Requires `Plots` to be loaded.
"""
@userplot SlicePlot

@recipe function f(sp::SlicePlot)
    length(sp.args) == 1 && sp.args[1] isa AbstractDataFrame ||
        error("`sliceplot` takes the data frame returned by `slice`.")
    df = sp.args[1]
    :slice ∈ propertynames(df) && :loglik ∈ propertynames(df) ||
        error("`sliceplot`: the data frame needs `slice` and `loglik` columns; use `slice_design` and `slice`.")
    vars = unique(df.slice)
    layout --> (1,length(vars))
    size --> (400*length(vars),350)
    legend --> false
    for (i,v) ∈ enumerate(vars)
        sub = df[df.slice .== v,:]
        @series begin
            subplot := i
            seriestype := :scatter
            xlabel := string(v)
            ylabel := "loglik"
            if :se ∈ propertynames(sub)
                yerror := [isfinite(s) ? s : 0.0 for s ∈ sub.se]
            end
            collect(Float64,sub[!,v]), collect(Float64,sub.loglik)
        end
    end
end

"""
    mcapplot(m::MCAP)

Plot an [`mcap`](@ref) result: the profile points, the smooth and the
quadratic fit, the point estimate, the confidence interval, and the
log-likelihood cutoff that defines it. Requires `Plots` to be loaded.
"""
@userplot MCAPPlot

@recipe function f(mp::MCAPPlot)
    length(mp.args) == 1 && mp.args[1] isa MCAP ||
        error("`mcapplot` takes an `MCAP` object.")
    m = mp.args[1]
    top = maximum(m.fit.smoothed)
    lo = min(minimum(m.logLik),top-m.delta)
    margin = 0.1*(top-lo)
    xlabel --> "parameter"
    ylabel --> "logLik"
    ylims --> (lo-margin,top+margin)
    legend --> :bottomright
    @series begin
        seriestype := :scatter
        label := "estimates"
        m.parameter, m.logLik
    end
    @series begin
        seriestype := :path
        label := "smoothed"
        m.fit.parameter, m.fit.smoothed
    end
    @series begin
        seriestype := :path
        linestyle := :dash
        label := "quadratic"
        m.fit.parameter, m.fit.quadratic
    end
    @series begin
        seriestype := :vline
        label := "MLE"
        [m.mle]
    end
    @series begin
        seriestype := :vline
        linestyle := :dot
        label := "$(round(Int,100*m.level))% CI"
        [m.ci[1],m.ci[2]]
    end
    @series begin
        seriestype := :hline
        linestyle := :dot
        label := "cutoff"
        [top-m.delta]
    end
end
