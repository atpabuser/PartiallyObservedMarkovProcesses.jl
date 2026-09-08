import Statistics: mean

struct PfilterdPompObject{
    T <: Time,
    X <: NamedTuple,
    P <: PompObject{T,X},
    W <: AbstractFloat
    } <: AbstractPompObject
    pompobj::P
    Np::Int
    trigger::W
    target::W
    x0::Array{X,1}
    filt::Array{X,2}
    pred::Array{X,2}
    weights::Array{W,2}
    logweights::Array{W,1}
    eff_sample_size::Array{W,1}
    cond_logLik::Array{W,1}
    resampled::Array{Bool,1}
    logLik::W
end

pomp(object::PfilterdPompObject) = object.pompobj
logLik(object::PfilterdPompObject) = object.logLik
eff_sample_size(object::PfilterdPompObject) = object.eff_sample_size
cond_logLik(object::PfilterdPompObject) = object.cond_logLik

"""
    resampled(object)

`resampled` extracts the vector of booleans recording, for each
observation time, whether systematic resampling occurred.
"""
resampled(object::PfilterdPompObject) = object.resampled

"""
    pfilter(object; Np = 1, trigger = 1, target = 0, params, rinit,
            rprocess, logdmeasure, kwargs...)

`pfilter` runs a sequential Monte Carlo computation, also known as a
particle filter, giving an unbiased estimate of the likelihood of the
data under the specified partially observed Markov process model.

Systematic resampling occurs at an observation time whenever the
effective sample size there falls to `trigger*Np` or below; `trigger`
must lie in ``[0,1]``. `trigger = 1` resamples at every observation
time; `trigger = 0` never resamples.

When resampling occurs, particles are selected with probability
proportional to ``w^{1-\\text{target}}``, and each selected particle
retains a weight proportional to ``w^{\\text{target}}``, renormalized
to unit mean; `target` must lie in ``[0,1]``. `target = 0` gives the
standard equally weighted resampling; `target = 1` carries the full
weighted particle representation forward across the resampling step.
When a step does not resample, its incremental log weights are carried
forward and combined with the next step's, so that `cond_logLik` always
sums to a valid estimate of the overall log likelihood, regardless of
when resampling occurs.

At least the `rinit`, `rprocess`, and `logdmeasure` basic components are
needed. `kwargs...` can be used to modify or unset additional fields.
"""
pfilter(
    object::ValidPompData;
    Np::Integer = 1,
    trigger::Real = 1,
    target::Real = 0,
    params::P = coef(object),
    rinit::Union{Function,Nothing,Missing} = missing,
    rprocess::Union{PompPlugin,Nothing,Missing} = missing,
    logdmeasure::Union{Function,Nothing,Missing} = missing,
    kwargs...,
) where {P<:NamedTuple} = begin
    Np ≥ 1 || error("`Np` must be a positive integer.")
    (0 ≤ trigger ≤ 1) || error("`trigger` must lie in [0,1].")
    (0 ≤ target ≤ 1) || error("`target` must lie in [0,1].")
    object = pomp(
        object;
        params,rinit,rprocess,logdmeasure,
        kwargs...,
    )
    t0 = timezero(object)
    t = times(object)
    y = obs(object)
    N = length(t)
    trig = LogLik(trigger)
    targ = LogLik(target)
    x0 = POMP.rinit(object;t0,nsim=Np)
    xf = similar(x0,N,Np)
    xp = similar(x0,N,Np)
    xt = similar(x0,N)
    w = similar(Array{LogLik},N,Np)
    cond_logLik = similar(w,N)
    eff_sample_size = similar(w,N)
    resamp = similar(Array{Bool},N)
    perm = similar(Array{Int},N,Np)
    if trig == 1 && targ == 0
        work = similar(Array{LogLik},Np)
        pfilter_internal!(
            object,
            x0,
            reshape(xf,N,1,Np),
            reshape(xp,N,1,Np),
            reshape(w,N,1,Np,1),
            t0,t,
            reshape(y,N,1,1),
            work,
            eff_sample_size,
            cond_logLik,
            resamp,
            perm
        )
        i = trace_ancestry!(xt,xf,perm)
        logweights = zeros(LogLik,Np)
    else
        wcarry = ones(LogLik,Np)
        work = similar(wcarry)
        pfilter_internal!(
            object,
            x0,
            reshape(xf,N,1,Np),
            reshape(xp,N,1,Np),
            reshape(w,N,1,Np,1),
            t0,t,
            reshape(y,N,1,1),
            wcarry,work,
            eff_sample_size,
            cond_logLik,
            resamp,
            perm,
            trig,targ,
        )
        i = trace_ancestry!(xt,xf,perm,wcarry,work)
        logweights = log.(wcarry)
    end
    PfilterdPompObject(
        PompObject(object,init_state=x0[i],states=xt),
        Np,trig,targ,vec(x0),xf,xp,w,logweights,
        eff_sample_size,
        cond_logLik,
        resamp,
        sum(cond_logLik)
    )
end

"""
    pfilter(object; Np = object.Np, trigger = object.trigger,
            target = object.target, kwargs...)

Running `pfilter` on a `PfilterdPompObject` re-runs the particle filter.
One can adjust the parameters, number of particles (`Np`), the
resampling `trigger` and `target`, or pomp model components.
"""
pfilter(
    object::PfilterdPompObject;
    Np::Integer = object.Np,
    trigger::Real = object.trigger,
    target::Real = object.target,
    kwargs...,
) = pfilter(pomp(object; kwargs...); Np, trigger, target)

pfilter(_...) = error("Incorrect call to `pfilter`.")

## fully-resampled (classic bootstrap) engine: trigger=1, target=0
pfilter_internal!(
    object::AbstractPompObject,
    x0::AbstractArray{X,2},
    xf::AbstractArray{X,3},
    xp::AbstractArray{X,3},
    w::AbstractArray{LogLik,4},
    t0::T,
    t::AbstractArray{T,1},
    y::AbstractArray{Y,3},
    work::AbstractArray{W,1},
    eff_sample_size::AbstractArray{W,1},
    cond_logLik::AbstractArray{W,1},
    resamp::AbstractArray{Bool,1},
    perm::AbstractArray{I,2}
) where {W<:AbstractFloat,T<:Time,X<:NamedTuple,Y<:NamedTuple,I<:Integer} = begin
    for k ∈ eachindex(t)
        advance_particles!(
            object,
            t0,x0,
            @view(xp[[k],:,:]),
            @view(w[[k],:,:,:]),
            @view(t[[k]]),
            @view(y[[k],:,:]),
        )
        pfilt_step_comps!(
            @view(cond_logLik[k]),
            @view(eff_sample_size[k]),
            @view(w[k,1,:,1]),
            @view(perm[k,:]),
            @view(xp[k,1,:]),
            @view(xf[k,1,:]),
            work,
            @view(resamp[k]),
        )
        t0 = t[k]
        x0 = view(xf,k,:,:)
    end
    nothing
end

## general weighted engine: carries a linear, unit-mean weight vector
## across observation times, resampling whenever the effective sample
## size falls to `trigger*Np` or below
pfilter_internal!(
    object::AbstractPompObject,
    x0::AbstractArray{X,2},
    xf::AbstractArray{X,3},
    xp::AbstractArray{X,3},
    w::AbstractArray{LogLik,4},
    t0::T,
    t::AbstractArray{T,1},
    y::AbstractArray{Y,3},
    wcarry::AbstractArray{W,1},
    work::AbstractArray{W,1},
    eff_sample_size::AbstractArray{W,1},
    cond_logLik::AbstractArray{W,1},
    resamp::AbstractArray{Bool,1},
    perm::AbstractArray{I,2},
    trigger::W,
    target::W,
) where {W<:AbstractFloat,T<:Time,X<:NamedTuple,Y<:NamedTuple,I<:Integer} = begin
    for k ∈ eachindex(t)
        advance_particles!(
            object,
            t0,x0,
            @view(xp[[k],:,:]),
            @view(w[[k],:,:,:]),
            @view(t[[k]]),
            @view(y[[k],:,:]),
        )
        pfilt_step_comps!(
            @view(cond_logLik[k]),
            @view(eff_sample_size[k]),
            @view(w[k,1,:,1]),
            @view(perm[k,:]),
            @view(xp[k,1,:]),
            @view(xf[k,1,:]),
            wcarry,work,
            trigger,target,
            @view(resamp[k]),
        )
        t0 = t[k]
        x0 = view(xf,k,:,:)
    end
    nothing
end

advance_particles!(
    object::AbstractPompObject,
    t0::T,
    x0::AbstractArray{X,2},
    xp::AbstractArray{X,3},
    w::AbstractArray{W,4},
    t::AbstractArray{T,1},
    y::AbstractArray{Y,3},
) where {W<:AbstractFloat,T<:Time,X<:NamedTuple,Y<:NamedTuple} = begin
    flexmap!(axes(x0,2)) do j
        rprocess!(object, @view(xp[:,:,[j]]); x0=@view(x0[:,[j]]), t0, times=t)
        logdmeasure!(object, @view(w[:,:,[j],:]); times=t, y, x=@view(xp[:,:,[j]]))
    end
    nothing
end

## classic step: resample to equal weights whenever the maximum log
## weight is finite
pfilt_step_comps!(
    logLik::AbstractArray{W,0},
    ess::AbstractArray{W,0},
    logw::AbstractArray{W,1},
    p::AbstractArray{I,1},
    xp::AbstractArray{X,1},
    xf::AbstractArray{X,1},
    work::AbstractArray{W,1},
    resamp::AbstractArray{Bool,0},
    n::Integer = length(logw),
) where {W<:AbstractFloat,I<:Integer,X<:NamedTuple} = begin
    logwmax = compute_ess_logLik!(ess, logLik, logw)
    if isfinite(logwmax)
        systematic_resample!(p, logw, work)
        @inbounds xf .= xp[p]
        resamp[] = true
    else
        p .= collect(eachindex(p))
        xf .= xp
        resamp[] = false
    end
    nothing
end

## general step: resample (to the power `target`) whenever the maximum
## log weight is finite and the effective sample size falls to
## `trigger*n` or below; otherwise carry the full weighted
## representation forward unchanged
pfilt_step_comps!(
    logLik::AbstractArray{W,0},
    ess::AbstractArray{W,0},
    logw::AbstractArray{W,1},
    p::AbstractArray{I,1},
    xp::AbstractArray{X,1},
    xf::AbstractArray{X,1},
    w::AbstractArray{W,1},
    work::AbstractArray{W,1},
    trigger::W,
    target::W,
    resamp::AbstractArray{Bool,0},
    n::Integer = length(logw),
) where {W<:AbstractFloat,I<:Integer,X<:NamedTuple} = begin
    logwmax = compute_ess_logLik!(ess, logLik, logw, w)
    if isfinite(logwmax) && ess[] ≤ trigger*n
        logLik[] += systematic_resample!(p, w, work, target)
        @inbounds xf .= xp[p]
        resamp[] = true
    else
        p .= collect(eachindex(p))
        xf .= xp
        resamp[] = false
    end
    nothing
end

## Computes the effective sample size (`ess`) and conditional log
## likelihood (`logLik`) from the incremental log weights `logw` (the
## classic, equally weighted variant: entry log weights carry no prior
## information).
compute_ess_logLik!(
    ess::AbstractArray{W,0},
    logLik::AbstractArray{W,0},
    logw::AbstractArray{W,1},
) where {W <: AbstractFloat} = begin
    logwmax::W = maximum(logw)
    @assert(
        !isnan(logwmax) && logwmax < Inf,
        "the measurement density returned an invalid (NaN or +∞) log likelihood"
    )
    if isfinite(logwmax)
        s::W = 0
        ss::W = 0
        @inbounds for k ∈ eachindex(logw)
            logw[k] -= logwmax
            v::W = exp(logw[k])
            s += v
            ss += v*v
        end
        lik = s/length(logw)
        ess[] = s*s/ss
        s = log(lik)
        logLik[] = logwmax+s
        logw .-= s
    else
        ess[] = 0
        logLik[] = W(-Inf)
        logw .= zero(W)
    end
    logwmax
end

## As above, but folding in a carried weighted particle representation
## `w`: on entry, `w` must have unit mean, encoding the (linear) weights
## accumulated at earlier observation times; `logw` holds the current
## step's incremental log weights. `w` is folded into `logw` and, on
## exit, `w = exp(logw)` and again has unit mean; the conditional log
## likelihood is `logwmax + log(mean)`.
compute_ess_logLik!(
    ess::AbstractArray{W,0},
    logLik::AbstractArray{W,0},
    logw::AbstractArray{W,1},
    w::AbstractArray{W,1},
) where {W <: AbstractFloat} = begin
    logwmax::W = maximum(logw)
    @assert(
        !isnan(logwmax) && logwmax < Inf,
        "the measurement density returned an invalid (NaN or +∞) log likelihood"
    )
    @assert length(w)==length(logw)
    if isfinite(logwmax)
        s::W = 0
        ss::W = 0
        @inbounds for k ∈ eachindex(logw)
            logw[k] += log(w[k])-logwmax
            v::W = exp(logw[k])
            s += v
            ss += v*v
            w[k] = v
        end
        lik = s/length(w)       # unit-mean assumption is needed here
        ess[] = s*s/ss
        s = log(lik)
        logLik[] = logwmax+s
        logw .-= s
        w ./= lik               # enforces unit-mean on return
    else
        ess[] = 0
        logLik[] = W(-Inf)
        logw .= zero(W)
        w .= one(W)
    end
    logwmax
end

## Classic systematic resampling: selection probability proportional to
## `exp.(logw)`. The indices of the selected particles are returned in
## `p`; `ucum` is working memory.
systematic_resample!(
    p::AbstractArray{I,1},
    logw::AbstractArray{W,1},
    ucum::AbstractArray{W,1},
) where {I,W} = begin
    @assert length(ucum)==length(logw)==length(p)
    s::W = 0
    @inbounds for j ∈ eachindex(logw)
        s += exp(logw[j])
        ucum[j] = s
    end
    n::I = length(logw)
    i::I = 1
    du::W = s/n
    u::W = -du*rand(W)
    @inbounds for j ∈ eachindex(p)
        u += du
        while (u > ucum[i] && i < n)
            i += 1
        end
        p[j] = i
    end
    nothing
end

## Power-tempered systematic resampling: particles are selected with
## probability proportional to `w^(1-β)`, and the retained weight
## following each selected particle is `w^β` (each particle's weight
## factors as w = w^(1-β)·w^β, and the retained factor must accompany
## the particle it came from, not the position it lands in). The
## indices of the selected particles are returned in `p`, and `w` is
## overwritten with the (unit-mean renormalized) retained weights.
## `ucum` is working memory.
##
## The properly weighted representation after selection assigns the
## selected particle the weight w^β·(Σᵢ wᵢ^(1-β))/n; the factor removed
## in renormalizing to unit mean is therefore returned (as a log), to
## be credited to the conditional log likelihood so that the product of
## the conditional likelihoods remains the unbiased likelihood estimate
## for every β. This quantity vanishes at β = 0 and β = 1.
systematic_resample!(
    p::AbstractArray{I,1},
    w::AbstractArray{W,1},
    ucum::AbstractArray{W,1},
    β::W,
) where {I,W} = begin
    @assert length(ucum)==length(w)==length(p)
    s::W = 0
    α = 1-β
    @inbounds for j ∈ eachindex(w)
        s += w[j]^α
        ucum[j] = s
    end
    n::I = length(w)
    i::I = 1
    du::W = s/n
    u::W = -du*rand(W)
    @inbounds for j ∈ eachindex(p)
        u += du
        while (u > ucum[i] && i < n)
            i += 1
        end
        p[j] = i
    end
    @inbounds for j ∈ eachindex(p)
        ucum[j] = w[p[j]]^β
    end
    @inbounds for j ∈ eachindex(w)
        w[j] = ucum[j]
    end
    m::W = mean(w)
    w ./= m # subsequent steps rely on the weights having unit mean
    log(m*s/n)
end

trace_ancestry!(
    traj::AbstractArray{X,1},
    filt::AbstractArray{X,2},
    perm::AbstractArray{I,2},
) where {X,I<:Integer} = begin
    @assert size(traj,1)==size(perm,1)
    @assert size(filt)==size(perm)
    j::I = rand(axes(perm,2))
    for i ∈ Iterators.reverse(axes(perm,1))
        @inbounds traj[i] = filt[i,j]
        @inbounds j = perm[i,j]
    end
    j
end

"""
    trace_ancestry!(traj, filt, perm, w, work)

Weighted variant of ancestral-lineage tracing, used when the final
particle representation is not necessarily equally weighted (as can
occur under [`pfilter`](@ref) with `trigger < 1` or `target > 0`). The
trajectory is initiated by a draw with probability proportional to the
linear weights `w`, rather than a uniform draw. `work` is working
memory.
"""
trace_ancestry!(
    traj::AbstractArray{X,1},
    filt::AbstractArray{X,2},
    perm::AbstractArray{I,2},
    w::AbstractArray{W,1},
    work::AbstractArray{W,1},
) where {X,I<:Integer,W<:AbstractFloat} = begin
    @assert size(traj,1)==size(perm,1)
    @assert size(filt)==size(perm)
    @assert length(w)==size(perm,2)
    n = length(w)
    wmax::W = -Inf
    for k ∈ eachindex(w)
        @inbounds wmax = (w[k] > wmax) ? w[k] : wmax
    end
    j::I = if isfinite(wmax)
        s::W = 0
        for k ∈ eachindex(w)
            @inbounds v::W = w[k]
            s += v
            @inbounds work[k] = s
        end
        u::W = s*rand(LogLik)
        jj::I = 1
        while (u > work[jj] && jj < n)
            jj += 1
        end
        jj
    else
        rand(axes(perm,2))
    end
    for i ∈ Iterators.reverse(axes(perm,1))
        @inbounds traj[i] = filt[i,j]
        @inbounds j = perm[i,j]
    end
    j
end

pretty_string(object::PfilterdPompObject) = begin
    pretty_string(pomp(object)) *
        ", Np=$(object.Np)" *
        ", trigger=$(round(object.trigger,digits=2))" *
        ", target=$(round(object.target,digits=2))" *
        ", logLik=$(round(object.logLik,digits=2))"
end
