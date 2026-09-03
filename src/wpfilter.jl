struct WpfilterdPompObject{
    T <: Time,
    X <: NamedTuple,
    P <: PompObject{T,X},
    W <: AbstractFloat
    } <: AbstractPompObject
    pompobj::P
    Np::Int
    trigger::W
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

pomp(object::WpfilterdPompObject) = object.pompobj

logLik(object::WpfilterdPompObject) = object.logLik
eff_sample_size(object::WpfilterdPompObject) = object.eff_sample_size
cond_logLik(object::WpfilterdPompObject) = object.cond_logLik

"""
    resampled(object)

`resampled` extracts the vector of booleans recording, for each
observation time, whether resampling was triggered.
"""
resampled(object::WpfilterdPompObject) = object.resampled

"""
    wpfilter(object; Np = 1, trigger = 1, params, rinit, rprocess, logdmeasure, kwargs...)

`wpfilter` runs a particle filter in which resampling occurs only when
triggered by deficiency in the effective sample size (ESS), rather than
at every observation time as in [`pfilter`](@ref). Resampling is
triggered at an observation time whenever the ESS there falls below
`trigger*Np`. `trigger` must lie in ``[0,1]``: `trigger = 1` resamples
at (essentially) every step, reproducing `pfilter`; `trigger = 0` never
resamples.

When a step does not resample, its log-weights are carried forward and
combined with the next step's incremental log-weights, so that
`cond_logLik` always sums to a valid estimate of the overall
log-likelihood, regardless of when resampling occurs.

At least the `rinit`, `rprocess`, and `logdmeasure` basic components are
needed. `kwargs...` can be used to modify or unset additional fields.
"""
wpfilter(
    object::ValidPompData;
    Np::Integer = 1,
    trigger::Real = 1,
    params::P = coef(object),
    rinit::Union{Function,Nothing,Missing} = missing,
    rprocess::Union{PompPlugin,Nothing,Missing} = missing,
    logdmeasure::Union{Function,Nothing,Missing} = missing,
    kwargs...,
) where {P<:NamedTuple} = begin
    @assert Np ≥ 1 "`Np` must be a positive integer."
    if !(0 ≤ trigger ≤ 1)
        error("`trigger` must lie in [0,1].")
    end
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
    x0 = POMP.rinit(object;t0,nsim=Np)
    xf = similar(x0,N,Np)
    xp = similar(x0,N,Np)
    xt = similar(x0,N)
    w = similar(Array{LogLik},N,Np)
    cond_logLik = similar(w,N)
    eff_sample_size = similar(w,N)
    wcarry = similar(w,Np)
    fill!(wcarry,0)
    cumw = similar(w,Np)
    resamp = similar(Array{Bool},N)
    perm = similar(Array{Int},N,Np)
    wpfilter_internal!(
        object,
        x0,
        reshape(xf,N,1,Np),
        reshape(xp,N,1,Np),
        reshape(w,N,1,Np,1),
        t0,t,
        reshape(y,N,1,1),
        wcarry,cumw,
        eff_sample_size,
        cond_logLik,
        resamp,
        perm,
        trig,
    )
    i = trace_ancestry!(xt,xf,perm,wcarry,cumw)
    WpfilterdPompObject(
        PompObject(object,init_state=x0[i],states=xt),
        Np,trig,vec(x0),xf,xp,w,wcarry,
        eff_sample_size,
        cond_logLik,
        resamp,
        sum(cond_logLik)
    )
end

"""
    wpfilter(object; Np = object.Np, trigger = object.trigger, kwargs...)

Running `wpfilter` on a `WpfilterdPompObject` re-runs the filter.
One can adjust the parameters, number of particles (`Np`), resampling
`trigger`, or pomp model components.
"""
wpfilter(
    object::WpfilterdPompObject;
    Np::Integer = object.Np,
    trigger::Real = object.trigger,
    kwargs...,
) = wpfilter(pomp(object; kwargs...); Np, trigger)

wpfilter(_...) = error("Incorrect call to `wpfilter`.")

wpfilter_internal!(
    object::AbstractPompObject,
    x0::AbstractArray{X,2},
    xf::AbstractArray{X,3},
    xp::AbstractArray{X,3},
    w::AbstractArray{LogLik,4},
    t0::T,
    t::AbstractArray{T,1},
    y::AbstractArray{Y,3},
    wcarry::AbstractArray{W,1},
    cumw::AbstractArray{W,1},
    eff_sample_size::AbstractArray{W,1},
    cond_logLik::AbstractArray{W,1},
    resamp::AbstractArray{Bool,1},
    perm::AbstractArray{I,2},
    trigger::W,
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
        wpfilt_step_comps!(
            @view(cond_logLik[k]),
            @view(eff_sample_size[k]),
            @view(resamp[k]),
            wcarry,
            @view(w[k,1,:,1]),
            cumw,
            @view(perm[k,:]),
            @view(xp[k,1,:]),
            @view(xf[k,1,:]),
            trigger,
        )
        t0 = t[k]
        x0 = view(xf,k,:,:)
    end
    nothing
end

"""
    wpfilt_step_comps!(logLik, ess, resamp, wcarry, logw, cumw, p, xp, xf, trigger, n = length(wcarry))

Perform one step of the ESS-triggered particle filter.

`wcarry` holds, on entry, the log-weights carried forward from the
previous step, normalized so that `sum(exp.(wcarry)) == n` (all zero
when the previous step resampled); it is updated in place for the next
call. `logw` holds the incremental log-weights on entry and, on exit,
the total (unnormalized) log-weights `wcarry_in .+ logw_in`. `p` is
filled with ancestry indices -- the identity permutation when no
resampling occurs, so that ancestry tracing remains valid uniformly.
Resampling is triggered whenever the effective sample size falls below
`trigger*n`.
"""
wpfilt_step_comps!(
    logLik::AbstractArray{W,0},
    ess::AbstractArray{W,0},
    resamp::AbstractArray{Bool,0},
    wcarry::AbstractArray{W,1},
    logw::AbstractArray{W,1},
    cumw::AbstractArray{W,1},
    p::AbstractArray{I,1},
    xp::AbstractArray{X,1},
    xf::AbstractArray{X,1},
    trigger::W,
    n::Integer = length(wcarry),
) where {W<:AbstractFloat,I<:Integer,X<:NamedTuple} = begin
    vmax::W = -Inf
    for k ∈ eachindex(logw)
        @inbounds logw[k] += wcarry[k]
        @inbounds vmax = (logw[k] > vmax) ? logw[k] : vmax
    end
    if isfinite(vmax)
        s::W = 0
        ss::W = 0
        for k ∈ eachindex(logw)
            @inbounds v::W = exp(logw[k]-vmax)
            s += v
            ss += v*v
            @inbounds cumw[k] = s
        end
        ess[] = s*s/ss
        logLik[] = vmax+log(s/n)
        if ess[] < trigger*n
            resamp[] = true
            du::W = s/n
            u::W = -du*rand(LogLik)
            i::I = 1
            for j ∈ eachindex(p)
                u += du
                @inbounds while (u > cumw[i] && i < n)
                    i += 1
                end
                @inbounds p[j] = i
            end
            @inbounds xf[:] = xp[p]
            for k ∈ eachindex(wcarry)
                @inbounds wcarry[k] = 0
            end
        else
            resamp[] = false
            for j ∈ eachindex(p)
                @inbounds p[j] = j
            end
            @inbounds xf[:] = xp[:]
            for k ∈ eachindex(wcarry)
                @inbounds wcarry[k] = logw[k]-logLik[]
            end
        end
    else
        ess[] = 0
        logLik[] = W(-Inf)
        resamp[] = false
        p[:] = collect(eachindex(p))
        @inbounds xf[:] = xp[:]
        for k ∈ eachindex(wcarry)
            @inbounds wcarry[k] = 0
        end
    end
    nothing
end

"""
    trace_ancestry!(traj, filt, perm, w, cumw)

Weighted variant of ancestry tracing, used when the final particle
cloud is not necessarily equally weighted (as can occur under
[`wpfilter`](@ref) with `trigger < 1`). The trajectory is initiated by
a draw proportional to `exp.(w)` rather than a uniform draw.
"""
trace_ancestry!(
    traj::AbstractArray{X,1},
    filt::AbstractArray{X,2},
    perm::AbstractArray{I,2},
    w::AbstractArray{W,1},
    cumw::AbstractArray{W,1},
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
            @inbounds v::W = exp(w[k]-wmax)
            s += v
            @inbounds cumw[k] = s
        end
        u::W = s*rand(LogLik)
        jj::I = 1
        while (u > cumw[jj] && jj < n)
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

pretty_string(object::WpfilterdPompObject) = begin
    pretty_string(pomp(object)) *
        ", Np=$(object.Np)" *
        ", trigger=$(round(object.trigger,digits=2))" *
        ", logLik=$(round(object.logLik,digits=2))"
end
