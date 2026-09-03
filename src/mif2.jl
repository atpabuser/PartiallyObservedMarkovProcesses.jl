"""
    Mif2dPompObject

The result of an [`mif2`](@ref) computation. Holds the final parameter
cloud (`paramcloud`, on the natural scale; `estcloud`, on the estimation
scale) and its final carried log-weights (`logweights`), the settings
used, and the trace of per-iteration point estimates ([`traces`](@ref)).

The diagnostics `eff_sample_size`, `cond_logLik`, and `resampled` refer
to the *final* IF2 iteration only.
"""
struct Mif2dPompObject{
    T <: Time,
    X <: NamedTuple,
    P <: PompObject{T,X},
    W <: AbstractFloat,
    Q <: NamedTuple,
    E <: NamedTuple,
    R <: NamedTuple,
    } <: AbstractPompObject
    pompobj::P
    Nmif::Int
    Np::Int
    rw_sd::NamedTuple
    rw_sd_init::NamedTuple
    cooling_type::Symbol
    cooling_fraction_50::W
    trigger::W
    transform::Function
    inverse_transform::Function
    paramcloud::Vector{Q}
    estcloud::Vector{E}
    logweights::Vector{W}
    traces::Vector{R}
    cond_logLik::Vector{W}
    eff_sample_size::Vector{W}
    resampled::Vector{Bool}
    perturbed_logLik::W
    monitor_logLik::W
    Nmonitor::Int
end

pomp(object::Mif2dPompObject) = object.pompobj

"""
    traces(object)

`traces` extracts the vector of per-iteration records (iteration number,
perturbed-model log likelihood, and parameter point estimate -- and,
when available, the monitored log likelihood) from a [`Mif2dPompObject`](@ref).
"""
traces(object::Mif2dPompObject) = object.traces

eff_sample_size(object::Mif2dPompObject) = object.eff_sample_size
cond_logLik(object::Mif2dPompObject) = object.cond_logLik
resampled(object::Mif2dPompObject) = object.resampled

"""
    logLik(object::Mif2dPompObject)

Unlike `logLik(::PfilterdPompObject)`, an IF2 run does not by itself give
a clean likelihood estimate at its point estimate: the perturbed-model
log likelihood recorded in `traces(object)` is the likelihood of the
*perturbed* model, not of the focal model. If `object` was fit with
`Nmonitor > 0`, `logLik` returns the averaged unperturbed-filter estimate
recorded at the final iteration; otherwise it errors, directing the user
to `pfilter`.
"""
logLik(object::Mif2dPompObject) = begin
    if object.Nmonitor > 0
        object.monitor_logLik
    else
        error(
            "A `mif2` computation does not by itself yield a clean likelihood "*
            "estimate at its point estimate: the log likelihood recorded in "*
            "`traces(object)` is the so-called mif log likelihood, which is "*
            "the log likelihood of the perturbed model, not of the focal "*
            "model. To obtain the latter, run `logLik(pfilter(object; Np=...))` "*
            "(better: `logmeanexp` over several such runs), or re-run `mif2` "*
            "with `Nmonitor > 0`."
        )
    end
end

pretty_string(object::Mif2dPompObject) = begin
    pretty_string(pomp(object)) *
        ", Nmif=$(object.Nmif)" *
        ", Np=$(object.Np)"
end

## ---------------------------------------------------------------- cooling

cooling_setup(cooling_type::Symbol, α::Real, N::Integer) = begin
    if cooling_type === :geometric
        zero(float(α))
    elseif cooling_type === :hyperbolic
        (50*N*α-1)/(1-α)
    else
        error("`cooling_type` must be `:geometric` or `:hyperbolic`.")
    end
end

cooling(
    cooling_type::Symbol,
    m::Integer, n::Integer, N::Integer,
    α::Real, s::Real,
) = begin
    if α == 1
        one(float(α))
    elseif cooling_type === :geometric
        α^((n-1+(m-1)*N)/(50*N))
    else
        (s+1)/(s+n+(m-1)*N)
    end
end

## ------------------------------------------------------------ perturbation

perturb!(
    est::AbstractVector{E},
    sd::NamedTuple,
    c::Real,
) where {E<:NamedTuple} = begin
    ks = keys(sd)
    if !isempty(ks)
        for i ∈ eachindex(est)
            @inbounds est[i] = merge(
                est[i],
                NamedTuple{ks}(
                    ntuple(
                        d -> getproperty(est[i],ks[d])+c*getproperty(sd,ks[d])*randn(),
                        length(ks)
                    )
                )
            )::E
        end
    end
    nothing
end

## ------------------------------------------------------------- point estimate

weighted_mean(
    est::AbstractVector{E},
    w::AbstractVector{W},
) where {E<:NamedTuple,W<:AbstractFloat} = begin
    ks = keys(est[1])
    wmax::W = maximum(w)
    tot::W = 0
    acc = zeros(W,length(ks))
    for i ∈ eachindex(est)
        u::W = exp(w[i]-wmax)
        tot += u
        for d ∈ eachindex(ks)
            @inbounds acc[d] += u*getproperty(est[i],ks[d])
        end
    end
    NamedTuple{ks}(ntuple(d -> acc[d]/tot, length(ks)))
end

## -------------------------------------------------------- particle advance

## `advance_particles!` (in pfilter.jl) parallelizes over the *nsim* axis
## (axis 2), which has length 1 in mif2's layout (Np parameter-particles
## live on axis 1). This routine instead chunks and parallelizes over the
## parameter axis, and materializes each chunk's parameters into a plain
## `Vector` (never a view/SubArray -- see the caution below).
advance_particles_mif!(
    object::AbstractPompObject,
    t0::T,
    x0::AbstractArray{X,2},
    xp::AbstractArray{X,3},
    w::AbstractArray{W,4},
    t::AbstractArray{T,1},
    y::AbstractArray{Y,3},
    theta::AbstractVector{Q},
    chunks::AbstractVector{<:AbstractRange},
    thetabufs::AbstractVector{<:AbstractVector{Q}},
) where {W<:AbstractFloat,T<:Time,X<:NamedTuple,Y<:NamedTuple,Q<:NamedTuple} = begin
    flexmap!(eachindex(chunks)) do c
        jj = chunks[c]
        buf = thetabufs[c]
        for (b,j) ∈ enumerate(jj)
            @inbounds buf[b] = theta[j]
        end
        ## NB: `params` must be a genuine `Vector`, never a view: `rprocess!`/
        ## `logdmeasure!` call `val_array(params)`, and a `SubArray` falls
        ## through to `val_array`'s scalar-wrapping fallback, silently
        ## wrapping the whole view as a single "parameter set" and then
        ## failing the length assertions inside those functions.
        rprocess!(object, @view(xp[:,jj,:]); x0=@view(x0[jj,:]), t0, times=t, params=buf)
        logdmeasure!(object, @view(w[:,jj,:,:]); times=t, y=@view(y[:,jj,:]), x=@view(xp[:,jj,:]), params=buf)
    end
    nothing
end

mif_chunks(Np::Integer, Q::Type) = begin
    nchunks = max(1,min(Np,Threads.nthreads()))
    bounds = round.(Int,range(0,Np,length=nchunks+1))
    chunks = [(bounds[c]+1):bounds[c+1] for c ∈ 1:nchunks if bounds[c+1] > bounds[c]]
    thetabufs = [Vector{Q}(undef,length(jj)) for jj ∈ chunks]
    (chunks,thetabufs)
end

## ------------------------------------------------------------- validation

validate_mif2_args(
    params::NamedTuple,
    transform::Function,
    inverse_transform::Function,
    rw_sd::NamedTuple,
    rw_sd_init::NamedTuple,
    cooling_type::Symbol,
    cooling_fraction_50::Real,
    trigger::Real,
    Nmif::Integer,
    Np::Integer,
    Nmonitor::Integer,
    Np_monitor::Integer,
    N::Integer,
) = begin
    est0 = transform(params)
    est0 isa NamedTuple || error("`transform` must return a `NamedTuple`.")
    theta0 = inverse_transform(est0)
    theta0 isa NamedTuple || error("`inverse_transform` must return a `NamedTuple`.")
    keys(theta0) == keys(params) ||
        error("`transform` and `inverse_transform` must be mutual inverses: key mismatch.")
    for k ∈ keys(params)
        v0 = getproperty(params,k)
        v1 = getproperty(theta0,k)
        ok = (v0 isa Real && v1 isa Real) ? isapprox(v0,v1;rtol=1e-8,atol=1e-8) : v0==v1
        ok || error("`transform`/`inverse_transform` round-trip failed for parameter `$k`.")
    end
    rwk = keys(rw_sd)
    rwik = keys(rw_sd_init)
    isempty(setdiff(rwk,keys(est0))) ||
        error("`rw_sd` names not found among the (transformed) parameters: $(setdiff(rwk,keys(est0))).")
    isempty(setdiff(rwik,keys(est0))) ||
        error("`rw_sd_init` names not found among the (transformed) parameters: $(setdiff(rwik,keys(est0))).")
    isempty(intersect(rwk,rwik)) ||
        error("`rw_sd` and `rw_sd_init` must not share parameter names: $(intersect(rwk,rwik)).")
    for k ∈ rwk
        v = getproperty(rw_sd,k)
        (v isa Real && isfinite(v) && v ≥ 0) ||
            error("`rw_sd.$k` must be a finite, nonnegative real number.")
    end
    for k ∈ rwik
        v = getproperty(rw_sd_init,k)
        (v isa Real && isfinite(v) && v ≥ 0) ||
            error("`rw_sd_init.$k` must be a finite, nonnegative real number.")
    end
    for k ∈ union(rwk,rwik)
        getproperty(est0,k) isa AbstractFloat ||
            error("parameters to be estimated must be floating-point on the estimation "*
                  "scale; got $(typeof(getproperty(est0,k))) for `$k`.")
    end
    cooling_type ∈ (:geometric,:hyperbolic) ||
        error("`cooling_type` must be `:geometric` or `:hyperbolic`.")
    (0 < cooling_fraction_50 ≤ 1) ||
        error("`cooling_fraction_50` must lie in (0,1].")
    if cooling_type === :hyperbolic && cooling_fraction_50 < 1
        cooling_fraction_50 > 1/(50*N) ||
            error("`cooling_fraction_50` is too small for hyperbolic cooling with "*
                  "$N observations; it must exceed $(1/(50*N)).")
    end
    (0 ≤ trigger ≤ 1) || error("`trigger` must lie in [0,1].")
    Nmif ≥ 0 || error("`Nmif` must be a nonnegative integer.")
    Np ≥ 1 || error("`Np` must be a positive integer.")
    Nmonitor ≥ 0 || error("`Nmonitor` must be a nonnegative integer.")
    Np_monitor ≥ 1 || error("`Np_monitor` must be a positive integer.")
    nothing
end

## ----------------------------------------------------------------- engine

mif2_run(
    object::AbstractPompObject,
    m0::Integer,
    est::Vector{E},
    tr0::Vector{R0},
    Nmif::Integer,
    Np::Integer,
    rw_sd::NamedTuple,
    rw_sd_init::NamedTuple,
    cooling_type::Symbol,
    cooling_fraction_50::Real,
    transform::Function,
    inverse_transform::Function,
    trigger::Real,
    Nmonitor::Integer,
    Np_monitor::Integer,
) where {E<:NamedTuple,R0<:NamedTuple} = begin
    t0_0 = timezero(object)
    t = times(object)
    y = obs(object)
    N = length(t)
    trig = LogLik(trigger)
    α = Float64(cooling_fraction_50)
    s_cool = cooling_setup(cooling_type,α,N)

    theta = [inverse_transform(est[i]) for i ∈ eachindex(est)]
    Q = eltype(theta)
    estbuf = similar(est)
    wcarry = zeros(LogLik,Np)
    cumw = similar(wcarry)
    perm = zeros(Int,Np)
    cll = fill(LogLik(NaN),N)
    ess = fill(LogLik(NaN),N)
    resamp = fill(false,N)
    chunks,thetabufs = mif_chunks(Np,Q)

    xf = POMP.rinit(object;t0=t0_0,params=theta,nsim=1) # (Np,1); establishes X
    xp = similar(xf,1,Np,1)
    ell4 = Array{LogLik}(undef,1,Np,1,1)
    ybuf = Array{eltype(y)}(undef,1,Np,1)

    traces_ = copy(tr0)
    monitor_ll = LogLik(NaN)

    for m ∈ (m0+1):(m0+Nmif)
        fill!(wcarry,0)
        t0 = t0_0
        for n ∈ 1:N
            c = cooling(cooling_type,m,n,N,α,s_cool)
            if n==1 && !isempty(rw_sd_init)
                perturb!(est,merge(rw_sd,rw_sd_init),c)
            else
                perturb!(est,rw_sd,c)
            end
            for i ∈ eachindex(est)
                @inbounds theta[i] = inverse_transform(est[i])
            end
            if n==1
                rinit!(object,xf;t0,params=theta)
            end
            fill!(ybuf,y[n])
            advance_particles_mif!(
                object,t0,xf,xp,ell4,
                @view(t[[n]]),ybuf,theta,chunks,thetabufs,
            )
            wpfilt_step_comps!(
                @view(cll[n]),@view(ess[n]),@view(resamp[n]),
                wcarry,@view(ell4[1,:,1,1]),cumw,perm,
                @view(xp[1,:,1]),@view(xf[:,1]),
                trig,
            )
            if resamp[n]
                for j ∈ eachindex(perm)
                    @inbounds estbuf[j] = est[perm[j]]
                end
                est,estbuf = estbuf,est
            end
            t0 = t[n]
        end
        estbar = weighted_mean(est,wcarry)
        thetabar = inverse_transform(estbar)
        if Nmonitor > 0
            reps = [
                logLik(pfilter(object;Np=Np_monitor,params=thetabar))
                for _ ∈ 1:Nmonitor
            ]
            monitor_ll = logmeanexp(reps)
            push!(traces_,(;iteration=m,logLik=sum(cll),monitor_logLik=monitor_ll,thetabar...))
        else
            push!(traces_,(;iteration=m,logLik=sum(cll),thetabar...))
        end
    end

    thetafinal = inverse_transform(weighted_mean(est,wcarry))

    Mif2dPompObject(
        pomp(object;params=thetafinal),
        m0+Nmif,Np,
        rw_sd,rw_sd_init,
        cooling_type,α,trig,
        transform,inverse_transform,
        [inverse_transform(e) for e ∈ est],
        est,
        wcarry,
        traces_,
        cll,ess,resamp,
        sum(cll),
        monitor_ll,
        Nmonitor,
    )
end

"""
    mif2(
        object;
        Nmif = 1, Np = 1, rw_sd, rw_sd_init = (;),
        params = coef(object),
        cooling_type = :geometric, cooling_fraction_50 = 0.5,
        transform = identity, inverse_transform = identity,
        trigger = 1, Nmonitor = 0, Np_monitor = Np,
        kwargs...,
    )

`mif2` implements the IF2 iterated-filtering algorithm of Ionides et al.
(2015, PNAS 112:719-724) for maximum-likelihood parameter estimation. It
carries a cloud of `Np` paired (state, parameter) particles through
`Nmif` particle-filtering passes over the data; at each observation time,
before the state is propagated, the parameter particles are perturbed by
a Gaussian random walk (on a user-specified "estimation" scale) whose
standard deviation is cooled geometrically or hyperbolically toward zero
across iterations, following R `pomp`'s `mif2`. States and parameters are
resampled together, using the same ancestry indices.

## Arguments

- `Nmif`: number of IF2 iterations.
- `Np`: number of particles.
- `rw_sd`: a `NamedTuple` of random-walk standard deviations, on the
  estimation scale, for the parameters to be estimated. Parameters not
  named in `rw_sd` (or `rw_sd_init`) are held fixed.
- `rw_sd_init`: as `rw_sd`, but for parameters perturbed only at the
  first observation time (matching R `pomp`'s `ivp`). Its keys must be
  disjoint from those of `rw_sd`.
- `cooling_type`: `:geometric` or `:hyperbolic`.
- `cooling_fraction_50`: the fraction by which the random-walk standard
  deviation shrinks after 50 IF2 iterations.
- `transform`, `inverse_transform`: functions mapping a natural-scale
  parameter `NamedTuple` to and from an unconstrained estimation scale
  (e.g. log/logit), and back. Perturbation is applied on the estimation
  scale. Default to `identity`.
- `trigger`: the ESS-resampling trigger passed to the shared
  [`wpfilt_step_comps!`](@ref) step routine (see [`wpfilter`](@ref)).
  The IF2 theory of Ionides et al. (2015) requires resampling at every
  observation time, so the default is `trigger = 1`; it is exposed as an
  advanced knob for experimentation only.
- `Nmonitor`, `Np_monitor`: if `Nmonitor > 0`, an additional `Nmonitor`
  unperturbed `pfilter` runs (at `Np_monitor` particles) are performed at
  the end of each iteration, and their log-mean-exp is recorded as
  `monitor_logLik` in `traces(object)` -- this gives a clean likelihood
  estimate at the current point estimate, sidestepping the well-known
  fact that the perturbed-model log likelihood is not one (see
  [`logLik`](@ref)`(::Mif2dPompObject)`).

At least the `rinit`, `rprocess`, and `logdmeasure` basic components are
needed. `kwargs...` can be used to modify or unset additional fields.
"""
mif2(
    object::ValidPompData;
    Nmif::Integer = 1,
    Np::Integer = 1,
    rw_sd::NamedTuple,
    rw_sd_init::NamedTuple = (;),
    params::P = coef(object),
    cooling_type::Symbol = :geometric,
    cooling_fraction_50::Real = 0.5,
    transform::Function = identity,
    inverse_transform::Function = identity,
    trigger::Real = 1,
    Nmonitor::Integer = 0,
    Np_monitor::Integer = Np,
    rinit::Union{Function,Nothing,Missing} = missing,
    rprocess::Union{PompPlugin,Nothing,Missing} = missing,
    logdmeasure::Union{Function,Nothing,Missing} = missing,
    kwargs...,
) where {P<:NamedTuple} = begin
    object = pomp(object;params,rinit,rprocess,logdmeasure,kwargs...)
    N = length(times(object))
    validate_mif2_args(
        params,transform,inverse_transform,rw_sd,rw_sd_init,
        cooling_type,cooling_fraction_50,trigger,
        Nmif,Np,Nmonitor,Np_monitor,N,
    )
    est0 = fill(transform(params),Np)
    tr0 = if Nmonitor > 0
        [(;iteration=0,logLik=LogLik(NaN),monitor_logLik=LogLik(NaN),params...)]
    else
        [(;iteration=0,logLik=LogLik(NaN),params...)]
    end
    mif2_run(
        object,0,est0,tr0,
        Nmif,Np,rw_sd,rw_sd_init,
        cooling_type,cooling_fraction_50,
        transform,inverse_transform,
        trigger,Nmonitor,Np_monitor,
    )
end

"""
    mif2(object::Mif2dPompObject; Nmif = 1, kwargs...)

Continues an existing `mif2` computation for `Nmif` additional
iterations, resuming the cooling schedule and the parameter cloud where
the previous run left off (matching R `pomp`'s `continue`). Unlike
`pfilter(::PfilterdPompObject)`, this does *not* start over: to restart
from scratch, call `mif2(pomp(object); kwargs...)`.

`Np` cannot be changed on continuation (the stored particle cloud has a
fixed size); other settings default to the values used previously but
may be overridden.
"""
mif2(
    object::Mif2dPompObject;
    Nmif::Integer = 1,
    Np::Integer = object.Np,
    rw_sd::NamedTuple = object.rw_sd,
    rw_sd_init::NamedTuple = object.rw_sd_init,
    cooling_type::Symbol = object.cooling_type,
    cooling_fraction_50::Real = object.cooling_fraction_50,
    transform::Function = object.transform,
    inverse_transform::Function = object.inverse_transform,
    trigger::Real = object.trigger,
    Nmonitor::Integer = object.Nmonitor,
    Np_monitor::Integer = Np,
    kwargs...,
) = begin
    Np == object.Np ||
        error("`Np` cannot be changed on continuation (got Np=$Np, stored cloud has $(object.Np)).")
    base = pomp(object;kwargs...)
    N = length(times(base))
    params0 = coef(base)
    validate_mif2_args(
        params0,transform,inverse_transform,rw_sd,rw_sd_init,
        cooling_type,cooling_fraction_50,trigger,
        Nmif,Np,Nmonitor,Np_monitor,N,
    )
    mif2_run(
        base,object.Nmif,copy(object.estcloud),copy(object.traces),
        Nmif,Np,rw_sd,rw_sd_init,
        cooling_type,cooling_fraction_50,
        transform,inverse_transform,
        trigger,Nmonitor,Np_monitor,
    )
end

mif2(_...) = error("Incorrect call to `mif2`.")
