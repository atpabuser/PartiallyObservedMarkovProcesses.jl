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
    cooling::Function          # the resolved schedule, callable as (m,n,N)
    cooling_type::Symbol       # `:custom` when `cooling` was user-supplied
    cooling_fraction_50::W     # `NaN` when `cooling_type === :custom`
    trigger::W
    target::W
    perturb::Union{Function,NamedTuple,Nothing}   # the raw specification
    transform::Function          # to the averaging scale
    inverse_transform::Function  # from the averaging scale
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
    Np_monitor::Int
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

## `SymbolCooling` wraps the two built-in schedules as a callable of
## `(m,n,N)`, so that `mif2`'s engine sees one uniform interface whether
## the schedule was named by a `Symbol` or supplied as a function. The
## setup constant `s` (needed only by the hyperbolic form) is resolved
## once, at construction, exactly as before.
struct SymbolCooling{W<:AbstractFloat} <: Function
    cooling_type::Symbol
    α::W
    s::W
end

(k::SymbolCooling)(m::Integer, n::Integer, N::Integer) =
    cooling(k.cooling_type,m,n,N,k.α,k.s)

"""
    geometric_cooling(cooling_fraction_50, N)
    hyperbolic_cooling(cooling_fraction_50, N)

The two built-in perturbation cooling schedules, as callables of
`(m,n,N)` returning the factor multiplying every random-walk standard
deviation at iteration `m` and observation index `n`. Supplied for
symmetry with a user-written schedule; passing `cooling_type` and
`cooling_fraction_50` to [`mif2`](@ref) is equivalent.
"""
geometric_cooling(α::Real, N::Integer) =
    SymbolCooling(:geometric,Float64(α),Float64(cooling_setup(:geometric,α,N)))

hyperbolic_cooling(α::Real, N::Integer) =
    SymbolCooling(:hyperbolic,Float64(α),Float64(cooling_setup(:hyperbolic,α,N)))

## Resolves the `cooling` / `cooling_type` / `cooling_fraction_50`
## keywords into a single callable schedule, together with the
## `cooling_type` and `cooling_fraction_50` to be recorded on the
## returned object. A user-supplied function is recorded as
## `cooling_type = :custom` with `cooling_fraction_50 = NaN`; that
## sentinel is never subjected to the `(0,1]` check, since the branch is
## taken on the type of the specification rather than on the stored
## value. `:custom` is unreachable from user input, as the `Symbol`
## branch admits only the two built-in names.
normalize_cooling(
    cooling::Union{Symbol,Function,Nothing},
    cooling_type::Union{Symbol,Nothing},
    cooling_fraction_50::Real,
    N::Integer,
) = begin
    (cooling === nothing || cooling_type === nothing) ||
        error("supply either `cooling` or `cooling_type`, not both.")
    spec = cooling === nothing ?
        (cooling_type === nothing ? :geometric : cooling_type) : cooling
    if spec isa Symbol
        spec ∈ (:geometric,:hyperbolic) ||
            error("`cooling_type` must be `:geometric` or `:hyperbolic`.")
        α = Float64(cooling_fraction_50)
        (0 < α ≤ 1) || error("`cooling_fraction_50` must lie in (0,1].")
        if spec === :hyperbolic && α < 1
            α > 1/(50*N) ||
                error("`cooling_fraction_50` is too small for hyperbolic cooling with "*
                      "$N observations; it must exceed $(1/(50*N)).")
        end
        (SymbolCooling(spec,α,Float64(cooling_setup(spec,α,N))),spec,α)
    else
        (spec,:custom,Float64(NaN))
    end
end

## ------------------------------------------------- perturbation kernel

## Resolves the `perturb` keyword into an `RWKernel`. `nothing` gives the
## transform-derived default, which is the historical behaviour and what
## R `pomp` does; a bare function is taken as a natural-scale kernel of
## the whole parameter tuple; a `NamedTuple` names one scalar kernel per
## estimated parameter.
normalize_perturb(perturb::Nothing, to::Function, from::Function, ::NamedTuple) =
    TransformRW(to,from)

normalize_perturb(perturb::Function, ::Function, ::Function, ::NamedTuple) =
    FunctionRW(perturb)

normalize_perturb(perturb::NamedTuple, ::Function, ::Function, sdkeys::NamedTuple) = begin
    for k ∈ keys(perturb)
        getproperty(perturb,k) isa ScalarRW ||
            error("`perturb.$k` must be a per-parameter kernel such as "*
                  "`lognormal_rw()`; got $(typeof(getproperty(perturb,k))).")
    end
    ## Every perturbed parameter needs a kernel, and a kernel for a
    ## parameter that is never perturbed would silently do nothing.
    missing_ = setdiff(keys(sdkeys),keys(perturb))
    isempty(missing_) ||
        error("`perturb` must name every parameter named in `rw_sd`/`rw_sd_init`; "*
              "missing: $(Tuple(missing_)).")
    extra = setdiff(keys(perturb),keys(sdkeys))
    isempty(extra) ||
        error("`perturb` names parameter(s) not named in `rw_sd`/`rw_sd_init`, which "*
              "would never be applied: $(Tuple(extra)).")
    PerKeyRW(perturb)
end

## A kernel maps its declared support into itself (K4), so checking each
## starting value once is enough and no guard is needed in the inner
## loop. This runs before the averaging transformation is composed,
## because that composition would itself fail first on an out-of-support
## value -- taking the logit of a parameter exceeding one, say -- and the
## resulting domain error says much less than this does.
validate_kernel_support(::RWKernel, ::NamedTuple) = nothing

validate_kernel_support(kernel::PerKeyRW, params::NamedTuple) = begin
    for k ∈ keys(kernel.kers)
        kk = getproperty(kernel.kers,k)
        haskey(params,k) ||
            error("`perturb` names parameter `$k`, which is not among the parameters.")
        x = getproperty(params,k)
        in_support(kk,x) ||
            error("the starting value of `$k` is $x, which lies outside the support of "*
                  "its `perturb` kernel: $(typeof(kk)) requires a value "*
                  "$(support_description(kk)).")
    end
    nothing
end

## Resolves the scale on which the particle cloud is averaged to form the
## point estimate. The cloud must be averaged on the scale on which the
## kernel perturbs symmetrically, or a multiplicative perturbation is
## paired with an arithmetic mean and the estimate is biased for a skewed
## cloud. For the per-key form the kernels declare that scale themselves,
## so the composition is exact and the mismatch is unconstructible; for
## an opaque function the user's `transform` is used and the
## responsibility is theirs.
normalize_average(
    perturb::Union{Nothing,Function},
    transform_raw,
    to::Function,
    from::Function,
) = (to,from)

normalize_average(
    perturb::NamedTuple,
    transform_raw,
    to::Function,
    from::Function,
) = begin
    kernel_tags = NamedTuple{keys(perturb)}(map(mean_scale,values(perturb)))
    if transform_raw isa Function
        transform_raw === identity ||
            error("a per-parameter `perturb` cannot be combined with a functional "*
                  "`transform`: the averaging scale is composed from the kernels' "*
                  "own declarations, which cannot be merged into an opaque function "*
                  "pair. Supply `transform` as a `NamedTuple` of tags, or a "*
                  "`ParameterTransform`, or omit it.")
        pt = ParameterTransform(kernel_tags)
        (pt.to,pt.from)
    else
        tags,groups = transform_raw isa ParameterTransform ?
            partrans_spec(transform_raw) : (transform_raw,())
        declared = union(keys(tags),(s for g ∈ groups for s ∈ g))
        clash = intersect(declared,keys(kernel_tags))
        isempty(clash) ||
            error("parameter(s) $(Tuple(clash)) are given both a `perturb` kernel and "*
                  "a `transform` tag; the averaging scale would be doubly specified. "*
                  "Name each estimated parameter in one or the other.")
        pt = ParameterTransform(merge(tags,kernel_tags);logbarycentric=groups)
        (pt.to,pt.from)
    end
end

## --------------------------------------------------------------- rw_sd

"""
    ivp(sd; lags = 1)

Marks a random-walk standard deviation, for use as an entry of `rw_sd`
in [`mif2`](@ref), as applying only immediately before the observation
times whose 1-based index is in `lags` (an integer, or a vector or
range of integers), and as zero at every other observation time --
matching R `pomp`'s `ivp`, generalized to arbitrary lags. `rw_sd_init`
is equivalent to naming the same parameters with `ivp(sd,lags=1)` in
`rw_sd`.
"""
struct Ivp
    sd::Float64
    lags::Union{Int,AbstractVector{<:Integer}}
end

ivp(sd::Real; lags::Union{Integer,AbstractVector{<:Integer}} = 1) =
    Ivp(Float64(sd),lags isa Integer ? Int(lags) : lags)

## Resolves one `rw_sd` entry (a nonnegative real, a per-observation-index
## vector, a function of the observation index, or an `Ivp`) to a
## `Vector{Float64}` of standard deviations, one per observation time,
## validating as it goes.
resolve_rw_sd_entry(v::Real, N::Integer, k::Symbol) = begin
    (isfinite(v) && v ≥ 0) ||
        error("`rw_sd.$k` must be finite, nonnegative at every observation time.")
    fill(Float64(v),N)
end

resolve_rw_sd_entry(v::AbstractVector{<:Real}, N::Integer, k::Symbol) = begin
    length(v)==N ||
        error("`rw_sd.$k` is a vector but does not have length equal to the number "*
              "of observation times ($N); got length $(length(v)).")
    out = Float64.(v)
    all(x -> isfinite(x) && x ≥ 0, out) ||
        error("`rw_sd.$k` must be finite, nonnegative at every observation time.")
    out
end

resolve_rw_sd_entry(f::Function, N::Integer, k::Symbol) = begin
    out = Float64[f(n) for n ∈ 1:N]
    all(x -> isfinite(x) && x ≥ 0, out) ||
        error("`rw_sd.$k` must be finite, nonnegative at every observation time.")
    out
end

resolve_rw_sd_entry(v::Ivp, N::Integer, k::Symbol) = begin
    (isfinite(v.sd) && v.sd ≥ 0) ||
        error("`rw_sd.$k` must be finite, nonnegative at every observation time.")
    out = zeros(Float64,N)
    lags = v.lags isa Integer ? (v.lags:v.lags) : v.lags
    for n ∈ lags
        (1 ≤ n ≤ N) ||
            error("`rw_sd.$k`: an `ivp` lag ($n) falls outside the observation "*
                  "index range 1:$N.")
        out[n] = v.sd
    end
    out
end

## Resolves the full `rw_sd`/`rw_sd_init` specification, once N (the
## number of observation times) is known, to a `Vector` of length N of
## `NamedTuple`s, each giving every perturbed parameter's random-walk
## standard deviation at that observation time. `rw_sd_init` is merged
## in as `ivp(sd,lags=1)` for each of its names.
resolve_sdschedule(rw_sd::NamedTuple, rw_sd_init::NamedTuple, N::Integer) = begin
    merged = merge(rw_sd,NamedTuple{keys(rw_sd_init)}(map(ivp,values(rw_sd_init))))
    ks = keys(merged)
    resolved = NamedTuple{ks}(map(k -> resolve_rw_sd_entry(getproperty(merged,k),N,k),ks))
    [NamedTuple{ks}(ntuple(d -> getproperty(resolved,ks[d])[n],length(ks))) for n ∈ 1:N]
end

## ------------------------------------------------------------- point estimate

## `w` holds linear (not log) nonnegative weights.
weighted_mean(
    est::AbstractVector{E},
    w::AbstractVector{W},
) where {E<:NamedTuple,W<:AbstractFloat} = begin
    ks = keys(est[1])
    tot::W = 0
    acc = zeros(W,length(ks))
    for i ∈ eachindex(est)
        u::W = w[i]
        tot += u
        for d ∈ eachindex(ks)
            @inbounds acc[d] += u*getproperty(est[i],ks[d])
        end
    end
    NamedTuple{ks}(ntuple(d -> acc[d]/tot, length(ks)))
end

## -------------------------------------------------------- particle advance

## ---------------------------------------------------------- transform

## Normalizes the three accepted forms of `mif2`'s `transform` argument
## to a plain `(to,from)` pair of `Function`s. For the two declarative
## forms, the inverse is derived, so a non-default `inverse_transform`
## is rejected.
normalize_transform(transform::Function, inverse_transform::Union{Function,Nothing}) =
    (transform,inverse_transform === nothing ? identity : inverse_transform)

normalize_transform(transform::NamedTuple, inverse_transform::Union{Function,Nothing}) = begin
    inverse_transform === nothing ||
        error("`inverse_transform` must not be supplied when `transform` is a "*
              "`NamedTuple` of tags or a `ParameterTransform`: the inverse is "*
              "derived automatically.")
    pt = ParameterTransform(transform)
    (pt.to,pt.from)
end

normalize_transform(transform::ParameterTransform, inverse_transform::Union{Function,Nothing}) = begin
    inverse_transform === nothing ||
        error("`inverse_transform` must not be supplied when `transform` is a "*
              "`NamedTuple` of tags or a `ParameterTransform`: the inverse is "*
              "derived automatically.")
    (transform.to,transform.from)
end

## For a declarative `transform`, checks that every tagged or
## log-barycentrically-grouped parameter name is among `params`' names;
## an arbitrary function pair is not checked (as it never has been).
validate_partrans_names(transform::Function, params0::NamedTuple) = nothing

validate_partrans_names(transform::NamedTuple, params0::NamedTuple) = begin
    unk = setdiff(keys(transform),keys(params0))
    isempty(unk) ||
        error("`transform` names parameter(s) not present among `params`: $unk.")
    nothing
end

validate_partrans_names(transform::ParameterTransform, params0::NamedTuple) = begin
    tags,groups = partrans_spec(transform)
    declared = union(keys(tags),(s for g ∈ groups for s ∈ g))
    unk = setdiff(declared,keys(params0))
    isempty(unk) ||
        error("`transform` names parameter(s) not present among `params`: $unk.")
    nothing
end

## ------------------------------------------------------------- validation

validate_mif2_args(
    params::NamedTuple,
    transform::Function,
    inverse_transform::Function,
    rw_sd::NamedTuple,
    rw_sd_init::NamedTuple,
    trigger::Real,
    target::Real,
    Nmif::Integer,
    Np::Integer,
    Nmonitor::Integer,
    Np_monitor::Integer,
    N::Integer,
    kernel::RWKernel,
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
    ## The perturbed entries are written back into whichever cloud the
    ## kernel writes, under a `::E`/`::Q` assertion, so they must already
    ## be floating-point on that cloud's scale -- the estimation scale for
    ## the transform-derived default, the natural scale for any kernel
    ## that acts there.
    natural = primary(kernel) === :theta
    scalename = natural ? "natural" : "estimation"
    checked = natural ? params : est0
    for k ∈ union(rwk,rwik)
        getproperty(checked,k) isa AbstractFloat ||
            error("parameters to be estimated must be floating-point on the "*
                  "$scalename scale; got $(typeof(getproperty(checked,k))) for `$k`.")
    end
    ## The cooling specification is validated in `normalize_cooling`,
    ## which is where the three keywords are reconciled.
    (0 ≤ trigger ≤ 1) || error("`trigger` must lie in [0,1].")
    (0 ≤ target ≤ 1) || error("`target` must lie in [0,1].")
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
    schedule::Function,
    cooling_type::Symbol,
    cooling_fraction_50::Real,
    kernel::RWKernel,
    perturb::Union{Function,NamedTuple,Nothing},
    transform::Function,
    inverse_transform::Function,
    trigger::Real,
    target::Real,
    Nmonitor::Integer,
    Np_monitor::Integer,
) where {E<:NamedTuple,R0<:NamedTuple} = begin
    t0_0 = timezero(object)
    t = times(object)
    y = obs(object)
    N = length(t)
    trig = LogLik(trigger)
    targ = LogLik(target)
    α = Float64(cooling_fraction_50)
    sdschedule = resolve_sdschedule(rw_sd,rw_sd_init,N)
    ## A user-supplied schedule is checked here rather than in
    ## `validate_mif2_args`, which does not receive `m0` and so cannot
    ## evaluate the schedule over the range of iterations actually to be
    ## run. Monotonicity is deliberately not required: a non-monotone
    ## schedule is unusual but not incoherent.
    if Nmif > 0
        for (mm,nn) ∈ ((m0+1,1),(m0+Nmif,N))
            c = schedule(mm,nn,N)
            (c isa Real && isfinite(c) && c ≥ 0) ||
                error("the `cooling` schedule must return a finite, nonnegative "*
                      "real number; at (m=$mm, n=$nn, N=$N) it returned $c.")
        end
    end

    theta = [inverse_transform(est[i]) for i ∈ eachindex(est)]
    Q = eltype(theta)
    estbuf = similar(est)
    thetabuf = similar(theta)
    w = ones(LogLik,Np)
    work = similar(w)
    perm = zeros(Int,Np)
    cll = fill(LogLik(NaN),N)
    ess = fill(LogLik(NaN),N)
    resamp = fill(false,N)
    chunks,thetabufs = chunk_params(Np,Q)

    ## The filtered-state array only needs its element type, which the
    ## model already carries: `init_state` is declared of the latent
    ## state type. Calling `rinit` here to discover it -- as this did
    ## previously -- would draw and immediately discard Np initial
    ## states, consuming randomness before the loop and so offsetting
    ## the stream of every continuation relative to an equivalent single
    ## run. The first `rinit!` below, at n == 1, fills `xf` for real.
    xf = Array{typeof(init_state(object))}(undef,Np,1)
    xp = similar(xf,1,Np,1)
    ell4 = Array{LogLik}(undef,1,Np,1,1)
    ybuf = Array{eltype(y)}(undef,1,Np,1)

    traces_ = copy(tr0)
    monitor_ll = LogLik(NaN)
    ## the weighted parameter mean at the final observation time, as in
    ## R pomp; with Nmif=0 (the `for m` loop below never executes) this
    ## is the plain mean of the untouched starting cloud
    estbar = weighted_mean(est,w)

    for m ∈ (m0+1):(m0+Nmif)
        fill!(w,1)
        t0 = t0_0
        for n ∈ 1:N
            c = schedule(m,n,N)
            ## On return the kernel's primary cloud is current. The
            ## transform-derived default writes `est` and derives
            ## `theta`; a natural-scale kernel writes `theta` and leaves
            ## `est` stale until it is needed at n == N.
            perturb_cloud!(kernel,theta,est,sdschedule[n],c)
            if n==1
                rinit!(object,xf;t0,params=theta)
            end
            fill!(ybuf,y[n])
            advance_particles_cloud!(
                object,t0,xf,xp,ell4,
                @view(t[[n]]),ybuf,theta,chunks,thetabufs,
            )
            pfilt_step_comps!(
                @view(cll[n]),@view(ess[n]),
                @view(ell4[1,:,1,1]),perm,
                @view(xp[1,:,1]),@view(xf[:,1]),
                w,work,
                trig,targ,
                @view(resamp[n]),
            )
            if n==N
                ## the general step normalizes the total log weights in
                ## place into `ell4` and leaves them untouched by
                ## resampling, so this is the pre-resampling combined
                ## weighted particle representation, aligned with `est`
                ## before the ancestry permutation just below
                sync_est!(kernel,theta,est,transform)
                estbar = weighted_mean(est,exp.(@view(ell4[1,:,1,1])))
            end
            if resamp[n]
                theta,thetabuf,est,estbuf =
                    permute_cloud!(kernel,theta,thetabuf,est,estbuf,perm)
            end
            t0 = t[n]
        end
        thetabar = inverse_transform(estbar)
        if Nmonitor > 0
            reps = [
                logLik(pfilter(object;Np=Np_monitor,params=thetabar))
                for _ ∈ 1:Nmonitor
            ]
            monitor_ll = logmeanexp(reps)
        end
        ## `monitor_logLik` is present in every trace row, and is `NaN`
        ## when monitoring is off. Emitting it conditionally would make
        ## the row type depend on `Nmonitor`, so a run started with
        ## `Nmonitor = 0` could not be continued with `Nmonitor > 0`:
        ## the stored trace vector is concretely typed and would reject
        ## the wider row.
        push!(traces_,(;iteration=m,logLik=sum(cll),monitor_logLik=monitor_ll,thetabar...))
    end

    thetafinal = inverse_transform(estbar)
    paramcloud,estcloud = final_clouds(kernel,theta,est,transform,inverse_transform)

    Mif2dPompObject(
        pomp(object;params=thetafinal),
        m0+Nmif,Np,
        rw_sd,rw_sd_init,
        schedule,cooling_type,α,trig,targ,
        perturb,transform,inverse_transform,
        paramcloud,
        estcloud,
        log.(w),
        traces_,
        cll,ess,resamp,
        sum(cll),
        monitor_ll,
        Nmonitor,
        Np_monitor,
    )
end

"""
    mif2(
        object;
        Nmif = 1, Np = 1, rw_sd, rw_sd_init = (;),
        params = coef(object),
        cooling_type = :geometric, cooling_fraction_50 = 0.5,
        transform = identity, inverse_transform = nothing,
        trigger = 1, target = 0, Nmonitor = 0, Np_monitor = Np,
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
- `Np`: number of particles. `params` may instead supply the starting
  parameter cloud directly (see below), in which case `Np` defaults to
  its length and, if given explicitly, must equal it.
- `rw_sd`: a `NamedTuple` of random-walk standard deviations, on the
  estimation scale, for the parameters to be estimated. Parameters not
  named in `rw_sd` (or `rw_sd_init`) are held fixed. Each entry may be a
  nonnegative real number (constant across observation times), a vector
  of length equal to the number of observation times, a function of the
  1-based observation index returning a nonnegative real number, or an
  [`ivp`](@ref) (nonzero only immediately before specified observation
  times).
- `perturb`: the perturbation kernel. Omitted (the default), the kernel
  is derived from `transform`: an additive Gaussian increment on the
  estimation scale, which for a `:log`-tagged parameter is a
  multiplicative lognormal perturbation on the natural scale, matching R
  `pomp`. Otherwise it acts on the *natural* scale and is either a
  `NamedTuple` naming one kernel per estimated parameter --
  [`normal_rw`](@ref), [`lognormal_rw`](@ref),
  [`logitnormal_rw`](@ref), [`student_rw`](@ref) -- or a function
  `(θ,sd,c)` of one particle's natural-scale parameters, the
  standard deviations for the current observation time, and the cooling
  factor, returning a `NamedTuple` of the parameters it changed.

  Note that `rw_sd` is then the *kernel's* scale parameter, and each
  kernel documents what it means: the standard deviation of ``\\theta``
  for `normal_rw`, of ``\\log\\theta`` for `lognormal_rw`, of
  ``\\mathrm{logit}\\,\\theta`` for `logitnormal_rw`. Under the default
  kernel with `transform = (θ=:log,)` it already is the standard
  deviation of ``\\log\\theta``, so these agree and no existing
  specification changes meaning.

  The particle cloud is averaged on the scale on which the kernel
  perturbs symmetrically, so that a multiplicative kernel is paired with
  a geometric mean. In the `NamedTuple` form the kernels declare that
  scale themselves and it is composed automatically, which is why naming
  a parameter in both `perturb` and `transform` is an error rather than
  a silent bias. A bare function cannot be introspected, so the
  averaging scale is taken from `transform` and keeping the two
  consistent is the caller's responsibility.
- `rw_sd_init`: as `rw_sd`, but restricted to a nonnegative real number
  for each named parameter, giving a perturbation applied only before
  the first observation time (equivalent to naming the same parameter
  with `ivp(sd,lags=1)` in `rw_sd`, matching R `pomp`'s `ivp`). Its keys
  must be disjoint from those of `rw_sd`.
- `cooling_type`: `:geometric` or `:hyperbolic`.
- `cooling_fraction_50`: the fraction by which the random-walk standard
  deviation shrinks after 50 IF2 iterations.
- `cooling`: an alternative to the two keywords above. Either one of the
  same two `Symbol`s, or a function of `(m,n,N)` -- the IF2 iteration
  number, the 1-based observation index, and the number of observation
  times -- returning the finite, nonnegative factor multiplying every
  random-walk standard deviation at that point. Supplying both `cooling`
  and `cooling_type` is an error. A function is evaluated at the two
  ends of the range of iterations actually to be run and must return a
  finite, nonnegative real at each; it is *not* required to be
  monotone. The two built-in schedules are also available in this form
  as [`geometric_cooling`](@ref) and [`hyperbolic_cooling`](@ref), so
  that a custom schedule can be written as a modification of one of
  them. Note that the built-ins advance at per-observation-time
  granularity, decreasing within an iteration as well as across
  iterations: the geometric form is `cooling_fraction_50^(s/(50N))` in
  the global step count `s = n-1+(m-1)N`.
- `params`: the starting parameter set, on the natural scale, for the
  particle cloud -- either a single `NamedTuple`, replicated `Np` times,
  or an `AbstractVector` of `NamedTuple`s (sharing the same parameter
  names) giving a non-degenerate starting cloud directly. In the latter
  case, the base parameter set and the iteration-0 entry of `traces`
  are the natural-scale image of the plain mean, on the estimation
  scale, of the starting cloud.
- `transform`, `inverse_transform`: the parameter transformation mapping
  the natural scale to and from the estimation scale on which
  perturbation is applied; `transform` accepts one of three forms: (i)
  a `Function`, paired with `inverse_transform` (a `Function`, default
  `identity`); (ii) a `NamedTuple` tagging parameters `:log`, `:logit`,
  or `:identity` (see [`ParameterTransform`](@ref)), with
  `inverse_transform` left at its default of `nothing` since the inverse
  is derived; or (iii) a [`ParameterTransform`](@ref), built the same
  way and additionally allowing log-barycentric-transformed groups,
  again with `inverse_transform` left at its default.
- `trigger`, `target`: as in [`pfilter`](@ref) -- (state, parameter)
  particle pairs are resampled together whenever the effective sample
  size falls to `trigger*Np` or below, with selection probability and
  retained weight governed by `target`. The IF2 theory of Ionides et al.
  (2015) requires equally weighted resampling at every observation time,
  so the defaults are `trigger = 1`, `target = 0`; both are exposed for
  experimentation only.
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
    Np::Union{Integer,Nothing} = nothing,
    rw_sd::NamedTuple,
    rw_sd_init::NamedTuple = (;),
    params::Union{NamedTuple,AbstractVector{<:NamedTuple}} = coef(object),
    perturb::Union{Function,NamedTuple,Nothing} = nothing,
    cooling::Union{Symbol,Function,Nothing} = nothing,
    cooling_type::Union{Symbol,Nothing} = nothing,
    cooling_fraction_50::Real = 0.5,
    transform::Union{Function,NamedTuple,ParameterTransform} = identity,
    inverse_transform::Union{Function,Nothing} = nothing,
    trigger::Real = 1,
    target::Real = 0,
    Nmonitor::Integer = 0,
    Np_monitor::Union{Integer,Nothing} = nothing,
    rinit::Union{Function,Nothing,Missing} = missing,
    rprocess::Union{PompPlugin,Nothing,Missing} = missing,
    logdmeasure::Union{Function,Nothing,Missing} = missing,
    kwargs...,
) = begin
    to,from = normalize_transform(transform,inverse_transform)
    if params isa AbstractVector
        isempty(params) &&
            error("`params`, given as a vector of initial parameter sets, must be nonempty.")
        ks0 = keys(params[1])
        all(p -> keys(p)==ks0,params) ||
            error("all entries of the initial parameter cloud `params` must share "*
                  "the same parameter names.")
        Np_ = if Np === nothing
            length(params)
        else
            Np == length(params) ||
                error("`Np` must equal `length(params)` when `params` is a vector of "*
                      "initial parameter sets (got Np=$Np, length(params)=$(length(params))).")
            Np
        end
        params0 = params[1]
        est0 = [to(p) for p ∈ params]
        params_base = from(weighted_mean(est0,ones(LogLik,length(est0))))
    else
        Np_ = Np === nothing ? 1 : Np
        params0 = params
        est0 = fill(to(params),Np_)
        params_base = params
    end
    Np_monitor_ = Np_monitor === nothing ? Np_ : Np_monitor
    validate_partrans_names(transform,params0)
    object = pomp(object;params=params_base,rinit,rprocess,logdmeasure,kwargs...)
    N = length(times(object))
    schedule,ctype,α = normalize_cooling(cooling,cooling_type,cooling_fraction_50,N)
    sdkeys = merge(rw_sd,rw_sd_init)
    kernel = normalize_perturb(perturb,to,from,sdkeys)
    validate_kernel_support(kernel,params0)
    to_avg,from_avg = normalize_average(perturb,transform,to,from)
    ## `est0` was built with the user's `to`, which for a per-key kernel
    ## is not the averaging transformation, so the starting cloud has to
    ## be carried across. When the two coincide -- always on the default
    ## and bare-function paths -- the cloud is used as it stands, which
    ## avoids a `to∘from` round trip whose floating-point residue
    ## resampling would then amplify.
    est1 = to_avg === to ? est0 : [to_avg(from(e)) for e ∈ est0]
    validate_mif2_args(
        params0,to_avg,from_avg,rw_sd,rw_sd_init,
        trigger,target,
        Nmif,Np_,Nmonitor,Np_monitor_,N,
        kernel,
    )
    tr0 = [(;iteration=0,logLik=LogLik(NaN),monitor_logLik=LogLik(NaN),params_base...)]
    mif2_run(
        object,0,est1,tr0,
        Nmif,Np_,rw_sd,rw_sd_init,
        schedule,ctype,α,
        kernel,perturb,
        to_avg,from_avg,
        trigger,target,Nmonitor,Np_monitor_,
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
may be overridden. As on a fresh call, a declarative `transform` (a
`NamedTuple` of tags or a [`ParameterTransform`](@ref)) must not be
paired with an explicit `inverse_transform`.

Supplying a new `transform` carries the stored cloud onto the new
estimation scale before resuming, so the particles retain their
natural-scale values; without that step each particle would be
reinterpreted under the wrong coordinates.
"""
mif2(
    object::Mif2dPompObject;
    Nmif::Integer = 1,
    Np::Integer = object.Np,
    rw_sd::NamedTuple = object.rw_sd,
    rw_sd_init::NamedTuple = object.rw_sd_init,
    ## A stored custom schedule is carried forward through `cooling`, and
    ## `cooling_type` is left at `nothing` so that `normalize_cooling`
    ## does not see both. For the two built-in schedules the reverse
    ## holds, so the stored `cooling_fraction_50` still governs.
    cooling::Union{Symbol,Function,Nothing} =
        (object.cooling_type === :custom ? object.cooling : nothing),
    cooling_type::Union{Symbol,Nothing} =
        (object.cooling_type === :custom ? nothing : object.cooling_type),
    cooling_fraction_50::Real = object.cooling_fraction_50,
    perturb::Union{Function,NamedTuple,Nothing} = object.perturb,
    transform::Union{Function,NamedTuple,ParameterTransform,Nothing} = nothing,
    inverse_transform::Union{Function,Nothing} = nothing,
    trigger::Real = object.trigger,
    target::Real = object.target,
    Nmonitor::Integer = object.Nmonitor,
    Np_monitor::Integer = object.Np_monitor,
    kwargs...,
) = begin
    Np == object.Np ||
        error("`Np` cannot be changed on continuation (got Np=$Np, stored cloud has $(object.Np)).")
    to,from = if transform === nothing
        (object.transform,object.inverse_transform)
    else
        normalize_transform(transform,inverse_transform)
    end
    ## The stored `estcloud` is expressed on the estimation scale of the
    ## *previous* run. If a new transformation is supplied, that cloud
    ## must be carried onto the new estimation scale before it is used,
    ## or every particle is reinterpreted under the wrong coordinates.
    ## `paramcloud` is the natural-scale image of `estcloud`, so the new
    ## estimation-scale cloud is its image under the new `to`. When the
    ## transformation is unchanged the stored cloud is used directly,
    ## which avoids an unnecessary `to∘from` round trip and keeps
    ## continuation bit-exact.
    est0 = if transform === nothing
        copy(object.estcloud)
    else
        [to(p) for p ∈ object.paramcloud]
    end
    base = pomp(object;kwargs...)
    N = length(times(base))
    params0 = coef(base)
    transform === nothing || validate_partrans_names(transform,params0)
    schedule,ctype,α = normalize_cooling(cooling,cooling_type,cooling_fraction_50,N)
    sdkeys = merge(rw_sd,rw_sd_init)
    kernel = normalize_perturb(perturb,to,from,sdkeys)
    validate_kernel_support(kernel,params0)
    ## When the transformation is not overridden, the stored pair is
    ## already the averaging pair, so recomposing it would be wasted work
    ## and, for the per-key form, would need the raw tags that are not
    ## stored. Recompose only when a new transformation was supplied.
    to_avg,from_avg = transform === nothing ?
        (to,from) : normalize_average(perturb,transform,to,from)
    validate_mif2_args(
        params0,to_avg,from_avg,rw_sd,rw_sd_init,
        trigger,target,
        Nmif,Np,Nmonitor,Np_monitor,N,
        kernel,
    )
    mif2_run(
        base,object.Nmif,est0,copy(object.traces),
        Nmif,Np,rw_sd,rw_sd_init,
        schedule,ctype,α,
        kernel,perturb,
        to_avg,from_avg,
        trigger,target,Nmonitor,Np_monitor,
    )
end

mif2(_...) = error("Incorrect call to `mif2`.")
