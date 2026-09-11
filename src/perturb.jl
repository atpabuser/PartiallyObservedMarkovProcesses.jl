## ------------------------------------------------------------- perturb
##
## The perturbation kernel of the IF2 procedure. On the m-th iteration,
## immediately before observation time n, each particle's parameter
## vector is perturbed by a random walk whose scale is c·σ, where σ is
## the entry of `rw_sd` for that parameter at that observation time and
## c is the cooling factor. The kernel is what turns (θ, σ, c) into a
## perturbed θ.
##
## Historically that step was fixed: an additive Gaussian increment on
## the estimation scale defined by `transform`, which for a `:log`-tagged
## parameter is a multiplicative lognormal perturbation on the natural
## scale, and which is what R `pomp` does (`mif2.c`'s
## `randwalk_perturbation` is `*xs += *xrw * norm_rand()`, applied to
## parameters already carried on the estimation scale). That remains the
## default. A kernel may now be supplied instead, in which case it acts
## on the natural scale.
##
## THE KERNEL CONTRACT. A kernel is called as `k(θ, sd, c)` and must
## return a `NamedTuple`:
##
##   K1  its keys are a subset of `keys(sd)`; every other parameter is
##       left alone, since the caller merges the result into θ.
##   K2  the number of random draws it makes does not depend on the
##       *values* in `sd`. In particular a zero standard deviation still
##       consumes its draw. This is not needed for correctness, but it
##       makes the random stream a function of the schedule's shape
##       rather than of its values, so that two equivalent spellings of
##       the same schedule -- `ivp(σ)` and the explicit vector
##       `[σ; zeros(N-1)]`, say -- give identical results.
##   K3  a zero standard deviation is a no-op in value.
##   K4  it maps its declared support into itself. This is what makes a
##       single check at validation time sufficient, with no guards in
##       the inner loop.
##
## A kernel also declares, through `mean_scale`, the scale on which its
## perturbations are symmetric. The engine averages the particle cloud
## on that scale, so that a multiplicative kernel is paired with a
## geometric mean rather than an arithmetic one. Getting this wrong is a
## silent statistical defect, which is why the declaration belongs to
## the kernel rather than to the user.

import Distributions: TDist

## ------------------------------------------------------- scalar kernels

"""
    ScalarRW

A per-parameter perturbation kernel, called as `k(x, s, c)` on a single
natural-scale parameter value `x`, its random-walk standard deviation
`s`, and the cooling factor `c`. See [`normal_rw`](@ref),
[`lognormal_rw`](@ref), [`logitnormal_rw`](@ref), [`student_rw`](@ref).
"""
abstract type ScalarRW <: Function end

struct NormalRW <: ScalarRW end
struct LognormalRW <: ScalarRW end
struct LogitnormalRW <: ScalarRW end
struct StudentRW{S} <: ScalarRW
    ν::Float64
end

(::NormalRW)(x::Real, s::Real, c::Real) = x + c*s*randn()
(::LognormalRW)(x::Real, s::Real, c::Real) = x*exp(c*s*randn())
(::LogitnormalRW)(x::Real, s::Real, c::Real) =
    1/(1+exp(-(log(x/(1-x))+c*s*randn())))

(k::StudentRW{:identity})(x::Real, s::Real, c::Real) = x + c*s*rand(TDist(k.ν))
(k::StudentRW{:log})(x::Real, s::Real, c::Real) = x*exp(c*s*rand(TDist(k.ν)))
(k::StudentRW{:logit})(x::Real, s::Real, c::Real) =
    1/(1+exp(-(log(x/(1-x))+c*s*rand(TDist(k.ν)))))

"""
    mean_scale(k)

The scale on which the kernel `k` perturbs symmetrically, one of `:log`,
`:logit`, or `:identity`. The IF2 point estimate is the weighted mean of
the particle cloud on this scale.
"""
mean_scale(::NormalRW) = :identity
mean_scale(::LognormalRW) = :log
mean_scale(::LogitnormalRW) = :logit
mean_scale(::StudentRW{S}) where {S} = S

## The set into which the kernel maps, checked once at validation time.
support(::NormalRW) = :real
support(::LognormalRW) = :positive
support(::LogitnormalRW) = :unit
support(::StudentRW{S}) where {S} =
    S === :log ? :positive : (S === :logit ? :unit : :real)

in_support(k::ScalarRW, x::Real) = begin
    sup = support(k)
    if sup === :positive
        isfinite(x) && x > 0
    elseif sup === :unit
        isfinite(x) && 0 < x < 1
    else
        isfinite(x)
    end
end

support_description(k::ScalarRW) = begin
    sup = support(k)
    sup === :positive ? "strictly positive" :
    sup === :unit ? "strictly between 0 and 1" : "finite"
end

"""
    normal_rw()

An additive Gaussian random walk on the natural scale:
``\\theta \\mapsto \\theta + c\\,\\sigma\\,Z``. `rw_sd` is the standard
deviation of ``\\theta`` itself. Averaging is arithmetic. Suitable for a
parameter that is unconstrained in sign.
"""
normal_rw() = NormalRW()

"""
    lognormal_rw()

A multiplicative lognormal random walk:
``\\theta \\mapsto \\theta\\,e^{c\\sigma Z}``, so `rw_sd` is the standard
deviation of ``\\log\\theta``. Positivity is preserved exactly, and
averaging is geometric. This is what the default kernel does for a
parameter tagged `:log`, and what R `pomp` does under the same tag.
"""
lognormal_rw() = LognormalRW()

"""
    logitnormal_rw()

A random walk that is Gaussian on the logistic scale:
``\\theta \\mapsto \\mathrm{logit}^{-1}(\\mathrm{logit}\\,\\theta +
c\\sigma Z)``, so `rw_sd` is the standard deviation of
``\\mathrm{logit}\\,\\theta``. Confinement to ``(0,1)`` is preserved
exactly.
"""
logitnormal_rw() = LogitnormalRW()

"""
    student_rw(ν; scale = :log)

A heavy-tailed random walk, Student-t with `ν` degrees of freedom on the
scale named by `scale` (`:log`, `:logit`, or `:identity`). The wider
tails let the cloud make occasional large excursions, which can help a
search escape a local optimum that a Gaussian walk would settle into.

Unlike the Gaussian kernels this does not make a fixed number of random
draws per call (the Student-t sampler is rejection-based), so K2 of the
kernel contract does not hold: two equivalent spellings of the same
`rw_sd` schedule will still agree, but a change elsewhere in the draw
sequence propagates differently than it would under a Gaussian kernel.
"""
student_rw(ν::Real; scale::Symbol = :log) = begin
    (isfinite(ν) && ν > 0) ||
        error("`student_rw` requires a finite, positive degrees of freedom; got $ν.")
    scale ∈ (:log,:logit,:identity) ||
        error("`student_rw` `scale` must be `:log`, `:logit`, or `:identity`; got `$scale`.")
    StudentRW{scale}(Float64(ν))
end

## ------------------------------------------------------- cloud kernels

"""
    RWKernel

A whole-cloud perturbation kernel, called as `k(θ, sd, c)` on one
particle's parameter `NamedTuple`. Three forms are built from the
`perturb` argument of [`mif2`](@ref): the transform-derived default, a
bare function, and a `NamedTuple` of per-parameter [`ScalarRW`](@ref)
kernels.
"""
abstract type RWKernel <: Function end

## The default. Rather than a per-key natural-scale rule, this pushes the
## whole parameter tuple onto the estimation scale, adds the Gaussian
## increment there, and pulls it back. That is not merely for backward
## compatibility: `to`/`from` do not decompose key by key in two
## supported cases -- log-barycentric groups, where each coordinate
## depends on the whole group's sum, and an arbitrary user-supplied
## function pair -- so a per-key reformulation would silently break both.
struct TransformRW{TT<:Function,FF<:Function} <: RWKernel
    to::TT
    from::FF
end

## A user-supplied natural-scale kernel, opaque to introspection.
struct FunctionRW{F<:Function} <: RWKernel
    f::F
end

(k::FunctionRW)(θ::NamedTuple, sd::NamedTuple, c::Real) = k.f(θ,sd,c)

## A `NamedTuple` of per-parameter kernels. The kernels live in a field
## rather than a type parameter: the `@generated` body below specializes
## on their types either way, but the field form does not require them to
## be `isbits` and so admits a closure as an escape hatch for one
## parameter without giving up specialization on the others.
struct PerKeyRW{K<:NamedTuple} <: RWKernel
    kers::K
end

@generated (k::PerKeyRW{K})(
    θ::NamedTuple,
    sd::NamedTuple{ks},
    c::Real,
) where {K,ks} = begin
    exprs = Expr[]
    for key ∈ ks
        q = QuoteNode(key)
        push!(exprs,:(getfield(k.kers,$q)(getproperty(θ,$q),getproperty(sd,$q),c)))
    end
    :(NamedTuple{$ks}(($(exprs...),)))
end

## The default kernel's own step: an additive Gaussian increment on the
## estimation-scale cloud, applied to every parameter named in `sd`.
## Note that a zero standard deviation still consumes a draw (K2): the
## key set of `sd` is fixed for the whole run, with time-localization
## expressed as zeros in the values, so this keeps the random stream a
## function of the schedule's shape rather than its values.
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

## --------------------------------------------------------- cloud primacy

## Which of the two clouds the kernel writes to, and therefore which one
## the engine must carry through resampling. The transform-derived
## default perturbs the estimation-scale cloud and derives the
## natural-scale one; every natural-scale kernel does the reverse.
primary(::TransformRW) = :est
primary(::RWKernel) = :theta

## Applies one perturbation step to the cloud. On return the primary
## cloud is current; the other may be stale.
perturb_cloud!(
    k::TransformRW,
    theta::AbstractVector{Q},
    est::AbstractVector{E},
    sd::NamedTuple,
    c::Real,
) where {Q<:NamedTuple,E<:NamedTuple} = begin
    perturb!(est,sd,c)
    @inbounds for i ∈ eachindex(est)
        theta[i] = k.from(est[i])::Q
    end
    nothing
end

perturb_cloud!(
    k::RWKernel,
    theta::AbstractVector{Q},
    est::AbstractVector{E},
    sd::NamedTuple,
    c::Real,
) where {Q<:NamedTuple,E<:NamedTuple} = begin
    @inbounds for i ∈ eachindex(theta)
        theta[i] = merge(theta[i],k(theta[i],sd,c))::Q
    end
    nothing
end

## Brings the estimation-scale cloud into agreement with the
## natural-scale one, for the weighted mean at the final observation
## time. A no-op when the estimation-scale cloud is already primary.
## Note that these dispatch on the kernel alone. The two clouds can have
## the same element type -- they do whenever the transformation is
## `identity` -- so constraining their type parameters here would make
## the methods ambiguous rather than ordered.
sync_est!(
    ::TransformRW,
    theta::AbstractVector,
    est::AbstractVector,
    to_avg::Function,
) = nothing

sync_est!(
    ::RWKernel,
    theta::AbstractVector,
    est::AbstractVector,
    to_avg::Function,
) = begin
    E = eltype(est)
    @inbounds for i ∈ eachindex(theta)
        est[i] = to_avg(theta[i])::E
    end
    nothing
end

## After the final iteration the primary cloud is authoritative; the
## other is brought into agreement so that the two clouds recorded on the
## returned object describe the same particles. Returns
## `(paramcloud, estcloud)`.
final_clouds(
    ::TransformRW,
    ::AbstractVector,
    est::AbstractVector,
    ::Function,
    from_avg::Function,
) = ([from_avg(e) for e ∈ est],est)

final_clouds(
    ::RWKernel,
    theta::AbstractVector,
    ::AbstractVector,
    to_avg::Function,
    ::Function,
) = (copy(theta),[to_avg(p) for p ∈ theta])

## Applies the resampling ancestry to the primary cloud, by permuting
## into its spare buffer and swapping. The non-primary cloud is rebuilt
## before it is next read, so it is left alone. Returns the four vectors
## in the order `(theta, thetabuf, est, estbuf)`; the caller rebinds, as
## a swap of local names cannot be done from inside a function.
permute_cloud!(
    ::TransformRW,
    theta::AbstractVector,
    thetabuf::AbstractVector,
    est::AbstractVector,
    estbuf::AbstractVector,
    perm::AbstractVector{<:Integer},
) = begin
    @inbounds for j ∈ eachindex(perm)
        estbuf[j] = est[perm[j]]
    end
    (theta,thetabuf,estbuf,est)
end

permute_cloud!(
    ::RWKernel,
    theta::AbstractVector,
    thetabuf::AbstractVector,
    est::AbstractVector,
    estbuf::AbstractVector,
    perm::AbstractVector{<:Integer},
) = begin
    @inbounds for j ∈ eachindex(perm)
        thetabuf[j] = theta[perm[j]]
    end
    (thetabuf,theta,est,estbuf)
end
