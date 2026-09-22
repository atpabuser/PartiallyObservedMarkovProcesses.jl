import DataFrames: DataFrame
import Distributions: Chisq, quantile
import LinearAlgebra: dot, SingularException

## A local-regression smoother matching R's `loess` with
## `degree = 2`, `family = "gaussian"`, and `surface = "direct"` for one
## predictor: at each evaluation point the `q = floor(span*n)` nearest
## observations receive tricube weights in their distance to that point,
## scaled by the distance of the `q`-th nearest, and a weighted quadratic
## is fitted. For `span > 1` all observations are used and the bandwidth
## is the largest distance times `sqrt(span)` (what R's C code does for
## one predictor, checked numerically against R). R's default `surface =
## "interpolate"` blends exact fits at the vertices of a k-d tree and
## differs from this direct evaluation at the fourth or fifth
## significant figure.
struct Loess
    x::Vector{Float64}
    y::Vector{Float64}
    span::Float64
end

loess(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}; span::Real = 0.75) = begin
    length(x) == length(y) || error("`loess`: `x` and `y` must have the same length.")
    length(x) ≥ 3 || error("`loess`: at least three observations are needed.")
    span > 0 || error("`loess`: `span` must be positive.")
    all(isfinite,x) && all(isfinite,y) ||
        error("`loess`: `x` and `y` must be finite.")
    (span > 1 || floor(Int,span*length(x)) ≥ 3) ||
        error("`loess`: `span*n` must be at least 3.")
    Loess(collect(Float64,x),collect(Float64,y),Float64(span))
end

(fit::Loess)(x0::Real) = begin
    x, y, span = fit.x, fit.y, fit.span
    n = length(x)
    d = abs.(x .- x0)
    h = if span ≤ 1
        partialsort(d,floor(Int,span*n))
    else
        maximum(d)*sqrt(span)
    end
    h > 0 || error("`loess`: zero bandwidth at $x0; the data have too few distinct `x` values.")
    w = @. ifelse(d < h, (1-(d/h)^3)^3, 0.0)
    u = x .- x0
    ## weighted quadratic: predict the intercept
    X = hcat(ones(n),u,u.^2)
    sw = sqrt.(w)
    β = (sw .* X) \ (sw .* y)
    β[1]
end

"""
    MCAP

The result of a [`mcap`](@ref) computation. Fields:
- `logLik`, `parameter`: the input points.
- `level`, `span`: the settings used.
- `fit`: a `DataFrame` with columns `parameter` (the evaluation grid),
  `smoothed` (the local-regression smooth), and `quadratic` (the
  weighted quadratic fit).
- `mle`: the maximizer of the smooth over the grid.
- `quadratic_max`: the maximizer of the quadratic fit.
- `ci`: the Monte Carlo adjusted confidence interval, as a tuple.
- `delta`: the adjusted log-likelihood cutoff defining `ci`.
- `se_stat`, `se_mc`, `se`: the statistical, Monte Carlo, and total
  standard errors of the point estimate.
- `coefs`: the quadratic fit `c + b*parameter - a*parameter^2` as
  `(c=..., a=..., b=...)`.
"""
struct MCAP
    logLik::Vector{Float64}
    parameter::Vector{Float64}
    level::Float64
    span::Float64
    fit::DataFrame
    mle::Float64
    quadratic_max::Float64
    ci::Tuple{Float64,Float64}
    delta::Float64
    se_stat::Float64
    se_mc::Float64
    se::Float64
    coefs::NamedTuple{(:c,:a,:b),NTuple{3,Float64}}
end

Base.show(io::IO, m::MCAP) = begin
    print(io,"MCAP: mle=$(m.mle), ci=$(m.ci), se=$(m.se) (stat $(m.se_stat), mc $(m.se_mc)), level=$(m.level)")
end

"""
    mcap(logLik, parameter; level = 0.95, span = 0.75, Ngrid = 1000)

Monte Carlo adjusted profile, a port of R `pomp`'s `mcap` (Ionides et
al. 2017). `logLik` and `parameter` are vectors of profile
log-likelihood estimates and the parameter values at which they were
obtained, typically the `loglik` column and the profiled column of the
data frame returned by [`profile`](@ref).

The points are smoothed by local quadratic regression with the given
`span`; the maximizer of the smooth over a grid of `Ngrid` points is the
point estimate `mle`. A weighted quadratic is then fitted around that
maximizer, from which the statistical standard error (from the
curvature) and the Monte Carlo standard error (from the coefficient
covariance) are obtained. The confidence interval at the given `level`
is the set of grid points at which the smooth lies within `delta` of
its maximum, where `delta` is the chi-square cutoff inflated for the
Monte Carlo error. Returns an [`MCAP`](@ref).

The smoother evaluates the local fit directly at every grid point,
which corresponds to R's `loess` with `surface = "direct"`; R's default
interpolating surface differs from it at the fourth or fifth significant
figure.
"""
mcap(
    logLik::AbstractVector{<:Real},
    parameter::AbstractVector{<:Real};
    level::Real = 0.95,
    span::Real = 0.75,
    Ngrid::Integer = 1000,
) = begin
    ll = collect(Float64,logLik)
    par = collect(Float64,parameter)
    n = length(ll)
    n == length(par) || error("`mcap`: `logLik` and `parameter` must have the same length.")
    all(isfinite,ll) ||
        error("`mcap`: `logLik` must be finite; drop the non-finite points first.")
    all(isfinite,par) || error("`mcap`: `parameter` must be finite.")
    0 < level < 1 || error("`mcap`: `level` must lie in (0,1).")
    Ngrid ≥ 2 || error("`mcap`: `Ngrid` must be at least 2.")
    trunc(Int,span*n) ≥ 1 ||
        error("`mcap`: `span*length(parameter)` must be at least 1.")
    smooth_fit = loess(par,ll;span)
    grid = collect(range(minimum(par),maximum(par),length=Ngrid))
    smoothed = smooth_fit.(grid)
    smooth_arg_max = grid[argmax(smoothed)]
    dist = abs.(par .- smooth_arg_max)
    included = dist .< sort(dist)[trunc(Int,span*n)]
    any(included) ||
        error("`mcap`: no points fall inside the quadratic fitting window; increase `span`.")
    maxdist = maximum(dist[included])
    w = zeros(Float64,n)
    w[included] .= (1 .- (dist[included]./maxdist).^3).^3
    ## weighted least squares: logLik ~ 1 + a + b, with a = -parameter^2, b = parameter
    X = hcat(ones(n),-par.^2,par)
    sw = sqrt.(w)
    β = (sw .* X) \ (sw .* ll)
    c, a, b = β
    r = ll .- X*β
    nnz = count(>(0),w)
    ## as R's `lm`: with no residual degrees of freedom the variance is
    ## undefined and the standard errors and interval come out NaN. The
    ## Gram matrix `X'*(w.*X)` can be exactly or near-exactly singular
    ## with few weighted points, or with weighted points that carry too
    ## little spread in `parameter` to pin down a quadratic -- `nnz > 3`
    ## catches the former; `inv` throwing (rather than silently
    ## returning non-finite entries) is the general safety net for both,
    ## including cases `nnz` alone would miss.
    nnz > 3 || @warn "`mcap`: only $nnz points carry weight in the quadratic fit; increase `span` or add points."
    var_a,var_b,cov_ab = if nnz > 3
        try
            σ2 = dot(w,r.^2)/(nnz-3)
            V = σ2 .* inv(X'*(w .* X))
            V[2,2],V[3,3],V[2,3]
        catch e
            e isa Union{ArgumentError,SingularException} || rethrow()
            @warn "`mcap`: the weighted design is singular; standard errors are not defined."
            NaN,NaN,NaN
        end
    else
        NaN,NaN,NaN
    end
    se_mc_squared = (1/(4*a*a))*(var_b - (2*b/a)*cov_ab + (b*b/a/a)*var_a)
    se_stat_squared = 1/2/a
    se_total_squared = se_mc_squared + se_stat_squared
    concave = a > 0
    concave || @warn "`mcap`: the quadratic fit is not concave; standard errors and the interval are not defined."
    delta = quantile(Chisq(1),level)*(a*se_mc_squared + 0.5)
    logLik_diff = maximum(smoothed) .- smoothed
    inside = grid[logLik_diff .< delta]
    ## `delta`'s chi-square cutoff is derived assuming a concave fit; a
    ## non-concave one can still pass `logLik_diff .< delta` at every
    ## grid point (a spuriously "whole-range" interval), which is not a
    ## confidence interval in any defined sense, so it is not returned
    ## as one -- matching the standard errors this same non-concavity
    ## already nulls, above.
    ci = (concave && !isempty(inside)) ? (minimum(inside),maximum(inside)) : (NaN,NaN)
    ## as R: the square root of a negative variance is NaN
    nsqrt(v) = v ≥ 0 ? sqrt(v) : NaN
    quadratic = c .+ b.*grid .- a.*grid.^2
    MCAP(
        ll,par,Float64(level),Float64(span),
        DataFrame(parameter=grid,smoothed=smoothed,quadratic=quadratic),
        smooth_arg_max,b/(2*a),ci,delta,
        nsqrt(se_stat_squared),nsqrt(se_mc_squared),nsqrt(se_total_squared),
        (c=c,a=a,b=b),
    )
end
