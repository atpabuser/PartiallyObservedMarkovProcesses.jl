using PartiallyObservedMarkovProcesses
using PartiallyObservedMarkovProcesses.Examples
using Random
using Test

@info h1("Gompertz model: exact Kalman likelihood check")

## The Gompertz state-space model reduces to a linear-Gaussian state-space
## model on the log scale, so its likelihood can be computed exactly (up to
## a change-of-variables constant) by the scalar Kalman filter, with no
## Monte Carlo error. This gives an exact target against which the
## particle filter's log-likelihood estimate can be checked.
##
## Writing Z_t = log(X_t), the process model
##   X_t = X_{t-1}^S K^(1-S) eps_t,   S = exp(-r),   eps_t ~ LogNormal(0,σₚ),
## becomes the linear-Gaussian autoregression
##   Z_t = (1-S) log(K) + S Z_{t-1} + w_t,   w_t ~ Normal(0,σₚ²),
## and the measurement model
##   pop_t ~ LogNormal(Z_t,σₘ)
## becomes, on the log scale,
##   log(pop_t) = Z_t + v_t,   v_t ~ Normal(0,σₘ²).
## `rinit` sets X_0 to the parameter X0 with no added noise, and the first
## observation time in the built-in data coincides with t0, so no process
## step intervenes before the first observation: the filtering
## distribution of Z at the first observation time is a point mass at
## log(X0), i.e. a Normal with zero variance. Every subsequent step
## applies one Gompertz recursion (dt=1) before the next observation.
## The `discrete_time` step function itself ignores its `dt` argument
## (it uses S=exp(-r) and passes σₚ bare to the LogNormal, with no
## sqrt(dt) rescaling), so σₚ is exactly the per-step log-scale standard
## deviation whenever, as here, `discrete_time` is called with dt=1 over
## unit-spaced annual observation times.
##
## The following is the textbook scalar Kalman filter recursion (predict,
## then update on the Gaussian innovation), returning the exact
## conditional log likelihood of log(pop_1),...,log(pop_N) under this
## linear-Gaussian state-space model.
kalman_loglik_logy = function (logy,r,K,σₚ,σₘ,X0)
    S = exp(-r)
    a = (1-S)*log(K)
    Q = σₚ^2
    R = σₘ^2
    m = log(X0)   # filtering mean of Z_1, before assimilating obs 1
    v = 0.0        # filtering variance of Z_1: point mass, since X_1=X0
    ll = 0.0
    for t ∈ eachindex(logy)
        if t > 1
            ## predict: one Gompertz step of the linear-Gaussian recursion
            m = a + S*m
            v = S^2*v + Q
        end
        ## innovation and its (exactly Gaussian) conditional log likelihood
        fvar = v + R
        innov = logy[t] - m
        ll += -0.5*(log(2π*fvar) + innov^2/fvar)
        ## update: filtering distribution of Z_t given obs up to time t
        gain = v/fvar
        m = m + gain*innov
        v = v - gain^2*fvar
    end
    ll
end

@testset verbose=true "Gompertz: exact Kalman likelihood check" begin

    P = gompertz()
    p1 = (r=4.5,K=210.0,σₚ=0.7,σₘ=0.1,X0=150.0)
    y = [o.pop for o ∈ obs(P)]
    logy = log.(Float64.(y))

    ## exact conditional log likelihood of \log(pop_{1:N}) under the
    ## linear-Gaussian reduction of the Gompertz state-space model
    ll_logy = kalman_loglik_logy(logy,p1.r,p1.K,p1.σₚ,p1.σₘ,p1.X0)
    @test isfinite(ll_logy)

    ## `logdmeasure` in the Gompertz model evaluates the LogNormal density
    ## of pop_t itself, not of log(pop_t). Since pop_t=exp(log(pop_t)),
    ## the two densities are related by the Jacobian of this change of
    ## variables, f_pop(y) = f_logpop(log y)/y, so
    ##   loglik(pop_1:N) = loglik(log(pop_1:N)) - sum_t log(pop_t).
    jacobian = -sum(logy)
    ll_y = ll_logy + jacobian
    @test isfinite(ll_y)

    ## the particle filter (classical bootstrap filter: trigger=1,
    ## target=0, the current defaults) estimates loglik(pop_1:N),
    ## exactly this quantity, up to Monte Carlo error that vanishes as
    ## Np grows.
    Random.seed!(20260907)
    Nps = [200,1000,5000]
    nreps = 30
    lme = [
        logmeanexp([logLik(pfilter(P,Np=Np,params=p1)) for _ ∈ 1:nreps],se=true)
        for Np ∈ Nps
    ]
    gap = [x.est - ll_logy for x ∈ lme]
    se = [x.se for x ∈ lme]

    ## primary check: a genuine defect in the filter (or in the model
    ## definition) would make the gap between the particle filter's
    ## estimate and the exact Kalman conditional log likelihood drift
    ## systematically with Np -- e.g. the familiar downward bias of the
    ## log of a Monte Carlo average of a likelihood scales like
    ## -Var/Np and would shrink towards zero as Np grows -- whereas
    ## a mere bookkeeping constant (such as an error in the Jacobian
    ## above) would shift `gap` by the same fixed amount at every Np.
    ## The tolerance is set from the jackknife Monte Carlo standard error
    ## at the smallest (noisiest) Np tested.
    tol_spread = 6*maximum(se)
    @test maximum(gap)-minimum(gap) < tol_spread

    ## secondary, less critical check: `gap` should also be close in
    ## absolute terms to the Jacobian constant derived above, confirming
    ## that no such fixed bookkeeping error is in fact present.
    for (g,s) ∈ zip(gap,se)
        @test abs(g-jacobian) < 6*s
    end

end
