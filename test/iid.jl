using PartiallyObservedMarkovProcesses
import PartiallyObservedMarkovProcesses as POMP
using Distributions
using Random
using Statistics
using Test

## `h1` is defined in the package's test/helpers.jl; this lets the file
## also be run on its own.
@isdefined(h1) || (h1(s) = s)

@info h1("pfilter tests on iid model")

@testset verbose=true "iid" begin

    Random.seed!(263260083)

    rin = function(;σ,_...)
        d = Normal(0,1)
        (x=rand(d),)
    end

    rproc = function (;t,x,σ,_...)
        d = Normal(0,1)
        (x=rand(d),)
    end

    rmeas = function (;x,τ,_...)
        d = Normal(x,1)
        (y=rand(d),)
    end

    logdmeas = function (;x,y,τ,_...)
        d = Normal(x,τ)
        logpdf(d,y)
    end

    p1 = (σ=1.0,τ=1.0);

    P = simulate(
        t0=0,
        times=0:100,
        params=p1,
        rinit=rin,
        rprocess=discrete_time(rproc,dt=1),
        rmeasure=rmeas,
        logdmeasure=logdmeas
    )[1];

    loglik_iid(;y,σ,τ,_...) = begin
        d = Normal(0,sqrt(σ^2+τ^2))
        logpdf(d,y)
    end

    llexact = map(obs(P)) do x
        loglik_iid(;x...,p1...)
    end |> sum

    ll = logmeanexp([logLik(pfilter(P,Np=10000,params=p1,trigger=0.3,target=0.1)) for _ in 1:20],se=true,ess=true)
    @test abs(ll.est-llexact) < 4*ll.se

    ## NOTE.  The latent process above has no memory: xₜ is drawn afresh
    ## from N(0,1) at every step, independently of xₜ₋₁.  The one-step
    ## predictive density h(x) = ∫f(x′|x)g(yₜ₊₁|x′)dx′ is therefore the
    ## same for every particle, and when h is constant across the cloud
    ## the power-renormalized resampling is exactly unbiased whatever it
    ## does with the discarded mass.  This test is consequently blind to
    ## the `target > 0` bookkeeping, at any number of replicates.  The
    ## test below supplies the missing case.

end

@info h1("unbiasedness of the power-renormalized filter")

@testset verbose=true "power resampling is unbiased" begin

    ## Frozen binary latent state: X ~ Bernoulli(1/2) and Xₜ = X for all t,
    ## so the surviving particle determines the entire future -- the case
    ## the iid model cannot produce.  Two measurement potentials:
    ##
    ##     g₁(1) = 1.8   g₁(0) = 0.2
    ##     g₂(1) = 1.0   g₂(0) = 0.0
    ##
    ## The exact likelihood is Z = E[g₁(X)g₂(X)] = (1/2)(1.8)(1.0) = 0.9.
    ##
    ## With Np = 2 and target = 1/2, hand calculation gives E[Ẑ] = 0.8875
    ## when the mass discarded in renormalizing the retained weights to
    ## unit mean is not credited to the conditional log likelihood, and
    ## E[Ẑ] = 0.9 when it is.

    Random.seed!(263260083)

    rin   = function(;_...); (x = rand() < 0.5 ? 1.0 : 0.0,); end
    rproc = function(;x,_...); (x=x,); end
    ldm   = function(;t,x,y,_...)
        t == 1 ? log(x == 1.0 ? 1.8 : 0.2) : log(x == 1.0 ? 1.0 : 0.0)
    end

    P = pomp(
        [(y=0.0,),(y=0.0,)];
        t0 = 0.0,
        times = [1.0,2.0],
        params = (dummy=0.0,),
        rinit = rin,
        rprocess = discrete_time(rproc,dt=1.0),
        logdmeasure = ldm,
    )

    zexact = 0.9
    nrep = 200_000

    ## E[Ẑ] and its standard error, on the natural scale.  Ẑ ∈ [0,1.8]
    ## here, so a plain average is well behaved; logmeanexp's jack-knife
    ## standard error is O(nrep²) and impractical at this many replicates.
    meanZ(;Np,trigger,target) = begin
        s = 0.0; ss = 0.0
        for _ ∈ 1:nrep
            z = exp(logLik(pfilter(P;Np,trigger,target)))
            s += z; ss += z*z
        end
        mu = s/nrep
        (est=mu, se=sqrt(max(ss/nrep-mu^2,0.0)/nrep))
    end

    ## control: ordinary resampling, target = 0, must be unbiased
    r0 = meanZ(Np=2,trigger=1.0,target=0.0)
    @test abs(r0.est-zexact) < 4*r0.se

    ## the case at issue
    for target ∈ [0.25,0.5,0.75]
        r = meanZ(Np=2,trigger=1.0,target=target)
        @info "target=$target: E[Ẑ]=$(round(r.est,digits=6)) ± $(round(r.se,digits=6)), exact=$zexact"
        @test abs(r.est-zexact) < 4*r.se
    end

    ## The bias, if present, is Θ(1/Np), so it shrinks as the cloud
    ## grows.  Reported rather than asserted: at this many replicates the
    ## comparison is no longer decisive for Np ≳ 8.
    for Np ∈ [8,32,128]
        r = meanZ(Np=Np,trigger=1.0,target=0.5)
        @info "Np=$Np: E[Ẑ]=$(round(r.est,digits=6)) ± $(round(r.se,digits=6)), exact=$zexact"
    end

end
