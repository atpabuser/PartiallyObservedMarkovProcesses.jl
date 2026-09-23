using PartiallyObservedMarkovProcesses
import PartiallyObservedMarkovProcesses as POMP
using PartiallyObservedMarkovProcesses.Examples
using Random
using Test

@info h1("triggered and power-tempered resampling: exact identities")

@testset verbose=true "weighted filter identities" begin

    Random.seed!(263260083)
    P = gompertz()
    p1 = (r=4.5,K=210.0,σₚ=0.7,σₘ=0.1,X0=150.0)
    N = length(times(P))

    ## trigger = 1: the effective sample size never exceeds Np, so
    ## resampling occurs at every observation after the first. At the
    ## first, every particle sits at X0 with equal weight, and rounding
    ## can put the effective sample size a hair above Np.
    @test all(resampled(pfilter(P;Np=200,params=p1,trigger=1.0,target=0.0))[2:end])

    ## trigger = 0: no resampling ever, so the particles keep their own
    ## uninterrupted paths, and the likelihood estimate is exactly the
    ## importance-sampling average of the path likelihoods
    Random.seed!(7)
    W = pfilter(P;Np=5,params=p1,trigger=0.0,target=0.0)
    @test !any(resampled(W))
    @test W.filt == W.pred
    ell = logdmeasure(P;x=W.pred,params=p1)
    per_path = [sum(ell[k,1,i] for k ∈ 1:N) for i ∈ 1:5]
    @test logLik(W) ≈ logmeanexp(per_path) atol=1e-8

    ## a flat measurement density: every conditional likelihood equals it,
    ## whatever the resampling settings
    Pflat = pomp(P;logdmeasure=function (;_...) -1.234 end)
    for (trigger,target) ∈ ((0.0,0.0),(0.5,0.0),(1.0,0.0),(1.0,0.5),(0.5,0.75))
        Wk = pfilter(Pflat;Np=50,params=p1,trigger,target)
        @test all(cond_logLik(Wk) .≈ -1.234)
    end

    ## a degenerate filter: every particle impossible
    Pdeg = pomp(P;logdmeasure=function (;_...) -Inf end)
    Wd = pfilter(Pdeg;Np=100,params=p1,trigger=0.5,target=0.5)
    @test isinf(logLik(Wd))
    @test all(eff_sample_size(Wd) .== 0)

end
