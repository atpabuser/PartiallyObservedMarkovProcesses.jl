using PartiallyObservedMarkovProcesses
import PartiallyObservedMarkovProcesses as POMP
using Distributions
using Random
using Test

@info h1("wpfilter tests")

@testset verbose=true "wpfilter" begin

    Random.seed!(263260083)

    rin = function(;x0,_...)
        d = Poisson(x0)
        (x=rand(d),)
    end

    rlin = function (;t,a,x,_...)
        d = Poisson(a*x)
        (x=rand(d),)
    end

    rmeas = function (;x,k,_...)
        d = NegativeBinomial(k,k/(k+x))
        (y=rand(d),)
    end

    logdmeas = function (;x,y,k,_...)
        d = NegativeBinomial(k,k/(k+x))
        logpdf(d,y)
    end

    p1 = (a=1.5,k=7.0,x0=5.0);

    P = simulate(
        t0=0,
        times=0:20,
        params=p1,
        rinit=rin,
        rprocess=discrete_time(rlin,dt=1),
        rmeasure=rmeas,
        logdmeasure=logdmeas
    )[1];
    N = length(times(P))

    W = wpfilter(P,Np=1000,params=p1,trigger=1.0)
    @test W isa POMP.WpfilterdPompObject
    @test size(W.filt)==size(W.pred)==size(W.weights)==(N,1000)
    @test length(W.logweights)==1000
    @test length(cond_logLik(W))==N
    @test length(eff_sample_size(W))==N
    @test length(resampled(W))==N
    @test logLik(W)==sum(cond_logLik(W))
    @test occursin(r"WpfilterdPompObject .* Np=.*trigger=",sprint(show,W))

    d = melt(W)
    @test propertynames(d)==[:time,:y,:x,:ess,:cond_logLik,:resampled]
    @test size(d)==(N,6)

    @test_throws r"trigger.*\[0,1\]" wpfilter(P,Np=100,params=p1,trigger=1.5)
    @test_throws r"trigger.*\[0,1\]" wpfilter(P,Np=100,params=p1,trigger=-0.1)
    @test_throws r"Np.*positive" wpfilter(P,Np=0,params=p1)
    @test_throws r"keyword argument .* not assigned" wpfilter(P,Np=100,params=(a=1.5,k=7.0))

    ## trigger=1: resamples (essentially) every step
    W1 = wpfilter(P,Np=1000,params=p1,trigger=1.0)
    @test all(resampled(W1))

    ## trigger=0: never resamples; ESS should decay noticeably
    W0 = wpfilter(P,Np=1000,params=p1,trigger=0.0)
    @test all(.!resampled(W0))
    @test eff_sample_size(W0)[end] < eff_sample_size(W0)[1]
    @test all(W0.filt.==W0.pred)

    ## bit-identity with pfilter at trigger=1, single-threaded only
    if Threads.nthreads()==1
        Random.seed!(42)
        Q = pfilter(P,Np=500,params=p1)
        Random.seed!(42)
        Wt = wpfilter(P,Np=500,params=p1,trigger=1.0)
        @test cond_logLik(Wt)==cond_logLik(Q)
        @test eff_sample_size(Wt)==eff_sample_size(Q)
        @test logLik(Wt)==logLik(Q)
        @test Wt.pred==Q.pred
        @test Wt.filt==Q.filt
    end

    ## carry-forward bookkeeping: with trigger=0, no resampling ever occurs,
    ## so `pred` traces the particles' own uninterrupted trajectories, and
    ## the importance-sampling identity sum(cond_logLik) == logmeanexp(per-particle
    ## path likelihoods) must hold exactly (up to floating-point roundoff).
    Random.seed!(7)
    Wc = wpfilter(P,Np=5,params=p1,trigger=0.0)
    ell = logdmeasure(P,x=Wc.pred,y=obs(P),params=p1)
    per_path = [sum(ell[k,1,i,1] for k in 1:N) for i in 1:5]
    @test sum(cond_logLik(Wc)) ≈ logmeanexp(per_path) atol=1e-8

    ## constant-likelihood invariance: with a flat logdmeasure, the carried
    ## log-weights must stay exactly zero (mean-one) at every trigger.
    Pconst = pomp(P;logdmeasure=function (;_...) -1.234 end)
    for trig ∈ (0.0,0.5,1.0)
        Wk = wpfilter(Pconst,Np=50,params=p1,trigger=trig)
        @test all(cond_logLik(Wk).≈-1.234)
        @test all(Wk.logweights.≈0.0)
    end

    ## degenerate filter (logdmeasure always -Inf)
    Pdeg = pomp(P;logdmeasure=function (;_...) -Inf end)
    Wd = wpfilter(Pdeg,Np=100,params=p1)
    @test isinf(logLik(Wd))
    @test all(eff_sample_size(Wd).==0)
    @test all(isinf.(cond_logLik(Wd)))
    @test all(.!resampled(Wd))
    @test Wd.filt==Wd.pred
    @test all(Wd.logweights.==0)

    ## re-running on a WpfilterdPompObject
    W2 = wpfilter(W1,Np=200)
    @test W2.Np==200
    @test W2.trigger==W1.trigger

end
