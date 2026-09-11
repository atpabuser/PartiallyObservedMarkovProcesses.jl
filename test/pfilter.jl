using PartiallyObservedMarkovProcesses
import PartiallyObservedMarkovProcesses as POMP
using Distributions
using Random
using Test
using BenchmarkTools

@info h1("pfilter tests")

@testset verbose=true "pfilter" begin

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
    );
    P = P[1];
    @test_throws r"keyword argument .* not assigned" simulate(P,params=(a=1.5,k=7.0))

    P = pomp(
        obs(P),
        times=times(P),
        t0=0,
        rinit=rin,
        rprocess=discrete_time(rlin),
        rmeasure=rmeas,
        logdmeasure=logdmeas
    );
    @test P isa POMP.PompObject

    x0 = rinit(P,params=p1,nsim=10);
    y = obs(P);
    t = times(P);
    x = similar(x0,1,size(x0)...);
    rprocess!(P,x,x0=x0,times=t[1:1],params=[p1])
    @test x0==x[1,:,:]

    Q = pfilter(P,Np=1000,params=p1);
    @test :x ∉ paramsymbs(Q)
    @test Q isa POMP.PfilterdPompObject
    @test occursin(r"PfilterdPompObject .* Np=",sprint(show,Q))
    @test all(Q.x0.==Q.pred[1,:])
    @test_throws r"keyword argument .* not assigned" pfilter(Q,params=(a=1.5,k=7.0));
    @btime pfilter($Q,params=(a=1.5,k=7.0,x0=5.0));
    @btime pfilter($Q,params=(k=7.0,a=1.5,x0=5.0));
    x0 = rinit(Q,nsim=5)
    @test x0 isa Array{<:NamedTuple}
    @test size(x0)==(1,5)
    rinit!(Q,x0)

    d = melt(Q);
    @test size(d)==(21,6)
    @test propertynames(d)==[:time, :y, :x, :ess, :cond_logLik, :resampled]

    P1 = pomp(P;logdmeasure=function (;_...) -Inf end)
    Q1 = pfilter(P1,Np=100,params=p1)
    @test isinf(logLik(Q1))
    @test all(eff_sample_size(Q1).==0)
    @test all(eff_sample_size(Q1).==0)
    @test all(isinf.(cond_logLik(Q1)))
    @test Q1.filt==Q1.pred

    Q2 = [pfilter(Q,Np=100) for _ ∈ 1:5]
    @test logmeanexp(logLik.(Q2)) isa Float64
    @test logmeanexp(logLik.(Q2),se=true) isa @NamedTuple{est::Float64,se::Float64}
    @test logmeanexp(logLik.(Q2),ess=true) isa @NamedTuple{est::Float64,ess::Float64}
    @test logmeanexp(logLik.(Q2),ess=true,se=true) isa @NamedTuple{est::Float64,se::Float64,ess::Float64}
    @test logmeanexp(logLik.(Q2)) > mean(logLik.(Q2))

    ## consistency check: the weighted particle representation preserves
    ## the likelihood -- log-mean-exp of replicate log likelihoods should
    ## agree (within a generous tolerance) whether resampling is performed
    ## at every observation time with equally weighted particles
    ## (trigger=1, target=0), at every observation time with the full
    ## weighted representation carried forward (trigger=1, target=0.5),
    ## or only when triggered by effective-sample-size deficiency
    ## (trigger=0.5, target=0).
    Random.seed!(20260907)
    settings = [(trigger=1.0,target=0.0),(trigger=1.0,target=0.5),(trigger=0.5,target=0.0)]
    lls = [
        logmeanexp([
            logLik(pfilter(P,Np=500,params=p1,trigger=trigger,target=target))
            for _ ∈ 1:20
        ])
        for (trigger,target) ∈ settings
    ]
    @test maximum(lls)-minimum(lls) < 0.5

    ## --- the normalizing constant discarded at partial resampling ---------
    ##
    ## Ancestors are selected with probability qᵢ = wᵢ^(1-β)/S, where
    ## S = Σᵢ wᵢ^(1-β), so the properly weighted representation assigns the
    ## particle selected at position j the importance weight
    ## R_j = w_{A_j}/(J·q_{A_j}) = (S/J)·w_{A_j}^β, whose sample mean is
    ## C = (S/J)·mean_j w_{A_j}^β -- this step's contribution to the
    ## normalizing constant of the unnormalized measure. Renormalizing the
    ## retained weights to unit mean stores R_j/C, so C must multiply the
    ## likelihood accumulator.
    ##
    ## With J = 2 the systematic sweep is driven by a single uniform, so the
    ## ancestry is a piecewise-constant function of it and the expectation
    ## is a finite sum. This is therefore an exact check with no Monte Carlo
    ## tolerance -- the companion to the end-to-end test in test/iid.jl,
    ## which exercises the same property through the full filter.
    let w0 = [1.8,0.2], β = 0.5, g = [1.0,0.0]
        ## `systematic_resample!` returns log C and leaves `w` unit-mean,
        ## so both quantities can be read off a single call.
        step(p_forced) = begin
            α = 1-β
            S = sum(w0.^α)
            wr = [w0[j]^β for j ∈ p_forced]
            m = sum(wr)/2
            C = m*S/2
            ℓ = sum(wr[j]/m*g[p_forced[j]] for j ∈ 1:2)/2
            (C,ℓ)
        end

        ## cumulative selection mass (w₁^α, S); sweep points u and u+S/2
        ## with u ~ Uniform(0,S/2); p[1] = 1 always, p[2] = 1 iff u ≤ w₁^α-S/2
        α = 1-β
        S = sum(w0.^α)
        pr11 = clamp((w0[1]^α-S/2)/(S/2),0.0,1.0)
        @test pr11 ≈ 0.5

        (C11,l11) = step([1,1])
        (C12,l12) = step([1,2])
        @test C11 ≈ 1.2
        @test C12 ≈ 0.8

        exact   = sum(w0.*g)/2
        without = pr11*l11 + (1-pr11)*l12
        with    = pr11*C11*l11 + (1-pr11)*C12*l12

        @test exact ≈ 0.9
        @test with ≈ exact atol=1e-12       # retaining C is exact
        @test without ≈ 0.875 atol=1e-12    # dropping C is biased

        ## the factor is conditionally mean-one, and still shifts the
        ## expectation, because it is correlated with the ancestry
        EC = pr11*C11 + (1-pr11)*C12
        @test EC ≈ 1.0 atol=1e-12
        @test abs(with-EC*without) > 1e-6

        ## C ≡ 1 at both endpoints, so the classical and fully weighted
        ## filters are untouched
        for b ∈ (0.0,1.0)
            a = 1-b
            Sb = sum(w0.^a)
            pr = clamp((w0[1]^a-Sb/2)/(Sb/2),0.0,1.0)
            cs = map(pp -> begin
                wrb = [w0[j]^b for j ∈ pp]
                mb = sum(wrb)/2
                (mb*Sb/2, sum(wrb[j]/mb*g[pp[j]] for j ∈ 1:2)/2)
            end,([1,1],[1,2]))
            wo = pr*cs[1][2] + (1-pr)*cs[2][2]
            @test wo ≈ exact atol=1e-12
        end
    end

    ## the live routine must return log C, not `nothing`
    let w = [1.8,0.2], work = zeros(2), p = zeros(Int,2)
        Random.seed!(4242)
        lc = POMP.systematic_resample!(p,w,work,0.5)
        @test lc isa AbstractFloat
        @test isfinite(lc)
        @test sum(w)/2 ≈ 1.0 atol=1e-12          # left unit-mean
        @test exp(lc) ≈ (p==[1,1] ? 1.2 : 0.8) atol=1e-10
    end

    ## --- a parameter cloud, one set per particle ----------------------------

    ## A degenerate cloud -- every particle given the same parameters -- must
    ## reproduce the ordinary filter. The two paths use transposed array
    ## layouts and different parallel granularity, so they are not promised
    ## to consume the same random draws; the comparison is statistical.
    Random.seed!(20260909)
    reps_s = [logLik(pfilter(P,Np=200,params=p1)) for _ ∈ 1:30]
    reps_c = [logLik(pfilter(P,Np=200,params=fill(p1,200))) for _ ∈ 1:30]
    ls = logmeanexp(reps_s,se=true)
    lc = logmeanexp(reps_c,se=true)
    @test abs(ls.est-lc.est) < 4*sqrt(ls.se^2+lc.se^2)

    ## the cloud is carried and returned, permuted along with the states
    Qc = pfilter(P,Np=100,params=fill(p1,100))
    @test length(paramcloud(Qc))==100
    @test all(q -> keys(q)==keys(p1),paramcloud(Qc))
    @test Qc.Np==100

    ## `coef` reports the parameters of the one stored ancestral lineage,
    ## which is a different object from the whole final cloud
    @test coef(Qc) ∈ fill(p1,100)

    ## no cloud is recorded when a single parameter set was supplied
    @test paramcloud(pfilter(P,Np=10,params=p1)) === nothing

    ## a non-degenerate cloud runs, and the weighted paths accept one too
    Random.seed!(606)
    cloud = [merge(p1,(a=p1.a*exp(0.05*randn()),)) for _ ∈ 1:80]
    Qn = pfilter(P,Np=80,params=cloud)
    @test isfinite(logLik(Qn))
    @test length(paramcloud(Qn))==80
    @test coef(Qn) ∈ cloud
    for (trg,tgt) ∈ ((0.5,0.0),(1.0,0.4),(0.5,0.4))
        Qw = pfilter(P,Np=60,params=cloud[1:60],trigger=trg,target=tgt)
        @test isfinite(logLik(Qw))
        @test length(paramcloud(Qw))==60
    end

    ## `Np` may be omitted, and must agree when given
    @test pfilter(P,params=fill(p1,25)).Np==25
    @test_throws r"must equal" pfilter(P,Np=24,params=fill(p1,25))

    ## the cloud must be nonempty and internally consistent
    @test_throws r"nonempty" pfilter(P,params=typeof(p1)[])
    @test_throws r"same parameter names" pfilter(
        P,params=[p1,(r=1.0,K=2.0)]
    )

    ## `wpfilter` forwards to the same path
    Qwp = wpfilter(P,Np=40,params=fill(p1,40),trigger=0.5)
    @test isfinite(logLik(Qwp))
    @test length(paramcloud(Qwp))==40

    ## melt is unaffected by the presence of a cloud
    dc = melt(Qc)
    @test propertynames(dc)==[:time,:y,:x,:ess,:cond_logLik,:resampled]

end
