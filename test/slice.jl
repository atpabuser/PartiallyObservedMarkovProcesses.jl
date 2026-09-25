using PartiallyObservedMarkovProcesses
import PartiallyObservedMarkovProcesses as POMP
using DataFrames
using Distributions
using Random
using Statistics: std
using Test

@info h1("slice tests")

@testset verbose=true "slice" begin

    Random.seed!(1832445720)

    rin = function(;x0,_...)
        (x=rand(Poisson(x0)),)
    end
    rlin = function (;t,a,x,_...)
        (x=rand(Poisson(a*x)),)
    end
    rmeas = function (;x,k,_...)
        (y=rand(NegativeBinomial(k,k/(k+x))),)
    end
    logdmeas = function (;x,y,k,_...)
        logpdf(NegativeBinomial(k,k/(k+x)),y)
    end

    p1 = (a=1.5,k=7.0,x0=5.0)

    P = simulate(
        t0=0,
        times=0:20,
        params=p1,
        rinit=rin,
        rprocess=discrete_time(rlin,dt=1),
        rmeasure=rmeas,
        logdmeasure=logdmeas
    )[1]

    @testset "pfilter_loglik" begin
        Random.seed!(11)
        r = pfilter_loglik(P;Np=100,nreps=5)
        @test keys(r) == (:loglik,:se,:ess)
        @test isfinite(r.loglik)
        @test r.se ≥ 0
        @test 1 ≤ r.ess ≤ 5
        ## agrees with logmeanexp of replicate pfilter runs under the same seed
        Random.seed!(11)
        lls = [logLik(pfilter(P;Np=100,params=p1)) for _ ∈ 1:5]
        est = logmeanexp(lls;se=true,ess=true)
        @test r.loglik == est.est
        @test r.se == est.se
        @test r.ess == est.ess
        ## a single replicate has no standard error
        r1 = pfilter_loglik(P;Np=100,nreps=1)
        @test isfinite(r1.loglik) && isnan(r1.se) && r1.ess == 1
        ## all-infinite replicates
        @test POMP.summarize_loglik([-Inf,-Inf]) == (loglik=-Inf,se=NaN,ess=0.0) ||
            (POMP.summarize_loglik([-Inf,-Inf]).loglik == -Inf &&
             isnan(POMP.summarize_loglik([-Inf,-Inf]).se) &&
             POMP.summarize_loglik([-Inf,-Inf]).ess == 0)
        ## a replicate at -Inf is an estimate of zero and is kept
        s = POMP.summarize_loglik([-Inf,-10.0,-12.0])
        @test s.loglik == logmeanexp([-Inf,-10.0,-12.0])
        @test s.loglik < logmeanexp([-10.0,-12.0])
        @test s.ess == logmeanexp([-Inf,-10.0,-12.0];ess=true).ess
        @test_throws r"nreps" pfilter_loglik(P;Np=10,nreps=0)
    end

    @testset "pfilter_loglik is unbiased when some replicates are -Inf" begin
        ## Frozen binary state X ~ Bernoulli(1/2), with g₁ = (0.2,1.8) and
        ## g₂ = (0,1) at X = (0,1): exact likelihood 0.9, and with Np = 2
        ## about a quarter of the replicates are exactly zero.  Dropping
        ## them gave a mean of 1.20.
        Random.seed!(5)
        Pz = pomp(
            [(y=0.0,),(y=0.0,)];
            t0=0.0, times=[1.0,2.0], params=(dummy=0.0,),
            rinit=(;_...) -> (x = rand() < 0.5 ? 1.0 : 0.0,),
            rprocess=discrete_time((;x,_...) -> (x=x,),dt=1.0),
            logdmeasure=(;t,x,_...) -> t == 1 ? log(x == 1 ? 1.8 : 0.2) : log(x == 1 ? 1.0 : 0.0),
        )
        z = [exp(pfilter_loglik(Pz;Np=2,nreps=10).loglik) for _ ∈ 1:5000]
        @test abs(sum(z)/length(z)-0.9) < 5*std(z)/sqrt(length(z))
    end

    @testset "slice" begin
        d = slice_design(p1; a=[1.2,1.5,1.8], k=[4.0,7.0])
        Random.seed!(22)
        s = slice(P,d;Np=100,nreps=3)
        @test s isa DataFrame
        @test nrow(s) == 5
        @test propertynames(s) == [:a,:k,:x0,:slice,:loglik,:se,:ess]
        @test s[:,1:4] == d
        @test all(isfinite,s.loglik)
        @test all(s.ess .≤ 3)
        ## the truth should beat the far-off values along the `a` slice
        ia = findall(==(:a),s.slice)
        @test s.loglik[ia[2]] > s.loglik[ia[1]]
        @test s.loglik[ia[2]] > s.loglik[ia[3]]
        ## reproducible under the same seed
        Random.seed!(22)
        s2 = slice(P,d;Np=100,nreps=3)
        @test s2 == s
        ## design columns must be parameters
        bad = copy(d); bad.zz = ones(nrow(d))
        @test_throws r"not parameters of the model" slice(P,bad;Np=10)
        @test_throws r"no rows" slice(P,d[1:0,:];Np=10)
        ## a design need not carry every parameter
        s3 = slice(P,DataFrame(a=[1.4,1.6]);Np=50)
        @test propertynames(s3) == [:a,:loglik,:se,:ess]
    end

end
