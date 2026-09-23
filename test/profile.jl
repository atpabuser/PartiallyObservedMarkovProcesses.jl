using PartiallyObservedMarkovProcesses
import PartiallyObservedMarkovProcesses as POMP
using DataFrames
using Distributions
using Random
using Test

@info h1("profile tests")

@testset verbose=true "profile" begin

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

    ptb = @perturbn(@lognormal(k,0.05),@lognormal(x0,0.05))
    cool = geometric_cooling(0.5)
    d = profile_design(
        a=[1.2,1.35,1.5,1.65,1.8];
        lower=(k=3.0,x0=3.0),upper=(k=10.0,x0=8.0),
        nprof=2,rng=MersenneTwister(5),
    )

    @testset "profile" begin
        Random.seed!(33)
        pr = profile(P,d;Nmif=3,Np=50,perturbations=ptb,cooling=cool,nreps=2,Np_eval=100)
        @test pr isa DataFrame
        @test nrow(pr) == 10
        @test propertynames(pr) == [:a,:k,:x0,:loglik,:se,:ess]
        ## the profiled parameter is untouched, the others have moved
        @test pr.a == d.a
        @test all(pr.k .!= d.k)
        @test all(pr.x0 .!= d.x0)
        @test all(pr.k .> 0) && all(pr.x0 .> 0)
        @test all(isfinite,pr.loglik)
        @test all(pr.ess .≤ 2)
        ## reproducible under the same seed
        Random.seed!(33)
        pr2 = profile(P,d;Nmif=3,Np=50,perturbations=ptb,cooling=cool,nreps=2,Np_eval=100)
        @test pr2 == pr
        ## mcap runs on the profile output
        m = mcap(pr.loglik,pr.a;span=1.0)
        @test m isa MCAP
        @test minimum(pr.a) ≤ m.mle ≤ maximum(pr.a)
    end

    @testset "fixed parameters" begin
        ## perturbing a profiled parameter is refused
        bad = @perturbn(@lognormal(a,0.1),@lognormal(k,0.1))
        @test_throws r"must not be perturbed" profile(P,d;Nmif=1,Np=10,perturbations=bad,cooling=cool)
        badivp = @perturbn(@lognormal(k,0.1),@ivp(@lognormal(a,0.1)))
        @test_throws r"must not be perturbed" profile(P,d;Nmif=1,Np=10,perturbations=badivp,cooling=cool)
        ## a plain data frame without metadata is accepted as a design;
        ## an unperturbed parameter keeps its exact value
        pk = @perturbn(@lognormal(k,0.05))
        pr3 = profile(P,DataFrame(a=[1.5],k=[7.0],x0=[5.0]);Nmif=1,Np=20,perturbations=pk,cooling=cool)
        @test nrow(pr3) == 1
        @test pr3.x0 == [5.0] && pr3.a == [1.5]
        ## parameters absent from the design appear in the output
        pr4 = profile(P,DataFrame(a=[1.5]);Nmif=1,Np=20,perturbations=pk,cooling=cool)
        @test propertynames(pr4) == [:a,:k,:x0,:loglik,:se,:ess]
        @test pr4.x0 == [5.0] && pr4.a == [1.5] && pr4.k[1] != 7.0
    end

    @testset "finding the perturbed parameters leaves the random numbers alone" begin
        @test Set(POMP.perturbed_names(ptb,p1)) == Set([:k,:x0])
        @test Set(POMP.perturbed_names(@perturbn(@ivp(@lognormal(x0,0.1))),p1)) == Set([:x0])
        Random.seed!(44); u = rand()
        Random.seed!(44); POMP.perturbed_names(ptb,p1); @test rand() == u
    end

end
