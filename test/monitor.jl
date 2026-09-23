using PartiallyObservedMarkovProcesses
import PartiallyObservedMarkovProcesses as POMP
using PartiallyObservedMarkovProcesses.Examples
using DataFrames
using Random
using Test

@info h1("hyperbolic cooling, monitor, and resampled")

@testset verbose=true "mif additions" begin

    @testset "hyperbolic_cooling" begin
        c = hyperbolic_cooling(0.5)
        @test c(0) == 1
        @test c(50) ≈ 0.5
        a = 50*0.1/0.9
        @test hyperbolic_cooling(0.1)(7) ≈ a/(a+7)
        ## slower than geometric: same end point at 50, more left later
        @test hyperbolic_cooling(0.1)(200) > geometric_cooling(0.1)(200)
        ## `start` shifts the schedule for a continued computation
        c4 = hyperbolic_cooling(0.5;start=4)
        @test [c4(i) for i ∈ 0:5] == [c(i+4) for i ∈ 0:5]
        @test all(hyperbolic_cooling(1.0)(i) == 1 for i ∈ 0:10)
        @test_throws r"frac" hyperbolic_cooling(0.0)
        @test_throws r"frac" hyperbolic_cooling(1.5)
        @test_throws r"start" hyperbolic_cooling(0.5;start=-1)
    end

    P = gompertz()
    p1 = (r=4.5,K=210.0,σₚ=0.7,σₘ=0.1,X0=150.0)
    ptb = @perturbn(@lognormal(K,0.1),@lognormal(σₚ,0.1))
    Random.seed!(3)
    A = mif(P;params=p1,Np=200,Nmif=4,perturbations=ptb,cooling=hyperbolic_cooling(0.5))
    B = mif(A;Nmif=3,cooling=hyperbolic_cooling(0.5;start=4))

    @testset "resampled" begin
        @test resampled(A) isa Vector{Bool}
        @test length(resampled(A)) == length(times(A))
        @test resampled(A) == resampled(A.pfobj)
    end

    @testset "monitor" begin
        Random.seed!(9); u = rand(); Random.seed!(9)
        m = monitor([A,B];Np=300,nreps=2,every=2,seed=1)
        ## the default random-number stream is left where it was
        @test rand() == u
        ## numbered as in `traces` and counted on across the continuation:
        ## the start, every 2nd iteration after it, and the last
        @test m.iteration == [1,3,5,7,8]
        @test propertynames(m) == [:iteration,:loglik,:se,:ess,:r,:K,:σₚ,:σₘ,:X0]
        @test all(isfinite,m.loglik)
        ## the evaluation points are the trace's point estimates; the
        ## continuation's iteration 0 (a copy of A's end) is dropped
        tA = traces(A); tB = traces(B)
        @test m.K[1:3] == tA.K[[1,3,5]]
        @test m.K[4:5] == tB.K[[3,4]]
        ## reproducible from its seed, and a different seed differs
        @test monitor([A,B];Np=300,nreps=2,every=2,seed=1) == m
        @test monitor([A,B];Np=300,nreps=2,every=2,seed=2).loglik != m.loglik
        ## each value is a fixed-parameter evaluation at its point
        Random.seed!(1)
        @test m.loglik == [
            pfilter_loglik(P;Np=300,nreps=2,params=NamedTuple(r[[:r,:K,:σₚ,:σₘ,:X0]])).loglik
            for r ∈ eachrow(m)
        ]
        ## settings are recorded
        @test metadata(m,"Np") == 300 && metadata(m,"nreps") == 2
        @test metadata(m,"every") == 2 && metadata(m,"seed") == 1
        ## a single run
        @test monitor(A;Np=100,seed=1).iteration == traces(A).iteration == 1:5
        @test_throws r"every" monitor(A;Np=100,seed=1,every=0)
        @test_throws r"no `mif` results" monitor(POMP.MifdPompObject[];Np=100,seed=1)
    end

end
