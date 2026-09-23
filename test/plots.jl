ENV["GKSwstype"] = "100"

using PartiallyObservedMarkovProcesses
import PartiallyObservedMarkovProcesses as POMP
using DataFrames
using Distributions
using Plots
import Random
using Test

@info h1("plot recipe tests")

@testset verbose=true "plot recipes" begin

    Random.seed!(1953273041)

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

    ptb = @perturbn(@lognormal(a,0.02),@lognormal(k,0.05),@ivp(@lognormal(x0,0.1)))
    mf1 = mif(P;Nmif=5,Np=50,perturbations=ptb,cooling=geometric_cooling(0.5))
    mf2 = mif(P;Nmif=5,Np=50,perturbations=ptb,cooling=hyperbolic_cooling(0.5))
    mon2 = monitor(mf2;Np=50,seed=1,every=2)

    @testset "mif traces" begin
        panels = POMP.mif_panels([mf1],nothing,nothing)
        @test first.(panels) == ["eff.sample.size","cond.logLik","logLik","a","k","x0"]
        panels = POMP.mif_panels([mf2],(:k,),[mon2])
        @test first.(panels) == ["eff.sample.size","cond.logLik","logLik","monitor logLik","k"]
        ## the monitor panel shares the iteration axis of the trace panels
        @test panels[4][2] == [[1.0,3.0,5.0,6.0]]
        @test panels[3][2] == [collect(1.0:6.0)]
        @test_throws r"not a parameter" POMP.mif_panels([mf1],(:bogus,),nothing)
        @test_throws r"no `mif` results" POMP.mif_panels(POMP.MifdPompObject[],nothing,nothing)
        @test_throws r"one `monitor`" POMP.mif_panels([mf1,mf2],nothing,[mon2])
        pl = plot(mf1)
        @test pl isa Plots.Plot
        @test length(pl.subplots) == 6
        savefig(pl,"mif-01.png")
        @test isfile("mif-01.png")
        pl = plot([mf1,mf2];pars=(:a,:k))
        @test length(pl.subplots) == 6   # 5 panels in a 3×2 grid
        @test length(pl.series_list) == 10
        pl = plot(mf2;monitor=mon2)
        @test length(pl.subplots) == 8   # 7 panels in a 4×2 grid
        savefig(pl,"mif-02.png")
        @test isfile("mif-02.png")
    end

    @testset "sliceplot" begin
        d = slice_design(p1; a=[1.2,1.5,1.8], k=[4.0,7.0,10.0])
        s = slice(P,d;Np=50,nreps=2)
        pl = sliceplot(s)
        @test pl isa Plots.Plot
        @test length(pl.subplots) == 2
        savefig(pl,"slice-01.png")
        @test isfile("slice-01.png")
        ## without a standard error column
        pl2 = sliceplot(s[:,Not(:se)])
        @test length(pl2.subplots) == 2
        @test_throws r"`slice` and `loglik` columns" sliceplot(DataFrame(a=[1.0],loglik=[1.0]))
        @test_throws r"data frame" sliceplot(1)
    end

    @testset "mcapplot" begin
        par = collect(range(1.0,3.0,length=30))
        ll = -5 .* (par .- 2.1).^2 .+ 0.3 .* randn(30)
        m = mcap(ll,par)
        pl = mcapplot(m)
        @test pl isa Plots.Plot
        @test length(pl.series_list) == 6
        savefig(pl,"mcap-01.png")
        @test isfile("mcap-01.png")
        @test_throws r"MCAP" mcapplot(ll)
    end

end
