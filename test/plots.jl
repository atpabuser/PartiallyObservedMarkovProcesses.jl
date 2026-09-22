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

    mf1 = mif2(P;Nmif=5,Np=50,rw_sd=(a=0.02,k=0.05),rw_sd_init=(x0=0.1,),transform=(a=:log,k=:log,x0=:log))
    mf2 = mif2(P;Nmif=5,Np=50,rw_sd=(a=0.02,k=0.05),rw_sd_init=(x0=0.1,),transform=(a=:log,k=:log,x0=:log),Nmonitor=2)

    @testset "mif2 traces" begin
        panels = POMP.mif2_panels([mf1],nothing)
        @test first.(panels) == ["eff.sample.size","cond.logLik","logLik","a","k","x0"]
        panels = POMP.mif2_panels([mf2],(:k,))
        @test first.(panels) == ["eff.sample.size","cond.logLik","logLik","monitor_logLik","k"]
        @test_throws r"not a parameter" POMP.mif2_panels([mf1],(:bogus,))
        @test_throws r"no `mif2` results" POMP.mif2_panels(POMP.Mif2dPompObject[],nothing)
        pl = plot(mf1)
        @test pl isa Plots.Plot
        @test length(pl.subplots) == 6
        savefig(pl,"mif2-01.png")
        @test isfile("mif2-01.png")
        pl = plot([mf1,mf2];pars=(:a,:k))
        ## ess, cond.logLik, logLik, monitor_logLik (mf2 has Nmonitor > 0), a, k
        @test length(pl.subplots) == 6
        ## two objects give two series per panel
        @test length(pl.series_list) == 12
        savefig(pl,"mif2-02.png")
        @test isfile("mif2-02.png")
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
