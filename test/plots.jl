using PartiallyObservedMarkovProcesses
import PartiallyObservedMarkovProcesses as POMP
using DataFrames
using Distributions
using AlgebraOfGraphics
using CairoMakie
import Random
using Test

@info h1("plots (AlgebraOfGraphics extension)")

@testset verbose=true "plots" begin

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

    ext = Base.get_extension(PartiallyObservedMarkovProcesses,:PartiallyObservedMarkovProcessesAoGExt)
    @test !isnothing(ext)

    ## what a panel of a drawn figure actually shows: its plots of a given
    ## kind (:scatter, :lines, :errorbars, :vlines, :hlines), by their data
    shown(fg, i, kind) = [p[1][] for p ∈ fg.grid[i].axis.scene.plots if CairoMakie.Makie.plotkey(p) == kind]
    pts(x, y) = [CairoMakie.Point2(a,b) for (a,b) ∈ zip(x,y)]

    @testset "traceplot" begin
        d, order = ext.trace_data([mf1],nothing,nothing)
        @test order == ["logLik","a","k","x0"]
        @test Set(d.variable) == Set(order)
        ## the last trace row has no mif log likelihood
        @test count(==("logLik"),d.variable) == 5
        d, order = ext.trace_data([mf2],(:k,),[mon2])
        @test order == ["logLik","monitor logLik","k"]
        ## the monitor shares the iteration axis of the traces
        @test d.iteration[d.variable .== "monitor logLik"] == [1,3,5,6]
        @test d.iteration[d.variable .== "k"] == 1:6
        @test_throws r"not a parameter" ext.trace_data([mf1],(:bogus,),nothing)
        @test_throws r"no `mif` results" ext.trace_data(POMP.MifdPompObject[],nothing,nothing)
        @test_throws r"one `monitor`" ext.trace_data([mf1,mf2],nothing,[mon2])
        fg = traceplot(mf1)
        @test fg isa AlgebraOfGraphics.FigureGrid
        @test size(fg.grid) == (2,2)
        ## the first panel is the mif log likelihood against the iteration
        t = traces(mf1)
        @test shown(fg,1,:lines) == [pts(t.iteration[1:end-1],t.logLik[1:end-1])]
        save("mif-01.png",fg)
        @test isfile("mif-01.png")
        fg = traceplot([mf1,mf2];pars=(:a,:k))
        @test size(fg.grid) == (2,2)
        fg = traceplot(mf2;monitor=mon2)
        ## the second panel is the monitor, at its own iterations
        @test shown(fg,CartesianIndex(1,2),:lines) == [pts(mon2.iteration,mon2.loglik)]
        save("mif-02.png",fg)
        @test isfile("mif-02.png")
        @test_throws r"result of `mif`" traceplot(1)
    end

    @testset "sliceplot" begin
        d = slice_design(p1; a=[1.2,1.5,1.8], k=[4.0,7.0,10.0])
        s = slice(P,d;Np=50,nreps=2)
        fg = sliceplot(s)
        @test fg isa AlgebraOfGraphics.FigureGrid
        @test size(fg.grid) == (1,2)
        save("slice-01.png",fg)
        @test isfile("slice-01.png")
        ## the points and error bars drawn, on a small known slice: a
        ## missing standard error gives no error bar
        s0 = DataFrame(a=[1.0,2.0,1.5,1.5],k=[7.0,7.0,5.0,9.0],slice=[:a,:a,:k,:k],
            loglik=[-10.0,-12.0,-11.0,-13.0],se=[0.5,0.25,NaN,1.0])
        f0 = sliceplot(s0)
        @test shown(f0,1,:scatter) == [pts([1.0,2.0],[-10.0,-12.0])]
        @test shown(f0,2,:scatter) == [pts([5.0,9.0],[-11.0,-13.0])]
        @test [[v[3] for v ∈ e] for e ∈ shown(f0,1,:errorbars)] == [[0.5,0.25]]
        @test [[(v[1],v[3]) for v ∈ e] for e ∈ shown(f0,2,:errorbars)] == [[(9.0,1.0)]]
        ## without a standard error column there are no error bars
        f1 = sliceplot(s0[:,Not(:se)])
        @test size(f1.grid) == (1,2)
        @test isempty(shown(f1,1,:errorbars)) && isempty(shown(f1,2,:errorbars))
        ## options are passed to `draw`
        @test sliceplot(s;axis=(width=200,height=150)) isa AlgebraOfGraphics.FigureGrid
        @test_throws r"`slice` and `loglik` columns" sliceplot(DataFrame(a=[1.0],loglik=[1.0]))
        @test_throws r"data frame" sliceplot(1)
    end

    @testset "mcapplot" begin
        par = collect(range(1.0,3.0,length=30))
        ll = -5 .* (par .- 2.1).^2 .+ 0.3 .* randn(30)
        m = mcap(ll,par)
        fg = mcapplot(m)
        @test fg isa AlgebraOfGraphics.FigureGrid
        ## the profile points, the smoothed and quadratic fits, the
        ## estimate, both ends of the interval, and the cutoff
        @test shown(fg,1,:scatter) == [pts(m.parameter,m.logLik)]
        fits = shown(fg,1,:lines)
        @test pts(m.fit.parameter,m.fit.smoothed) ∈ fits
        @test pts(m.fit.parameter,m.fit.quadratic) ∈ fits
        @test sort(reduce(vcat,shown(fg,1,:vlines))) ≈ sort([m.mle,m.ci...])
        @test only(only(shown(fg,1,:hlines))) ≈ maximum(m.fit.smoothed)-m.delta
        ## a profile without an interval draws neither interval nor cutoff
        mc = @test_logs (:warn,r"not concave") match_mode=:any mcap(-ll,par)
        fc = mcapplot(mc)
        @test reduce(vcat,shown(fc,1,:vlines)) == [mc.mle]
        @test isempty(shown(fc,1,:hlines))
        save("mcap-01.png",fg)
        @test isfile("mcap-01.png")
        @test_throws r"MCAP" mcapplot(ll)
    end

end
