using PartiallyObservedMarkovProcesses
import PartiallyObservedMarkovProcesses as POMP
using Distributions
using Random
using Test

@info h1("mif2 tests")

@testset verbose=true "mif2" begin

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
    variance(v) = sum(abs2,v.-sum(v)/length(v))/length(v)

    ## --- cooling helper -------------------------------------------------

    for α ∈ (0.1,0.5,0.9)
        s = POMP.cooling_setup(:geometric,α,N)
        @test POMP.cooling(:geometric,1,1,N,α,s) ≈ 1.0
        @test POMP.cooling(:geometric,50,N,N,α,s) ≈ α^((50*N-1)/(50*N)) rtol=1e-10
        sh = POMP.cooling_setup(:hyperbolic,α,N)
        @test POMP.cooling(:hyperbolic,1,1,N,α,sh) ≈ 1.0
        @test POMP.cooling(:hyperbolic,50,N,N,α,sh) ≈ α rtol=1e-10
        cs = [POMP.cooling(:geometric,m,n,N,α,s) for m ∈ 1:60 for n ∈ 1:N]
        @test issorted(cs,rev=true)
        cs = [POMP.cooling(:hyperbolic,m,n,N,α,sh) for m ∈ 1:60 for n ∈ 1:N]
        @test issorted(cs,rev=true)
    end
    @test POMP.cooling(:geometric,5,3,N,1.0,0.0)==1.0
    @test POMP.cooling(:hyperbolic,5,3,N,1.0,0.0)==1.0

    ## --- validation errors ------------------------------------------------

    @test_throws r"rw_sd.*names not found" mif2(P;Nmif=1,Np=10,params=p1,rw_sd=(bogus=0.1,))
    @test_throws r"must not share parameter names" mif2(
        P;Nmif=1,Np=10,params=p1,rw_sd=(a=0.1,),rw_sd_init=(a=0.1,)
    )
    @test_throws r"finite, nonnegative" mif2(P;Nmif=1,Np=10,params=p1,rw_sd=(a=-0.1,))
    @test_throws r"cooling_type" mif2(P;Nmif=1,Np=10,params=p1,rw_sd=(a=0.1,),cooling_type=:bogus)
    @test_throws r"cooling_fraction_50.*\(0,1\]" mif2(
        P;Nmif=1,Np=10,params=p1,rw_sd=(a=0.1,),cooling_fraction_50=0.0
    )
    @test_throws r"cooling_fraction_50.*\(0,1\]" mif2(
        P;Nmif=1,Np=10,params=p1,rw_sd=(a=0.1,),cooling_fraction_50=1.5
    )
    @test_throws r"trigger.*\[0,1\]" mif2(P;Nmif=1,Np=10,params=p1,rw_sd=(a=0.1,),trigger=1.5)
    @test_throws r"Np.*positive" mif2(P;Nmif=1,Np=0,params=p1,rw_sd=(a=0.1,))
    @test_throws r"Nmif.*nonnegative" mif2(P;Nmif=-1,Np=10,params=p1,rw_sd=(a=0.1,))
    @test_throws r"must return a .NamedTuple." mif2(
        P;Nmif=1,Np=10,params=p1,rw_sd=(a=0.1,),
        transform=p->[p.a,p.k,p.x0],
    )
    @test_throws r"round-trip failed" mif2(
        P;Nmif=1,Np=10,params=p1,rw_sd=(a=0.1,),
        transform=p->merge(p,(a=log(p.a),)),inverse_transform=identity,
    )
    @test_throws r"floating-point" mif2(P;Nmif=1,Np=10,params=(a=1.5,k=7,x0=5.0),rw_sd=(k=0.1,))

    ## --- structural / unit tests -------------------------------------------

    fit = mif2(P; Nmif=5, Np=200, params=p1, rw_sd=(a=0.02,k=0.02))
    @test fit isa POMP.Mif2dPompObject
    @test keys(coef(fit))==keys(p1)
    @test all(isfinite,values(coef(fit)))
    @test length(traces(fit))==6
    @test traces(fit)[1].iteration==0
    @test isnan(traces(fit)[1].logLik)
    @test keys(traces(fit)[1])==(:iteration,:logLik,keys(p1)...)
    @test [r.iteration for r ∈ traces(fit)]==0:5
    @test length(fit.paramcloud)==length(fit.estcloud)==200
    @test length(cond_logLik(fit))==N
    @test length(eff_sample_size(fit))==N
    @test length(resampled(fit))==N
    @test fit.perturbed_logLik==sum(cond_logLik(fit))
    @test occursin(r"Mif2dPompObject .* Nmif=.*Np=",sprint(show,fit))

    ## fixed parameters (not named in rw_sd) must be untouched exactly:
    ## use k=7.0, an exactly-representable value whose weighted mean over
    ## an unperturbed, never-diverging cloud is exact in floating point.
    fit2 = mif2(P; Nmif=3, Np=100, params=p1, rw_sd=(a=0.02,))
    @test coef(fit2).k === p1.k
    @test coef(fit2).x0 === p1.x0

    ## Nmif = 0: no iterations, coef unchanged, single trace row, and the
    ## per-timestep diagnostics must be well-defined placeholders (NaN/false),
    ## not uninitialized memory.
    fit0 = mif2(P; Nmif=0, Np=50, params=p1, rw_sd=(a=0.02,k=0.02))
    @test coef(fit0)==p1
    @test all(isnan,cond_logLik(fit0))
    @test all(isnan,eff_sample_size(fit0))
    @test all(.!resampled(fit0))
    @test isnan(fit0.perturbed_logLik)
    @test length(traces(fit0))==1

    ## rw_sd_init: perturbs only the first observation time (matching R
    ## pomp's `ivp`), unlike rw_sd which perturbs at every observation time.
    ## This must be a real behavioral difference, not merely "doesn't
    ## crash": with no cooling (cooling_fraction_50=1), a full random walk
    ## over N observation times accumulates N independent perturbations
    ## (variance ~ N*sd^2), while an ivp-only perturbation contributes just
    ## one (variance ~ sd^2). If the implementation instead merged
    ## `rw_sd_init` into every timestep (or dropped it), these would come
    ## out indistinguishable rather than differing by a factor of order N.
    Random.seed!(11)
    fit_ivp = mif2(P; Nmif=1, Np=2000, params=p1, rw_sd=(;), rw_sd_init=(a=0.05,), cooling_fraction_50=1.0)
    Random.seed!(11)
    fit_rw = mif2(P; Nmif=1, Np=2000, params=p1, rw_sd=(a=0.05,), cooling_fraction_50=1.0)
    var_ivp = variance([e.a for e ∈ fit_ivp.estcloud])
    var_rw = variance([e.a for e ∈ fit_rw.estcloud])
    @test var_ivp > 0
    @test var_rw > var_ivp*N/3

    ## disjoint rw_sd/rw_sd_init are both honored simultaneously: `k`
    ## (in rw_sd) should show the ~N-step random-walk spread, `a` (in
    ## rw_sd_init) only the single-step spread.
    fit_both = mif2(
        P; Nmif=1, Np=2000, params=p1,
        rw_sd=(k=0.05,), rw_sd_init=(a=0.05,), cooling_fraction_50=1.0,
    )
    var_k_both = variance([e.k for e ∈ fit_both.estcloud])
    var_a_both = variance([e.a for e ∈ fit_both.estcloud])
    @test var_k_both > var_a_both*N/3

    ## transform/inverse_transform on the log scale
    tr = p -> merge(p,(a=log(p.a),k=log(p.k)))
    itr = p -> merge(p,(a=exp(p.a),k=exp(p.k)))
    fit3 = mif2(P; Nmif=5, Np=200, params=p1, rw_sd=(a=0.02,k=0.02), transform=tr, inverse_transform=itr)
    @test coef(fit3).a > 0 && coef(fit3).k > 0

    ## melt
    d = melt(fit)
    @test propertynames(d)==[:iteration,:logLik,:a,:k,:x0]
    @test size(d)==(6,5)

    ## continuation
    fitc = mif2(fit3; Nmif=3)
    @test fitc.Nmif==fit3.Nmif+3
    @test length(traces(fitc))==length(traces(fit3))+3
    @test isequal(traces(fitc)[1:length(traces(fit3))],traces(fit3))
    @test [r.iteration for r ∈ traces(fitc)]==0:fitc.Nmif
    @test_throws r"Np. cannot be changed" mif2(fit3;Nmif=1,Np=fit3.Np+1)

    ## cloud persistence: the parameter cloud must not collapse to its mean
    ## between mif2 calls. Test this by continuing with rw_sd=0 (no further
    ## perturbation): resampling of an *already-identical* cloud can only
    ## ever preserve that identity (it cannot manufacture diversity), so if
    ## continuation had collapsed the cloud to a single point, feeding it
    ## through a zero-perturbation iteration would give exactly zero
    ## variance. Nonzero variance therefore proves the diverse
    ## post-iteration-1 cloud was carried forward intact.
    Random.seed!(9)
    fitA = mif2(P; Nmif=1, Np=300, params=p1, rw_sd=(a=0.1,k=0.1))
    varA = variance([e.a for e ∈ fitA.estcloud])
    @test varA > 0
    fitB = mif2(fitA; Nmif=1, rw_sd=(a=0.0,k=0.0))
    varB = variance([e.a for e ∈ fitB.estcloud])
    @test varB > 0

    ## with rw_sd all zero (no perturbation), the cloud must equal Np exact
    ## copies of the starting point (on the estimation scale) throughout
    fitz = mif2(P; Nmif=2, Np=20, params=p1, rw_sd=(a=0.0,k=0.0))
    @test all(e->e==fitz.estcloud[1],fitz.estcloud)

    ## Nmonitor
    fitm = mif2(P; Nmif=3, Np=100, params=p1, rw_sd=(a=0.02,k=0.02), Nmonitor=2, Np_monitor=200)
    @test logLik(fitm) isa Float64
    @test isfinite(logLik(fitm))
    @test_throws r"perturbed model" logLik(fit)

    ## multi-threaded chunking edge cases: Np smaller than thread count
    for Np ∈ (1,3,7)
        fitn = mif2(P; Nmif=2, Np=Np, params=p1, rw_sd=(a=0.02,k=0.02))
        @test all(isfinite,values(coef(fitn)))
    end

    ## --- the key falsifiable convergence test ------------------------------
    ## Simulate from a Gompertz model at known truth θ*, start mif2 displaced
    ## from the truth, and require the point estimate to move substantially
    ## closer to the truth (in squared log-scale distance) than the start was.
    ##
    ## Calibrated over 8 seeds (1001-1008) at Nmif=50, Np=1000: observed
    ## ratios d(coef(fit))/d(θ0) ranged 0.0002-0.458 (median ~0.12, max
    ## 0.458). RATIO=0.7 gives a >1.5x margin over the worst observed run.
    gin = function (;X0,_...)
        (;X=X0,)
    end
    gproc = discrete_time(
        function (;t,X,σₚ,r,K,_...)
            s = exp(-r)
            d = LogNormal(s*log(X)+(1-s)*log(K),σₚ)
            (;X=rand(d),)
        end,
        dt=1
    )
    gmeas = function (;X,σₘ,_...)
        d = LogNormal(log(X),σₘ)
        (;pop=rand(d),)
    end
    glogdmeas = function (;pop,X,σₘ,_...)
        logpdf(LogNormal(log(X),σₘ),pop)
    end
    mkgompertz() = pomp(
        t0=0, times=1:100,
        init_state=(;X=zero(Float64)),
        rinit=gin, rprocess=gproc, rmeasure=gmeas, logdmeasure=glogdmeas,
    )
    θstar = (r=0.1,K=1.0,σₚ=0.1,σₘ=0.1,X0=1.0)
    gtr = p -> merge(p,(r=log(p.r),K=log(p.K)))
    gitr = p -> merge(p,(r=exp(p.r),K=exp(p.K)))
    θ0 = merge(θstar,(r=3*θstar.r,K=2*θstar.K))
    dist(θ) = (log(θ.r)-log(θstar.r))^2+(log(θ.K)-log(θstar.K))^2
    RATIO = 0.7
    for seed ∈ 1001:1005
        Random.seed!(seed)
        Pg = simulate(mkgompertz();params=θstar,nsim=1)[1]
        fitg = mif2(
            Pg; Nmif=50, Np=1000, params=θ0,
            rw_sd=(r=0.02,K=0.02), cooling_fraction_50=0.5,
            transform=gtr, inverse_transform=gitr,
        )
        @test dist(coef(fitg)) < RATIO*dist(θ0)
    end

end
