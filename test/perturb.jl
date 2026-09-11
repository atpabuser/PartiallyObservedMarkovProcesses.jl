using PartiallyObservedMarkovProcesses
import PartiallyObservedMarkovProcesses as POMP
using Distributions
using Random
using Statistics
using Test

@isdefined(h1) || (h1(s) = s)
@isdefined(h2) || (h2(s) = s)

@info h1("perturbation kernels")

@testset verbose=true "perturb" begin

    ## --- the scalar kernels -------------------------------------------------

    @testset "scalar kernels" begin

        for (k,scale,x) ∈ (
            (normal_rw(),     :identity, 1.5),
            (lognormal_rw(),  :log,      1.5),
            (logitnormal_rw(),:logit,    0.3),
        )
            ## K3: a zero standard deviation is a no-op in value
            @test k(x,0.0,1.0) ≈ x rtol=1e-14
            @test k(x,0.0,0.5) ≈ x rtol=1e-14
            ## the declared averaging scale
            @test POMP.mean_scale(k) === scale
        end

        ## K2: the number of draws does not depend on the value of `sd`.
        ## If it did, the draw following a zero-sd call would differ from
        ## the one following a nonzero-sd call.
        for k ∈ (normal_rw(),lognormal_rw(),logitnormal_rw())
            Random.seed!(1234); k(0.3,0.0,1.0); a = randn()
            Random.seed!(1234); k(0.3,0.7,1.0); b = randn()
            @test a == b
        end

        ## K4: each kernel maps its support into itself. Checked over many
        ## draws at a standard deviation large enough to leave the support
        ## were it not preserved exactly.
        Random.seed!(99)
        @test all(lognormal_rw()(1.0,3.0,1.0) > 0 for _ ∈ 1:10_000)
        @test all(0 < logitnormal_rw()(0.5,5.0,1.0) < 1 for _ ∈ 1:10_000)

        ## the perturbation is centred on the declared scale
        Random.seed!(7)
        v = [lognormal_rw()(2.0,0.3,1.0) for _ ∈ 1:200_000]
        @test mean(log.(v)) ≈ log(2.0) atol=4*0.3/sqrt(200_000)
        Random.seed!(7)
        v = [normal_rw()(2.0,0.3,1.0) for _ ∈ 1:200_000]
        @test mean(v) ≈ 2.0 atol=4*0.3/sqrt(200_000)

        ## Student-t: heavier tails than the Gaussian at matched scale
        Random.seed!(11)
        g = [normal_rw()(0.0,1.0,1.0) for _ ∈ 1:100_000]
        Random.seed!(11)
        s = [student_rw(3;scale=:identity)(0.0,1.0,1.0) for _ ∈ 1:100_000]
        @test maximum(abs,s) > maximum(abs,g)
        @test POMP.mean_scale(student_rw(3;scale=:logit)) === :logit

        @test_throws r"degrees of freedom" student_rw(0)
        @test_throws r"scale" student_rw(3;scale=:bogus)

    end

    ## --- `PerKeyRW` type stability ------------------------------------------

    @testset "PerKeyRW inference" begin

        θ12 = NamedTuple{ntuple(i->Symbol("p",i),12)}(ntuple(i->1.0+i/100,12))
        sd12 = NamedTuple{ntuple(i->Symbol("p",i),12)}(ntuple(i->0.01,12))

        for nk ∈ (2,5,12)
            ks = ntuple(i->Symbol("p",i),nk)
            kers = NamedTuple{ks}(ntuple(i->lognormal_rw(),nk))
            sd = NamedTuple{ks}(ntuple(i->0.01,nk))
            K = POMP.PerKeyRW(kers)
            @test @inferred(K(θ12,sd,1.0)) isa NamedTuple
            @test keys(K(θ12,sd,1.0))==ks
        end

        ## a closure as an escape hatch for one parameter, with the rest
        ## still specialized: the kernels live in a field, not a type
        ## parameter, so a non-isbits entry is admissible
        mine = (x,s,c) -> x*exp(c*s*randn())
        Kmix = POMP.PerKeyRW((p1=lognormal_rw(),p2=mine))
        @test @inferred(Kmix(θ12,(p1=0.01,p2=0.01),1.0)) isa NamedTuple

        ## K1: only the keys of `sd` are returned
        K = POMP.PerKeyRW((p1=lognormal_rw(),p2=normal_rw()))
        r = K(θ12,(p1=0.01,p2=0.01),1.0)
        @test keys(r)==(:p1,:p2)

    end

    ## --- integration with mif2 ----------------------------------------------

    Random.seed!(263260083)
    rin = function(;x0,_...); (x=rand(Poisson(x0)),); end
    rlin = function (;t,a,x,_...); (x=rand(Poisson(a*x)),); end
    rmeas = function (;x,k,_...); (y=rand(NegativeBinomial(k,k/(k+x))),); end
    logdmeas = function (;x,y,k,_...); logpdf(NegativeBinomial(k,k/(k+x)),y); end

    p1 = (a=1.5,k=7.0,x0=5.0)
    P = simulate(
        t0=0,times=0:20,params=p1,
        rinit=rin,rprocess=discrete_time(rlin,dt=1),
        rmeasure=rmeas,logdmeasure=logdmeas,
    )[1]
    N = length(times(P))

    @testset "mif2 with a kernel" begin

        ## a per-key lognormal kernel keeps the cloud positive and averages
        ## geometrically, so the stored clouds satisfy paramcloud = exp(estcloud)
        Random.seed!(21)
        f = mif2(P; Nmif=2, Np=40, params=p1, rw_sd=(a=0.02,k=0.02),
                 perturb=(a=lognormal_rw(),k=lognormal_rw()))
        @test all(q -> q.a > 0 && q.k > 0,f.paramcloud)
        @test all(
            i -> isapprox(f.paramcloud[i].a,exp(f.estcloud[i].a);rtol=1e-12),
            eachindex(f.paramcloud)
        )
        @test f.perturb isa NamedTuple

        ## the point estimate is the natural-scale image of the
        ## estimation-scale weighted mean, so it stays positive
        @test coef(f).a > 0 && coef(f).k > 0

        ## A bare function kernel reproduces the per-key form when it is the
        ## same map -- but only if the averaging scale is made to agree too.
        ## The per-key form composes that scale from the kernels' own
        ## declarations (here `a` and `k` on the log scale, `x0` untouched);
        ## a function is opaque, so its averaging scale comes from
        ## `transform`, which must therefore be given to match.
        Random.seed!(21)
        g = mif2(P; Nmif=2, Np=40, params=p1, rw_sd=(a=0.02,k=0.02),
                 transform=(a=:log,k=:log),
                 perturb=(t,sd,c)->(a=t.a*exp(c*sd.a*randn()),k=t.k*exp(c*sd.k*randn())))
        @test coef(g).a ≈ coef(f).a rtol=1e-12
        @test coef(g).k ≈ coef(f).k rtol=1e-12

        ## Left to its default `transform`, the same function kernel averages
        ## arithmetically instead, and the point estimate genuinely differs.
        ## This is the mismatch the per-key form makes unconstructible.
        Random.seed!(21)
        g2 = mif2(P; Nmif=2, Np=40, params=p1, rw_sd=(a=0.02,k=0.02),
                  perturb=(t,sd,c)->(a=t.a*exp(c*sd.a*randn()),k=t.k*exp(c*sd.k*randn())))
        @test !isapprox(coef(g2).k,coef(f).k;rtol=1e-12)

        ## continuation inherits the kernel
        f2 = mif2(f; Nmif=1)
        @test f2.Nmif == 3
        @test f2.perturb isa NamedTuple
        @test all(q -> q.a > 0,f2.paramcloud)

        ## --- ivp against an explicit vector, on the kernel path -------------
        ## The mirror of the same check on the default path. This is what a
        ## kernel violating K2 would break.
        Random.seed!(31)
        i1 = mif2(P; Nmif=2, Np=30, params=p1,
                  rw_sd=(x0=ivp(0.05),a=0.02),
                  perturb=(x0=lognormal_rw(),a=lognormal_rw()))
        Random.seed!(31)
        i2 = mif2(P; Nmif=2, Np=30, params=p1,
                  rw_sd=(x0=[0.05;zeros(N-1)],a=0.02),
                  perturb=(x0=lognormal_rw(),a=lognormal_rw()))
        @test isequal(traces(i1),traces(i2))
        @test isequal(i1.paramcloud,i2.paramcloud)

        ## --- the kernel and the tag agree statistically ---------------------
        ## `lognormal_rw()` and the default kernel under `transform=(a=:log,)`
        ## are the same perturbation, but not bit-identical: the default adds
        ## on the log scale and exponentiates, the kernel multiplies by an
        ## exponential. Compare the spread of the resulting cloud, not the
        ## draws.
        Random.seed!(41)
        c1 = mif2(P; Nmif=1, Np=400, params=p1, rw_sd=(a=0.05,),
                  transform=(a=:log,k=:log,x0=:log))
        Random.seed!(41)
        c2 = mif2(P; Nmif=1, Np=400, params=p1, rw_sd=(a=0.05,),
                  perturb=(a=lognormal_rw(),))
        s1 = std(log.([q.a for q ∈ c1.paramcloud]))
        s2 = std(log.([q.a for q ∈ c2.paramcloud]))
        @test s1 ≈ s2 rtol=0.3

    end

    @testset "kernel validation errors" begin

        ## every perturbed parameter needs a kernel
        @test_throws r"must name every parameter" mif2(
            P;Nmif=1,Np=10,params=p1,rw_sd=(a=0.02,k=0.02),
            perturb=(a=lognormal_rw(),)
        )
        ## and a kernel for an unperturbed parameter would never fire
        @test_throws r"not named in" mif2(
            P;Nmif=1,Np=10,params=p1,rw_sd=(a=0.02,),
            perturb=(a=lognormal_rw(),k=lognormal_rw())
        )
        ## entries must be kernels
        @test_throws r"per-parameter kernel" mif2(
            P;Nmif=1,Np=10,params=p1,rw_sd=(a=0.02,),perturb=(a=0.5,)
        )
        ## a per-key kernel cannot be combined with a functional transform,
        ## since the averaging scale cannot be composed into an opaque pair
        @test_throws r"functional" mif2(
            P;Nmif=1,Np=10,params=p1,rw_sd=(a=0.02,),
            perturb=(a=lognormal_rw(),),
            transform=p->(a=log(p.a),k=p.k,x0=p.x0),
            inverse_transform=p->(a=exp(p.a),k=p.k,x0=p.x0)
        )
        ## nor may the averaging scale be specified twice
        @test_throws r"doubly specified" mif2(
            P;Nmif=1,Np=10,params=p1,rw_sd=(a=0.02,),
            perturb=(a=lognormal_rw(),),transform=(a=:log,)
        )
        ## a starting value outside the kernel's support is rejected up front
        @test_throws r"outside the support" mif2(
            P;Nmif=1,Np=10,params=merge(p1,(a=1.5,)),rw_sd=(a=0.02,),
            perturb=(a=logitnormal_rw(),)
        )

    end

    ## --- the log-barycentric group survives a per-key kernel elsewhere ------
    ## A group member must not also be given a kernel; the check has to look
    ## at the group members, not only at the scalar tags.
    @testset "log-barycentric interaction" begin

        ## `a` is left untagged here, so only the group members are declared
        pt = ParameterTransform((;);logbarycentric=(:k,:x0))

        ## a kernel on a group *member* clashes: the check must look at the
        ## group's members, not only at the scalar tags
        @test_throws r"doubly specified" POMP.normalize_average(
            (k=lognormal_rw(),),pt,pt.to,pt.from
        )
        @test_throws r"doubly specified" POMP.normalize_average(
            (x0=lognormal_rw(),),pt,pt.to,pt.from
        )

        ## a kernel on a parameter that is neither tagged nor in the group is
        ## fine, and the group survives into the composed averaging transform
        to,from = POMP.normalize_average((a=lognormal_rw(),),pt,pt.to,pt.from)
        θ = (a=2.0,k=0.25,x0=0.75)
        rt = from(to(θ))
        @test rt.a ≈ θ.a rtol=1e-12
        @test rt.k+rt.x0 ≈ 1.0 rtol=1e-12          # still on the simplex
        @test rt.k ≈ θ.k rtol=1e-12

        ## a scalar tag also clashes with a kernel on the same parameter
        pt2 = ParameterTransform((a=:log,))
        @test_throws r"doubly specified" POMP.normalize_average(
            (a=lognormal_rw(),),pt2,pt2.to,pt2.from
        )

    end

end
