using PartiallyObservedMarkovProcesses
import PartiallyObservedMarkovProcesses as POMP
using Test

@info h1("ParameterTransform tests")

@testset verbose=true "partrans" begin

    ## --- round trips ----------------------------------------------------

    let
        pt = ParameterTransform((a=:log,))
        θ = (a=2.3,b=1.0)
        T = @inferred pt.to(θ)
        B = @inferred pt.from(T)
        @test T.a ≈ log(θ.a)
        @test B.a ≈ θ.a
        @test B.b === θ.b
        @test keys(T)==keys(θ)==keys(B)
    end

    let
        pt = ParameterTransform((p=:logit,))
        θ = (p=0.3,q=9.0)
        T = @inferred pt.to(θ)
        B = @inferred pt.from(T)
        @test T.p ≈ log(θ.p/(1-θ.p))
        @test B.p ≈ θ.p
        @test B.q === θ.q
    end

    let
        pt = ParameterTransform((a=:identity,))
        θ = (a=2.3,)
        T = @inferred pt.to(θ)
        B = @inferred pt.from(T)
        @test T.a === θ.a
        @test B.a === θ.a
    end

    ## untagged parameters pass through unchanged, in the original order
    let
        pt = ParameterTransform((b=:log,))
        θ = (a=1.0,b=2.0,c=3.0)
        T = @inferred pt.to(θ)
        @test keys(T)==(:a,:b,:c)
        @test T.a===θ.a && T.c===θ.c
    end

    ## log-barycentric group: a bijection of the interior of the unit
    ## simplex; round trip and the simplex constraint on the recovered
    ## natural-scale values
    let
        pt = ParameterTransform(NamedTuple();logbarycentric=(:x,:y,:z))
        θ = (x=0.2,y=0.3,z=0.5)
        T = @inferred pt.to(θ)
        B = @inferred pt.from(T)
        @test B.x ≈ θ.x && B.y ≈ θ.y && B.z ≈ θ.z
        @test B.x+B.y+B.z ≈ 1.0
        ## the transformation is defined for any positive values, not
        ## only those already summing to one
        θ2 = (x=1.0,y=2.0,z=3.0)
        B2 = pt.from(pt.to(θ2))
        @test B2.x+B2.y+B2.z ≈ 1.0
        @test B2.x/B2.y ≈ θ2.x/θ2.y
    end

    ## --- error cases ------------------------------------------------------

    @test_throws r"must be .:log., .:logit., or .:identity." ParameterTransform((a=:bogus,))
    @test_throws r"must not also be named in .tags." ParameterTransform((a=:log,);logbarycentric=(:a,:b))
    @test_throws r"pairwise disjoint" ParameterTransform(NamedTuple();logbarycentric=((:a,:b),(:b,:c)))

    ## tags naming a parameter absent from the model: caught by mif2's
    ## own validation (not by ParameterTransform, which has no notion of
    ## which parameters the model actually has)
    rin = function(;x0,_...)
        (x=x0,)
    end
    rproc = function (;t,a,x,_...)
        (x=a*x,)
    end
    rmeas = function (;x,_...)
        (y=x,)
    end
    logdmeas = function (;x,y,_...)
        -abs2(x-y)
    end
    P = POMP.pomp(
        t0=0,times=1:5,init_state=(;x=0.0),
        rinit=rin,rprocess=POMP.discrete_time(rproc,dt=1),
        rmeasure=rmeas,logdmeasure=logdmeas,
    )
    @test_throws r"bogus" mif2(
        P;Nmif=1,Np=5,params=(a=1.1,x0=1.0),rw_sd=(a=0.01,),
        transform=(bogus=:log,),
    )

end
