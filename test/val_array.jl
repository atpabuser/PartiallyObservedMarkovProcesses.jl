using PartiallyObservedMarkovProcesses
import PartiallyObservedMarkovProcesses as POMP
using Test

@info h1("testing val_array")

@testset "val_array" begin
    y = fill((a=7,b=99,c="bob"),25,1)
    @test POMP.val_array("yes")==["yes"]
    @test_throws "size mismatch" POMP.val_array(y,11,2)
    @test size(POMP.val_array(3,1,1,1))==(1,1,1,1)
    @test size(POMP.val_array(rand(3,2,2)))==(12,)

    ## A one-dimensional view must pass through as the vector of values it
    ## is. It matches neither the `Vector` method (a view is not an `Array`)
    ## nor the `Array{X,N}` one, so before `AbstractVector` was admitted it
    ## fell through to the scalar fallback and was wrapped as a single
    ## value -- which silently reinterpreted a view of several parameter
    ## sets as one, and then failed the length assertions in `rprocess!`
    ## and `logdmeasure!`.
    ps = [(a=1.0,),(a=2.0,),(a=3.0,)]
    @test POMP.val_array(@view ps[1:2]) === view(ps,1:2)
    @test length(POMP.val_array(@view ps[1:2]))==2
    @test length(POMP.val_array(@view ps[:]))==3
    ## the workhorses accept such a view directly
    @test length(POMP.val_array(view(ps,2:3)))==2
end
