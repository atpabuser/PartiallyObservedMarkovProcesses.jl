using PartiallyObservedMarkovProcesses
using Test
using Crayons

h1 = crayon"bold blue"
h2 = s -> crayon"!bold light_yellow"("- "*s)

@testset verbose=true "POMP.jl" begin
    ## plots.jl loads Plots, whose GR backend must be loaded before R's
    ## graphics libraries (ggsave in the tests below) pull in the system
    ## glib; the other order fails to load Glib_jll.
    include("plots.jl")
    include("basic.jl")
    include("errors.jl")
    include("val_array.jl")
    include("helpers.jl")
    include("eulermultinomial.jl")
    include("bake.jl")
    include("melt.jl")
    include("gompertz.jl")
    include("gompertz_kalman.jl")
    include("brown.jl")
    include("sir.jl")
    include("rmca.jl")
    include("drmca.jl")
    include("flow.jl")
    include("trajmatch.jl")
    include("pfilter.jl")
    include("iid.jl")
    include("wpfilter.jl")
    include("partrans.jl")
    include("perturb.jl")
    include("mif2.jl")
    include("design.jl")
    include("profile.jl")
    include("mcap.jl")
    include("speed1.jl")
end
