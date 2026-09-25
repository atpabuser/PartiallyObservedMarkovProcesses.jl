using PartiallyObservedMarkovProcesses
import PartiallyObservedMarkovProcesses as POMP
using DataFrames
using RCall
using Random
using Test

@info h1("mcap tests")

@testset verbose=true "mcap" begin

    Random.seed!(1517486)
    par = collect(range(1.0,3.0,length=40))
    ll = -5 .* (par .- 2.1).^2 .+ 0.3 .* randn(40)
    grid = collect(range(minimum(par),maximum(par),length=1000))

    @testset "loess against R (direct surface)" begin
        fit = POMP.loess(par,ll;span=0.75)
        sm = fit.(grid)
        @rput par ll grid
        R"""
        f_direct <- loess(ll ~ par, span=0.75, control=loess.control(surface="direct"))
        sm_direct <- predict(f_direct, newdata=grid)
        f_default <- loess(ll ~ par, span=0.75)
        sm_default <- predict(f_default, newdata=grid)
        """
        @rget sm_direct sm_default
        @test maximum(abs.(sm .- sm_direct)) < 1e-10
        ## R's default interpolating surface differs only slightly
        @test maximum(abs.(sm .- sm_default)) < 0.05
        ## span > 1 uses every point with an inflated bandwidth
        fit2 = POMP.loess(par,ll;span=1.5)
        R"""
        f2 <- loess(ll ~ par, span=1.5, control=loess.control(surface="direct"))
        sm2 <- predict(f2, newdata=grid)
        """
        @rget sm2
        @test maximum(abs.(fit2.(grid) .- sm2)) < 1e-10
        @test_throws r"same length" POMP.loess(par,ll[1:5])
        @test_throws r"finite" POMP.loess(par,vcat(ll[1:end-1],NaN))
        @test_throws r"span\*n" POMP.loess(par,ll;span=0.05)
    end

    @testset "mcap against R pomp::mcap (direct surface)" begin
        ## pomp::mcap with its loess call switched to the direct surface,
        ## which is what the Julia port computes; everything else verbatim.
        R"""
        library(pomp)
        mcap_direct <- function (logLik, parameter, level = 0.95, span = 0.75, Ngrid = 1000) {
            smooth_fit <- loess(logLik ~ parameter, span = span, control=loess.control(surface="direct"))
            parameter_grid <- seq(min(parameter), max(parameter), length.out = Ngrid)
            smoothed_logLik <- predict(smooth_fit, newdata = parameter_grid)
            smooth_arg_max <- parameter_grid[which.max(smoothed_logLik)]
            dist <- abs(parameter - smooth_arg_max)
            included <- dist < sort(dist)[trunc(span * length(dist))]
            maxdist <- max(dist[included])
            weights <- numeric(length(parameter))
            weights[included] <- (1 - (dist[included]/maxdist)^3)^3
            quadratic_fit <- lm(logLik ~ a + b, weights = weights,
                data = data.frame(logLik = logLik, b = parameter, a = -parameter^2))
            b <- unname(coef(quadratic_fit)["b"]); a <- unname(coef(quadratic_fit)["a"])
            m <- vcov(quadratic_fit)
            var_b <- m["b", "b"]; var_a <- m["a", "a"]; cov_ab <- m["a", "b"]
            se_mc_squared <- (1/(4 * a * a)) * (var_b - (2 * b/a) * cov_ab + (b * b/a/a) * var_a)
            se_stat_squared <- 1/2/a
            delta <- qchisq(level, df = 1) * (a * se_mc_squared + 0.5)
            logLik_diff <- max(smoothed_logLik) - smoothed_logLik
            ci <- range(parameter_grid[logLik_diff < delta])
            list(mle = smooth_arg_max, ci = ci, delta = delta,
                 se_stat = sqrt(se_stat_squared), se_mc = sqrt(se_mc_squared),
                 se = sqrt(se_mc_squared + se_stat_squared),
                 quadratic_max = b/(2*a), a = a, b = b,
                 c = unname(coef(quadratic_fit)[1]),
                 smoothed = smoothed_logLik)
        }
        md <- mcap_direct(ll, par)
        mp <- pomp::mcap(ll, par)
        """
        @rget md mp
        m = mcap(ll,par)
        @test m isa MCAP
        @test m.mle ≈ md[:mle] atol=1e-12
        @test m.ci[1] ≈ md[:ci][1] atol=1e-12
        @test m.ci[2] ≈ md[:ci][2] atol=1e-12
        @test m.delta ≈ md[:delta] rtol=1e-10
        @test m.se_stat ≈ md[:se_stat] rtol=1e-10
        @test m.se_mc ≈ md[:se_mc] rtol=1e-8
        @test m.se ≈ md[:se] rtol=1e-10
        @test m.quadratic_max ≈ md[:quadratic_max] rtol=1e-10
        @test m.coefs.a ≈ md[:a] rtol=1e-10
        @test m.coefs.b ≈ md[:b] rtol=1e-10
        @test m.coefs.c ≈ md[:c] rtol=1e-10
        @test maximum(abs.(m.fit.smoothed .- md[:smoothed])) < 1e-10
        @test m.fit.parameter == grid
        @test m.fit.quadratic ≈ m.coefs.c .+ m.coefs.b .* grid .- m.coefs.a .* grid.^2
        @test m.level == 0.95 && m.span == 0.75
        @test m.logLik == ll && m.parameter == par
        ## and close to pomp::mcap as shipped (interpolating surface)
        step = grid[2]-grid[1]
        @test abs(m.mle - mp[:mle]) ≤ 10*step
        @test abs(m.ci[1] - mp[:ci][1]) ≤ 10*step
        @test abs(m.ci[2] - mp[:ci][2]) ≤ 10*step
        @test m.se_stat ≈ mp[:se_stat] rtol=0.01
        @test m.se_mc ≈ mp[:se_mc] rtol=0.01
        @test m.delta ≈ mp[:delta] rtol=0.01
        ## the interval brackets the truth and the estimate
        @test m.ci[1] < 2.1 < m.ci[2]
        @test m.ci[1] ≤ m.mle ≤ m.ci[2]
        @test m.se ≈ sqrt(m.se_stat^2+m.se_mc^2)
        @test occursin("mle=",sprint(show,m))
    end

    @testset "mcap options and errors" begin
        m1 = mcap(ll,par;level=0.9,span=0.5,Ngrid=200)
        @test m1.level == 0.9 && nrow(m1.fit) == 200
        m2 = mcap(ll,par;level=0.99)
        @test m2.ci[2]-m2.ci[1] > mcap(ll,par;level=0.9).ci[2]-mcap(ll,par;level=0.9).ci[1]
        ## integer inputs are accepted
        @test mcap(round.(Int,10 .* ll),round.(Int,10 .* par);span=1.0) isa MCAP
        @test_throws r"same length" mcap(ll,par[1:10])
        @test_throws r"finite" mcap(vcat(ll[1:end-1],-Inf),par)
        @test_throws r"level" mcap(ll,par;level=1.0)
        @test_throws r"Ngrid" mcap(ll,par;Ngrid=1)
        @test_throws r"span\*length" mcap(ll,par;span=0.01)
        ## too few points in the quadratic window: NaN results, with a warning
        few = @test_logs (:warn,r"carry weight") match_mode=:any mcap(ll[1:5],par[1:5];span=1.0)
        @test isnan(few.se) && all(isnan,few.ci)
        @test isfinite(few.mle)
        ## a convex set of points gives no standard errors, with a warning
        mc = @test_logs (:warn,r"not concave") mcap(-ll,par)
        @test isnan(mc.se_stat) && isnan(mc.se)
        @test all(isnan,mc.ci)
    end

    @testset "mcap does not depend on the units of the parameter" begin
        x = collect(range(1.0,3.0,length=40))
        y = -5 .* (x .- 2.1).^2 .+ 0.3 .* sin.(1:40)
        m1 = mcap(y,x)
        for c ∈ (1e-8,1e8)
            mc = mcap(y,c .* x)
            @test collect(mc.ci) ./ c ≈ collect(m1.ci) rtol=1e-8
            @test mc.mle/c ≈ m1.mle rtol=1e-8
            @test mc.se/c ≈ m1.se rtol=1e-6
            @test mc.se_mc/c ≈ m1.se_mc rtol=1e-6
            @test mc.delta ≈ m1.delta rtol=1e-6
            @test mc.coefs.a*c^2 ≈ m1.coefs.a rtol=1e-6
        end
    end

end
