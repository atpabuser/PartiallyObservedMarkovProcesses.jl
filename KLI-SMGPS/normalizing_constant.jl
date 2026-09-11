## The normalizing constant discarded by power-tempered resampling.
##
## Ancestors are selected with probability qᵢ = wᵢ^(1-β)/S, where
## S = Σᵢ wᵢ^(1-β) and the carried weights wᵢ have unit mean. Selecting
## from q rather than from the weights themselves changes the sampling
## measure, so the properly weighted representation assigns the particle
## selected at position j the importance weight
##
##     R_j = w_{A_j}/(J·q_{A_j}) = (S/J)·w_{A_j}^β,
##
## whose sample mean is
##
##     C = (1/J) Σⱼ R_j = (S/J)·m,      m = mean_j w_{A_j}^β.
##
## C is the resampling step's contribution to the normalizing constant
## of the unnormalized measure. Renormalizing the retained weights to
## unit mean stores V_j = R_j/C, so C must multiply the likelihood
## accumulator or it is lost.
##
## E[C | w] = 1 exactly, since E[m | w] = Σᵢ qᵢwᵢ^β = (Σᵢ wᵢ)/S = J/S.
## That is what makes dropping C look harmless, and it is not: C is a
## function of the selected ancestors, hence correlated with everything
## those ancestors go on to generate, so E[C] = 1 does not give
## E[C·ℓ] = E[ℓ].
##
## This file settles the question by exact enumeration rather than by
## simulation. An earlier version of this script compared the two
## estimators by Monte Carlo on the linear-Gaussian reduction of the
## Gompertz model and found no difference; that check had no power (the
## bias is Θ(1/J) per step and vanishes when the one-step predictive
## density is constant across the cloud), and acting on its null result
## was a mistake. Exact calculation is the right instrument here.

using Printf

## ---------------------------------------------------------------------
## Exact enumeration, J = 2.
##
## With J = 2 the systematic-resampling sweep is driven by a single
## uniform, so the map from that uniform to the ancestry is piecewise
## constant with finitely many pieces and the expectation is a finite
## sum. Cumulative selection mass is (w₁^α, w₁^α + w₂^α) with α = 1-β,
## the sweep points are u and u + S/2 for u ~ Uniform(0, S/2), and
## p[1] = 1 always, while p[2] = 1 iff u ≤ w₁^α - S/2.
enumerate_J2(w::Vector{Float64}, β::Float64, g::Vector{Float64}) = begin
    @assert length(w) == 2 == length(g)
    α = 1-β
    c1 = w[1]^α
    S  = c1 + w[2]^α
    du = S/2

    ## P(p = (1,1)): the second sweep point u + du still falls below c1
    pr11 = clamp((c1-du)/du, 0.0, 1.0)

    contribution(p::Vector{Int}) = begin
        wr = [w[p[j]]^β for j ∈ 1:2]
        m  = sum(wr)/2
        C  = m*S/2
        ℓ  = sum(wr[j]/m*g[p[j]] for j ∈ 1:2)/2   # unit-mean weights
        (C, ℓ)
    end

    (C11,l11) = contribution([1,1])
    (C12,l12) = contribution([1,2])

    exact    = sum(w.*g)/2                       # (1/J) Σᵢ wᵢ gᵢ
    without  = pr11*l11      + (1-pr11)*l12
    with     = pr11*C11*l11  + (1-pr11)*C12*l12
    (; pr11, C11, C12, exact, without, with)
end

## The case used in the correspondence and in test/iid.jl.
w = [1.8, 0.2]      # unit mean
β = 0.5
g = [1.0, 0.0]      # future contribution from each state

r = enumerate_J2(w, β, g)

println("J = 2,  w = $w,  β = $β,  g = $g")
@printf("  P(ancestry (1,1))      = %.6f\n", r.pr11)
@printf("  C on (1,1), (1,2)      = %.6f, %.6f\n", r.C11, r.C12)
@printf("  exact  (1/J)Σ wᵢgᵢ     = %.6f\n", r.exact)
@printf("  E[·] dropping C        = %.6f   (bias %+.6f)\n",
        r.without, r.without-r.exact)
@printf("  E[·] retaining C       = %.6f   (bias %+.6f)\n",
        r.with, r.with-r.exact)
println()

@assert isapprox(r.with, r.exact; atol=1e-12) "retaining C must be exact"
@assert abs(r.without-r.exact) > 1e-6 "dropping C must be biased here"

## ---------------------------------------------------------------------
## The factor is conditionally mean-one, and still biases the estimate.
## Reported side by side because this is the step where the argument for
## dropping C goes wrong.
EC = r.pr11*r.C11 + (1-r.pr11)*r.C12
covCl = r.with - EC*r.without
@printf("E[C]             = %.6f\n", EC)
@printf("Cov(C, ℓ)        = %+.6f   <- nonzero, which is the whole point\n", covCl)
println()

@assert isapprox(EC, 1.0; atol=1e-12) "E[C] = 1 exactly"
@assert abs(covCl) > 1e-6 "C must be correlated with ℓ"

## ---------------------------------------------------------------------
## C vanishes at the endpoints, so the ordinary bootstrap filter and the
## fully weighted filter are both untouched; the issue is strictly the
## partially retained-weight case.
println("bias against β:")
for b ∈ (0.0, 0.25, 0.5, 0.75, 1.0)
    rb = enumerate_J2(w, b, g)
    @printf("  β = %.2f   dropping C %+.6f   retaining C %+.6f\n",
            b, rb.without-rb.exact, rb.with-rb.exact)
end
println()

for b ∈ (0.0, 1.0)
    rb = enumerate_J2(w, b, g)
    @assert isapprox(rb.without, rb.exact; atol=1e-12) "C ≡ 1 at β = $b"
end

## ---------------------------------------------------------------------
## The bias is Θ(1/J), which is why it is invisible at the particle
## counts one would actually use -- and why a Monte Carlo comparison at
## J = 64 on a smooth model cannot settle the question.
println("all assertions passed")
