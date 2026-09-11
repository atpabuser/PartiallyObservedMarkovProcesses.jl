## A self-contained demonstration that power-tempered resampling loses a
## normalizing constant, and of why the obvious test for it cannot see it.
##
## Run from the package root:
##
##     julia --project=. KLI-SMGPS/demonstration.jl
##
## Nothing here is a unit test; the assertions in test/iid.jl and
## test/pfilter.jl do that job. This script exists to be read alongside
## its output.
##
## ---------------------------------------------------------------------
## THE QUANTITY AT ISSUE
##
## Power-tempered resampling (`target` = β) selects ancestor A_j with
## probability qᵢ = wᵢ^(1-β)/S, where S = Σᵢ wᵢ^(1-β), and carries the
## residual weight w^β forward with the selected particle. Selecting from
## q rather than from the weights themselves changes the sampling
## measure, so the properly weighted representation assigns the particle
## selected at position j the importance weight
##
##     R_j = w_{A_j}/(J·q_{A_j}) = (S/J)·w_{A_j}^β,
##
## whose sample mean is
##
##     C = (1/J) Σⱼ R_j = (S/J)·m,      m = mean_j w_{A_j}^β.
##
## C is this resampling step's contribution to the normalizing constant
## of the unnormalized measure. Renormalizing the retained weights to
## unit mean -- which the filter does, because every other routine
## assumes unit-mean weights on entry -- stores R_j/C, so C has to
## multiply the likelihood accumulator or it is gone.
##
## C is conditionally mean-one: E[m | w] = Σᵢ qᵢwᵢ^β = (Σᵢ wᵢ)/S = J/S,
## so E[C | w] = 1 exactly. That is what makes dropping it look harmless.
## It is not, because C is a function of the selected ancestors and hence
## correlated with everything those ancestors go on to generate.

using PartiallyObservedMarkovProcesses
import PartiallyObservedMarkovProcesses as POMP
using Random, Statistics, Printf

hr() = println("-"^72)
head(s) = (println(); hr(); println(s); hr())

## =====================================================================
head("1.  An exact calculation, no simulation")
## =====================================================================
##
## With J = 2 the systematic sweep is driven by a single uniform, so the
## ancestry is a piecewise-constant function of it and every expectation
## is a finite sum. Take unit-mean weights w = (1.8, 0.2) and β = 1/2.
## Then q = (0.75, 0.25) and the sweep gives ancestry (1,1) or (1,2),
## each with probability exactly 1/2.
##
## Write hᵢ for the one-step predictive density at particle i,
##
##     h(x) = ∫ f(x′|x) g(y|x′) dx′,
##
## the expected future likelihood contribution from a particle sitting at
## x. The next increment the filter computes is ℓ = (1/J) Σⱼ Vⱼ h(x^{A_j})
## with Vⱼ the unit-mean retained weights.

const W0 = [1.8,0.2]
const Β  = 0.5

outcomes(h1,h2) = begin
    α = 1-Β
    S = sum(W0.^α)
    rows = map(([1,1],[1,2])) do p
        wr = [W0[j]^Β for j ∈ p]
        m  = sum(wr)/2
        V  = wr ./ m
        C  = m*S/2
        ℓ  = (V[1]*[h1,h2][p[1]] + V[2]*[h1,h2][p[2]])/2
        (p=p, V=V, C=C, ℓ=ℓ)
    end
    Eℓ   = 0.5*rows[1].ℓ + 0.5*rows[2].ℓ         # C dropped
    ECℓ  = 0.5*rows[1].C*rows[1].ℓ + 0.5*rows[2].C*rows[2].ℓ   # C retained
    tgt  = (W0[1]*h1 + W0[2]*h2)/2               # correctly weighted
    EC   = 0.5*rows[1].C + 0.5*rows[2].C
    (rows=rows, Eℓ=Eℓ, ECℓ=ECℓ, tgt=tgt, EC=EC, cov=ECℓ-EC*Eℓ)
end

r = outcomes(1.0,0.0)     # a future that depends maximally on which particle survives
println("w = $W0 (unit mean),  β = $Β,  h = (1.0, 0.0)\n")
for o ∈ r.rows
    @printf("  ancestry %s   retained V = (%.2f, %.2f)   C = %.2f   ℓ = %.4f\n",
            o.p, o.V[1], o.V[2], o.C, o.ℓ)
end
println()
@printf("  correctly weighted value  (1/J)Σ wᵢhᵢ = %.6f\n", r.tgt)
@printf("  E[ℓ]   with C dropped               = %.6f   (bias %+.6f)\n",
        r.Eℓ, r.Eℓ-r.tgt)
@printf("  E[Cℓ]  with C retained              = %.6f   (bias %+.6f)\n",
        r.ECℓ, r.ECℓ-r.tgt)
println()
@printf("  E[C]        = %.6f   <- mean one, exactly\n", r.EC)
@printf("  Cov(C, ℓ)   = %+.6f   <- and yet not independent\n", r.cov)
println("\n  E[C] = 1 does not give E[Cℓ] = E[ℓ]. The gap is exactly Cov(C,ℓ).")

## The whole dependence on the model sits in h₂ - h₁:
println("\n  bias as a function of the two predictive densities:")
for (h1,h2) ∈ ((1.0,0.0),(1.0,0.9),(1.0,1.0),(1.0,2.0))
    o = outcomes(h1,h2)
    @printf("    h = (%.2f, %.2f)   bias = %+.6f   0.025·(h₂-h₁) = %+.6f\n",
            h1, h2, o.Eℓ-o.tgt, 0.025*(h2-h1))
end
println("""
  So  bias = 0.025·(h₂ - h₁)  for this cloud: proportional to how much the
  future differs between the particles. If h is the same at every particle
  the bias is zero -- not small, zero. Remember that.""")

## =====================================================================
head("2.  A model that cannot see it: an i.i.d. latent state")
## =====================================================================
##
## This is the natural first thing to test with, and it is useless here.
## For an i.i.d. latent process f(x′|x) = f(x′), so
##
##     h(x) = ∫ f(x′) g(y|x′) dx′
##
## does not depend on x at all. Then, because V is unit-mean by
## construction,
##
##     ℓ = (1/J) Σⱼ Vⱼ h(x^{A_j}) = h · (1/J) Σⱼ Vⱼ = h,
##
## which is DETERMINISTIC given the cloud. Cov(C,ℓ) = 0 exactly, and by
## the formula above the bias is exactly zero -- at every J, every β, and
## every number of replicates.

rin_iid   = function(;_...); (x=randn(),); end
rproc_iid = function(;_...); (x=randn(),); end
ldm_iid   = function(;x,y,τ,_...); -0.5*((y-x)^2/τ^2 + log(2π*τ^2)); end

P_iid = pomp(
    [(y=0.3,),(y=-0.2,),(y=0.5,)];
    t0=0.0, times=[1.0,2.0,3.0], params=(τ=1.0,),
    rinit=rin_iid, rprocess=discrete_time(rproc_iid,dt=1.0),
    logdmeasure=ldm_iid,
)
## exact: y_t are independent N(0, 1+τ²)
z_iid = sum(-0.5*((o.y)^2/(1+1.0^2) + log(2π*(1+1.0^2))) for o ∈ obs(P_iid))
@printf("exact log Z = %.6f\n\n", z_iid)

Random.seed!(20260909)
for β ∈ (0.0,0.5)
    reps = [logLik(wpfilter(P_iid;Np=2,trigger=1.0,target=β)) for _ ∈ 1:200_000]
    zs = exp.(reps)
    @printf("  β = %.2f   E[Ẑ]/Z = %.6f ± %.6f\n",
            β, mean(zs)/exp(z_iid), std(zs)/sqrt(length(zs))/exp(z_iid))
end
println("""
  Agreement, and it means nothing: the bias is identically zero on this
  model whether or not C is retained. A test here cannot fail. This is the
  first testset in test/iid.jl, kept deliberately, with a note saying so.""")

## =====================================================================
head("3.  A model that can: a frozen latent state")
## =====================================================================
##
## The opposite extreme. X ~ Bernoulli(1/2) and Xₜ = X for all t, so the
## surviving particle determines the entire future and h varies by a
## factor of 9 across the cloud. Two measurement potentials,
##
##     g₁(1) = 1.8   g₁(0) = 0.2        g₂(1) = 1.0   g₂(0) = 0.0
##
## give the exact likelihood Z = E[g₁(X)g₂(X)] = (1/2)(1.8)(1.0) = 0.9.

rin_fr   = function(;_...); (x = rand() < 0.5 ? 1.0 : 0.0,); end
rproc_fr = function(;x,_...); (x=x,); end
ldm_fr   = function(;t,x,y,_...)
    t == 1 ? log(x == 1.0 ? 1.8 : 0.2) : log(x == 1.0 ? 1.0 : 0.0)
end

P_fr = pomp(
    [(y=0.0,),(y=0.0,)];
    t0=0.0, times=[1.0,2.0], params=(dummy=0.0,),
    rinit=rin_fr, rprocess=discrete_time(rproc_fr,dt=1.0),
    logdmeasure=ldm_fr,
)
const ZFR = 0.9

## The filter as it now stands retains C.
Random.seed!(20260909)
nrep = 200_000
for β ∈ (0.0,0.25,0.5,0.75)
    zs = [exp(logLik(wpfilter(P_fr;Np=2,trigger=1.0,target=β))) for _ ∈ 1:nrep]
    μ,σ = mean(zs), std(zs)/sqrt(nrep)
    @printf("  β = %.2f   E[Ẑ] = %.6f ± %.6f   exact %.1f   (%+.1f sd)\n",
            β, μ, σ, ZFR, (μ-ZFR)/σ)
end

## And the same filter with C discarded, reimplemented here in a dozen
## lines so the comparison is visible rather than asserted. This is what
## the code did before the correction, and what a filter that renormalizes
## the retained weights without accounting for the normalizing constant
## does in general.
uncorrected(β, rng) = begin
    x = [rand(rng) < 0.5 ? 1.0 : 0.0 for _ ∈ 1:2]
    w = ones(2)
    ll = 0.0
    for t ∈ 1:2
        g = [t == 1 ? (xi == 1.0 ? 1.8 : 0.2) : (xi == 1.0 ? 1.0 : 0.0) for xi ∈ x]
        w .*= g
        inc = sum(w)/2
        inc == 0 && return -Inf
        ll += log(inc)
        w ./= inc                              # unit mean
        α = 1-β
        cum = cumsum(w.^α); s = cum[end]
        du = s/2; u = -du*rand(rng); i = 1
        p = map(1:2) do j
            u += du
            while (u > cum[i] && i < 2); i += 1; end
            p_ = i
        end
        wr = [w[p[j]]^β for j ∈ 1:2]
        x = x[p]
        w = wr ./ (sum(wr)/2)                  # ... and C is dropped here
    end
    ll
end

println()
rng = MersenneTwister(20260909)
for β ∈ (0.0,0.25,0.5,0.75)
    zs = [exp(uncorrected(β,rng)) for _ ∈ 1:nrep]
    μ,σ = mean(zs), std(zs)/sqrt(nrep)
    @printf("  β = %.2f   C dropped: E[Ẑ] = %.6f ± %.6f   exact %.1f   (%+.1f sd)\n",
            β, μ, σ, ZFR, (μ-ZFR)/σ)
end
println("""
  β = 0 agrees either way, because C ≡ 1 there and the step is ordinary
  resampling. Between the endpoints it does not.""")

## The measured figure is not the 0.875 of section 1, and the difference
## is worth accounting for rather than waving at: section 1 analysed a
## cloud already at w = (1.8, 0.2), whereas here the cloud is drawn.
##
## With J = 2 the two initial states are each Bernoulli(1/2), so
##
##   both at x=1   (prob 1/4):  weights (1.8,1.8), equal, so C = 1 and no
##                              bias is possible. Ẑ = 1.8·1 = 1.8.
##   both at x=0   (prob 1/4):  g₂ = 0 at both, so Ẑ = 0.
##   one of each   (prob 1/2):  weights (1.8,0.2) after the first
##                              increment of 1.0 -- exactly section 1.
##
## Only the mixed case has unequal weights, so only half the runs can
## carry any bias at all:
let mixed_correct = 0.9, mixed_dropped = 0.875
    pred_correct = 0.25*1.8 + 0.25*0.0 + 0.5*mixed_correct
    pred_dropped = 0.25*1.8 + 0.25*0.0 + 0.5*mixed_dropped
    println()
    @printf("  predicted E[Ẑ], C retained = 1/4·1.8 + 1/4·0 + 1/2·%.3f = %.6f\n",
            mixed_correct, pred_correct)
    @printf("  predicted E[Ẑ], C dropped  = 1/4·1.8 + 1/4·0 + 1/2·%.3f = %.6f\n",
            mixed_dropped, pred_dropped)
    println("""
  which is what the two blocks above measure, to within their standard
  errors. The single-step figure of 0.875 is diluted by the runs whose
  clouds happen to be homogeneous -- a reminder that the bias is a
  property of unequal weights meeting a state-dependent future, and
  disappears the moment either ingredient is missing.""")
end

## =====================================================================
head("4.  Why a smooth model is nearly as blind as the i.i.d. one")
## =====================================================================
##
## The Gompertz example is the obvious thing to validate against, since it
## is exactly linear-Gaussian on the log scale and so has a closed-form
## Kalman likelihood. On the log scale
##
##     Zₜ = (1-S)·log K + S·Zₜ₋₁ + wₜ,    S = e^{-r},
##
## and at the standard parameters r = 4.5, so S = 0.011. The latent
## process retains one per cent of its previous value: it is 99 per cent
## memoryless, which is to say 99 per cent of the way to the degenerate
## case of section 2.

gauss(x,m,s) = exp(-0.5*((x-m)/s)^2)/(s*sqrt(2π))
let r_=4.5, S=exp(-4.5), σp=0.7, σm=0.1, K=210.0
    a = (1-S)*log(K); sdp = sqrt(σp^2+σm^2)
    zbar = log(150.0); zsd = σp
    @printf("  S = e^{-r} = %.5f\n\n", S)
    println("  two particles one cloud-sd apart, observation δ predictive sds off centre:")
    for δ ∈ (0.0,0.5,1.0,2.0)
        logy = a + S*zbar + δ*sdp
        h1 = gauss(logy, a+S*(zbar+zsd), sdp)
        h2 = gauss(logy, a+S*(zbar-zsd), sdp)
        o = outcomes(h1,h2)
        @printf("    δ = %.1f   (h₂-h₁)/h₁ = %+.5f   relative bias at J=2 = %+.3e\n",
                δ, (h2-h1)/h1, (o.Eℓ-o.tgt)/o.tgt)
    end
    o_fr = outcomes(1.0,0.0)
    logy = a + S*zbar + sdp
    o_go = outcomes(gauss(logy,a+S*(zbar+zsd),sdp), gauss(logy,a+S*(zbar-zsd),sdp))
    println()
    @printf("  frozen state, for comparison:      relative bias at J=2 = %+.3e\n",
            (o_fr.Eℓ-o_fr.tgt)/o_fr.tgt)
    @printf("  ratio of the two: %.0f×\n", abs((o_fr.Eℓ-o_fr.tgt)/o_fr.tgt /
                                               ((o_go.Eℓ-o_go.tgt)/o_go.tgt)))
    @printf("  and since the bias scales like 1/J, Gompertz at J = 8 gives ~%.4f%%\n",
            100*abs((o_go.Eℓ-o_go.tgt)/o_go.tgt)*2/8)
end
println("""
  Note also the first row: at δ = 0, with the observation at the centre of
  the cloud, the two particles are equidistant from the predictive mean,
  h₁ = h₂, and the bias vanishes exactly even here. The effect depends on
  the innovation as well as on the memory.

  A paired comparison on this model over 400,000 replicates resolved about
  0.02% and found nothing. That was a statement about the experiment, not
  about the estimator.""")

head("Summary")
println("""
  bias = -Cov(C, ℓ)  per resampling step, and for a two-particle cloud
  exactly 0.025·(h₂ - h₁).

  It vanishes identically when the one-step predictive density is constant
  across the cloud -- an i.i.d. latent state -- and nearly so when the
  latent process mixes fast. It is visible when the surviving particle
  determines the future.

  C ≡ 1 at β = 0 and β = 1, so the ordinary bootstrap filter and the fully
  weighted filter are both untouched. This is strictly the partially
  retained-weight case.

  A null numerical result is worth nothing until the test model has been
  shown capable of exhibiting the effect.""")
println()
