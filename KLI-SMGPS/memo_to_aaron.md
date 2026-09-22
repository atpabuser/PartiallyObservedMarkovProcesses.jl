To:      Aaron King
From:    Debsurya
Date:    2026-09-08
Subject: A lost normalizing constant in the power-tempered resampling step, and some questions about the trigger/target filter

Aaron,

I have taken your unified trigger/target formulation as the basis for a Julia translation of the particle filter. On the classical path — trigger = 1, target = 0 — I have checked it against the Gompertz model, which is exactly linear-Gaussian on the log scale and so has a closed-form Kalman likelihood: a deterministic reference rather than two compared Monte Carlo estimators, worth having since Gompertz is your own worked example. The particle estimates agree with the exact likelihood, with the discrepancy constant across particle counts. Agreement with R pomp on that path is likewise good.

The weighted path, target > 0, is where I think there is a problem, and it is the reason for this note. I should say plainly that my earlier checks there were not decisive: a likelihood comparison between two implementations that share the same defect will agree perfectly, and a Monte Carlo check on a smooth model has almost no power against the effect in question. Item 2 below is therefore argued from exact calculation instead.

1. On the interpretation of β (your `target`): resampling proceeds with probabilities proportional to w^(1-β), and each surviving particle retains a weight proportional to w^β, renormalized to unit mean — β = 0 recovers standard systematic resampling with equal weights, β = 1 is uninformative resampling that carries the full weighted particle representation forward. Is this a genuine partial tempering of the resampling step, interpolating between the variance-reducing effect of resampling and the bias properties of the weighted filtering distribution — and do you have a preferred β, or an argument for setting it, beyond trial and error?

2. The substance of this note: I believe `systematic_resample!` loses a normalizing constant whenever 0 < β < 1, and that the loss biases the likelihood estimate. I got this wrong in both directions before arriving at it, so I will give the algebra and an exact case rather than an argument from authority.

Ancestors are selected with probability qᵢ = wᵢ^(1-β)/S, where S = Σᵢ wᵢ^(1-β). Selecting from q rather than from the weights themselves changes the sampling measure, so the properly weighted representation assigns the particle selected at position j the importance weight

  R_j = w_{A_j} / (J·q_{A_j}) = (S/J)·w_{A_j}^β,

whose sample mean is

  C = (1/J) Σⱼ R_j = (S/J)·m,    m = mean_j w_{A_j}^β.

C is the resampling step's contribution to the normalizing constant of the unnormalized measure. Both our versions instead store the retained weights renormalized to unit mean, V_j = R_j / C, and neither multiplies C into the likelihood accumulator:

```julia
# your systematic_resample! (as of 1d8dbc0)
w ./= mean(w)
nothing
```

So C is simply dropped. The tempting defence — the one I talked myself into for a day — is that E[C | w] = 1 exactly, which it is: E[m | w] = Σᵢ qᵢwᵢ^β = (Σᵢ wᵢ)/S = J/S. But C is a function of the selected ancestors, hence correlated with everything those ancestors go on to generate, so E[C]=1 does not give E[C·ℓ]=E[ℓ].

The cleanest demonstration I have is exact rather than statistical. Take J = 2, carried weights w = (1.8, 0.2) (unit mean), β = ½. Then q = (0.75, 0.25), and two-particle systematic resampling gives ancestry (1,1) or (1,2), each with probability exactly ½, carrying C = 1.2 and C = 0.8. Let the future contribution from the two states be g = (1, 0). The correctly weighted predictive quantity is (1/J)Σᵢ wᵢgᵢ = 0.9. Dropping C gives

  ½·1 + ½·0.75 = 0.875,

while retaining it gives

  ½·(1.2·1) + ½·(0.8·0.75) = 0.9,

exactly. Here E[C] = 1.000000 while Cov(C, ℓ) = +0.025000, which is the whole of it: the factor is mean-one and not independent, and 0.875 + 0.025 = 0.9. The enumeration is exact — with J = 2 the systematic sweep is driven by a single uniform, so the expectation is a finite sum, not a simulation. The bias peaks near β = ½ and vanishes at both endpoints (β = 0.25 and 0.75 give −0.0181).

I have also built this into an end-to-end filter test — a frozen binary latent state, where the surviving particle determines the entire future — and at J = 2 the dropped-C estimator returns E[Ẑ] = 0.8879 against an exact Z = 0.9, a 27-standard-error discrepancy that vanishes when C is restored. The gap is Θ(1/J) per resampling step, which is why it is invisible at the particle counts one would actually use.

Two caveats on my own account. First, I originally found this, then retracted it on the strength of a Gompertz check that showed no difference; that check had no power, because the bias is Θ(1/J) per step and vanishes identically whenever the one-step predictive density is constant across the cloud — which is exactly what happens when the latent process has no memory. A smooth model with a well-mixing state is close to that degenerate case, and my design could resolve about 0.02% against a bias under 0.15%. Second, C ≡ 1 at β = 0 and β = 1, so nothing here touches the ordinary bootstrap filter; it is strictly the partially retained-weight case.

If this is right, it affects your devel as well as my translation. Restoring C is a two-line change: return log(m·S/J) from `systematic_resample!` and add it to the conditional log likelihood in `pfilt_step_comps!`.

3. On the retained weights themselves: I had this in an earlier draft as a second finding, but you fixed it yourself in `1d8dbc0` before I got here, and your fix and mine agree — the retained weight must follow the selected ancestor rather than staying at the position it lands in. I mention it only so you know I was looking at the pre-`1d8dbc0` code, and that the observation is now yours, not mine.

4. On the terminal draw for the stored ancestral lineage — since resolved, and recorded here only because it came up alongside the above. Whenever β > 0, or `trigger` skips resampling at some step, the terminal cloud is unequally weighted. In the version I was reading, `trace_ancestry!` initiated the lineage with a uniform draw over the terminal particles:

```julia
# your trace_ancestry! (1d8dbc0, line 408)
j = rand(axes(perm,2))     # uniform draw, regardless of terminal weights
```

Your own FIXME at line 97 — "check that the first index in the ancestry is correct" — is, I think, exactly this. My translation initiates the draw proportional to the normalized terminal weights before tracing backward:

```julia
# my weighted trace_ancestry! (lines 523–536, simplified)
s = 0
for k ∈ eachindex(w)
    s += w[k]
    work[k] = s
end
u = s * rand()
j = 1
while (u > work[j] && j < n)
    j += 1
end
# then trace backward from j as before
```

This does not change the likelihood estimate — only the measure represented by the stored trajectory and the reported initial ancestor. Would that resolve your concern?

5. Separately, I have found it useful to record, at each observation time, whether resampling was actually triggered, as a quick diagnostic for weight degeneracy over a filtering run. Would that be worth carrying in your version as well?

6. Your remark that the R formulation of partrans is monolithic prompted a declarative alternative: each parameter is tagged with its transformation — logarithmic, logistic, or none — with log-barycentric groups for parameters on the unit simplex, no requirement that group members sit adjacent, since named tuples carry no positional layout. The inverse is derived from the declaration rather than supplied by the user, so the two cannot drift apart. In Julia this is:

```julia
ParameterTransform(
    (r = :log, K = :log, σₚ = :log, σₘ = :log);
    logbarycentric = (:a, :b, :c)    # simplex-constrained group
)
```

The tags and groups are type parameters on `@generated` callable structs, so the method body is specialized key by key at compile time.

One honest qualification on the simplex groups, which I had stated too strongly until recently. With Tᵢ = log(θᵢ/Σⱼθⱼ) and θᵢ = exp(Tᵢ)/Σⱼexp(Tⱼ), the composition from∘to is the identity on the interior of the simplex, but to∘from is *not* the identity on Euclidean space: softmax is invariant under T ↦ T + c·1, so `from` is not injective and the k coordinates carry only k−1 degrees of freedom. The image of `to` is the manifold {T : Σᵢexp(Tᵢ) = 1}. So this is a bijection onto that manifold, not onto ℝᵏ, and calling the pair mutually inverse is only correct in the one direction. For iterated filtering the redundancy is benign — a Gaussian step leaves the manifold and softmax projects back, giving a legitimate logistic-normal perturbation with the common direction quotiented out — but that direction does random-walk without restraint across iterations, which I have not yet decided whether to constrain. If you have a view on whether the redundant parameterization is worth trading for k−1 additive log-ratio coordinates, I would be glad of it.

Last, since mif2 now runs on the same shared filtering step as the weighted filter, iterated filtering under a weighted (target > 0) filter is available to try — something the R version cannot presently do, since its weighted filter does not resample parameters. That bears directly on question 1: if β is to be chosen deliberately rather than by trial and error, here is a setting where the choice visibly affects estimation, and I would enjoy exploring it jointly if of interest.

Best,
Debsurya
