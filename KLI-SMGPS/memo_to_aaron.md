To:      Aaron King
From:    Debsurya
Date:    2026-09-08
Subject: The unified trigger/target filter — validation, two observations on systematic resampling, and a further question or two

Aaron,

I have taken your unified trigger/target formulation as the basis for a Julia translation of the particle filter, validated two ways. Against the Gompertz model — exactly linear-Gaussian on the log scale, so its likelihood is available in closed form from the Kalman filter, a deterministic reference rather than two compared Monte Carlo estimators, worth flagging since Gompertz is your own worked example — the particle estimates agree with the exact likelihood, the discrepancy constant across particle counts, as it must be. Against R pomp 6.4.0.3: measurement densities agree to floating-point roundoff, the classical filter agrees with pomp within Monte Carlo standard error at every particle count tried, the conditional log likelihood agrees at all 27 observation times of the Parus major data including the first, and a grid over trigger and target agrees cell by cell. That gives me confidence enough to raise what follows.

1. On the interpretation of β (your `target`): resampling proceeds with probabilities proportional to w^(1-β), and each surviving particle retains a weight proportional to w^β, renormalized to unit mean — β = 0 recovers standard systematic resampling with equal weights, β = 1 is uninformative resampling that carries the full weighted particle representation forward. Is this a genuine partial tempering of the resampling step, interpolating between the variance-reducing effect of resampling and the bias properties of the weighted filtering distribution — and do you have a preferred β, or an argument for setting it, beyond trial and error?

2. Two observations on `systematic_resample!`, raised as questions rather than corrections. First, the retained weights: once ancestors are selected from the cumulative sums of w^(1-β), your Julia version raises the whole weight vector to the power β in its original positional order, then renormalizes:

```julia
# your Julia systematic_resample! (lines 360–362)
w .^= β
w ./= mean(w)
nothing
```

Your own R implementation — the C routine behind wpfilter — instead copies the retained weight of the selected ancestor, so the weight travels with the particle:

```c
// wpfilter.c (line 188, inside the sample[k] loop)
wt[k] = ws[sp];    // ws = carried weights, sp = selected ancestor index
```

My translation follows the R form:

```julia
# my systematic_resample! (lines 472–474)
@inbounds for j ∈ eachindex(p)
    ucum[j] = w[p[j]]^β     # retained weight of the *selected* ancestor
end
```

Your two versions differ from each other here, and my translation follows the R form. Since each weight factors as w = w^(1-β)·w^β, and w^(1-β) is already expressed through the multiplicity of selection, the retained w^β factor must accompany the selected particle for the weighted representation to target the right measure. Was the positional form intentional, or an oversight in translation?

3. Second, a question about the mass removed at resampling that I raise mainly because I got it wrong myself, and the resolution may be worth recording. Renormalizing the retained weights to unit mean rescales them by a common factor. Writing S′ for the sum of the w^(1-β) and m for the mean of the retained weights before renormalization, the properly weighted representation assigns each selected particle the weight w^β·S′/J, so the unit-mean convention discards the factor m·S′/J at every resampling step. I took this to be an omission and credited the factor back to the running likelihood, on the grounds that the credited form satisfies the proper-weighting identity

  E[ Ẑ · (1/J) Σⱼ Wⱼ φ(xʲ) ] = γₙ(φ)

exactly, while the uncredited form does not. That inference was wrong. The uncredited form satisfies a different identity — the telescoping one, in which the normalization applied at step n cancels against the increment at step n+1 — and is equally unbiased. I checked this numerically on the linear-Gaussian reduction of Gompertz, where the Kalman likelihood gives exact truth, computing both estimators from the same draws so the comparison is exactly paired: over 400,000 replicates at each of β ∈ {0.25, 0.5, 0.75} and J ∈ {8, 16, 64}, the two expectations agree to within 0.003% of L̂, with no significant difference anywhere. Since m·S′/J is a conditionally mean-one factor uncorrelated with the estimate, crediting it adds variance and nothing else, so I have removed it again — my translation now follows your convention exactly.

The reason I mention it at all: your R and Julia versions agree here, and it took the numerical check to convince me they were right and I was not. If the tempering is ever used inside a particle MCMC, where exact unbiasedness of L̂ is what makes the sampler valid, it may be worth having the argument written down somewhere.

4. On the terminal draw for the stored ancestral lineage: whenever β > 0, or `trigger` skips resampling at some step, the terminal cloud is unequally weighted. As I read `trace_ancestry!`, the lineage is initiated by a uniform draw over the terminal particles:

```julia
# your trace_ancestry! (line 394)
j = rand(axes(perm,2))     # uniform draw, regardless of terminal weights
```

I noticed your marginal note there, questioning whether the first index in the ancestry is right — I suspect this is exactly the issue. My translation initiates the draw proportional to the normalized terminal weights before tracing backward:

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

This was exercised in the validation above; would that resolve your concern?

5. Separately, I have found it useful to record, at each observation time, whether resampling was actually triggered, as a quick diagnostic for weight degeneracy over a filtering run. Would that be worth carrying in your version as well?

6. Your remark that the R formulation of partrans is monolithic prompted a declarative alternative: each parameter is tagged with its transformation — logarithmic, logistic, or none — with log-barycentric groups for parameters on the unit simplex, no requirement that group members sit adjacent, since named tuples carry no positional layout. The inverse transformation is derived automatically, so mutual inversion holds by construction rather than resting on the user. In Julia this is:

```julia
ParameterTransform(
    (r = :log, K = :log, σₚ = :log, σₘ = :log);
    logbarycentric = (:a, :b, :c)    # simplex-constrained group
)
```

The tags and groups are type parameters on `@generated` callable structs, so the method body is specialized key by key at compile time — no runtime dispatch, and `@inferred` passes. This now drives the iterated filtering procedure directly, and I would value your reaction.

Last, since mif2 now runs on the same shared filtering step as the weighted filter, iterated filtering under a weighted (target > 0) filter is available to try — something the R version cannot presently do, since its weighted filter does not resample parameters. That bears directly on question 1: if β is to be chosen deliberately rather than by trial and error, here is a setting where the choice visibly affects estimation, and I would enjoy exploring it jointly if of interest.

Best,
Debsurya
