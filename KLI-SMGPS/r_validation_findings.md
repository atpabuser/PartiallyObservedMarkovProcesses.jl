# Cross-validation against R pomp 6.4.0.3

This note summarizes the validation of PartiallyObservedMarkovProcesses.jl's
particle-filtering machinery against two independent references: the
exact Kalman-filter likelihood available on a linear-Gaussian reduction
of the Gompertz model, and R pomp 6.4.0.3's own filtering output on the
same model and data. The checks proceed from the most elementary
building block (agreement of measurement densities) up through the
full trigger/target-tempered filter, and close with the concrete issues
found in the course of it.

A warning about §§1–4 before they are read. They establish agreement,
and agreement is not correctness: the most consequential issue found
(§5) is one that both implementations share, so every comparison
between them agrees perfectly while both are biased. The one reference
in this note that is independent of any implementation is the Kalman
likelihood in §3, and the model it is evaluated on turns out to have
almost no power against the effect in question.

Throughout, the working example is the Gompertz population model
together with the *Parus major* (Great Tit) census data of Wytham
Wood, Oxfordshire — pomp's own standard worked example — comprising 27
annual observations.

## 1. Measurement density agreement

The most basic possible check: if the two implementations disagree on
the density of an observation given a state, nothing downstream — no
particle weight, no likelihood, no resampling decision — can be
trusted to agree either. The Julia `logdmeasure` and the R `dmeasure`
were evaluated at matching states, parameters, and observations on the
Gompertz model with the Parus major data, and agree to floating-point
roundoff. This establishes the common ground on which every
subsequent comparison rests.

## 2. Bootstrap filter agreement (trigger = 1, target = 0)

With `trigger = 1` and `target = 0` — resampling to equal weights at
every observation time, no residual tempered weight carried forward —
the filter reduces to the classical bootstrap particle filter of
Gordon, Salmond, and Smith. At this setting:

- The overall log-likelihood estimates from PartiallyObservedMarkovProcesses.jl
  agree with R pomp's bootstrap filter within Monte Carlo standard
  error, checked at Np = 100, 500, and 1000 particles.
- The conditional log-likelihoods — the per-observation-time
  increments that sum to the overall log likelihood — agree at all 27
  observation times of the Parus major series, including the first
  (where the filtering distribution is a point mass, since `rinit`
  places no noise on the initial state and the first observation time
  coincides with t0).

This validates the core filtering recursion — propagate, weight,
resample — independent of any of the tempering or triggering
machinery layered on top of it.

## 3. Kalman ground truth

The Gompertz model, on the log scale, is exactly a linear-Gaussian
autoregression: writing Z_t = log(X_t),

  Z_t = (1 - S) log(K) + S Z_{t-1} + w_t,   S = e^{-r},  w_t ~ N(0, σ_p²),

with the LogNormal measurement model becoming, on the log scale, a
Gaussian observation of Z_t with variance σ_m². This is the one case
in which the filtering log-likelihood is available in closed form,
with no Monte Carlo error, via the scalar Kalman filter — a genuinely
deterministic reference, not a second stochastic estimator being
checked against a first.

Comparing the particle filter's estimate of loglik(pop_1,...,pop_27)
against the exact Kalman value (after the Jacobian correction for the
log-to-population change of variables, since `logdmeasure` evaluates
the density of the population count itself, not of its logarithm)
gives two complementary checks:

- **No systematic drift.** The gap between the Monte Carlo estimate
  and the exact Kalman likelihood is constant across Np = 200, 1000,
  5000 — its spread across particle counts falls well within the
  Monte Carlo standard error at the smallest, noisiest Np. A genuine
  bookkeeping defect in the filter would instead show up as a gap that
  shrinks systematically with Np (the familiar O(1/Np) downward bias
  of a log of an averaged likelihood estimator), which is not what is
  observed.
- **No fixed offset either.** The gap is also close, in absolute
  terms, to the Jacobian constant derived from the log-to-population
  change of variables, at every Np tested — ruling out a constant
  bookkeeping error (e.g. a missing or duplicated Jacobian term)
  masquerading as agreement-up-to-drift.

Together these two checks — constancy across Np, and correctness of
the constant itself — rule out both the two most common classes of
silent filtering-implementation error.

## 4. Trigger/target grid

The unified filter generalizes the bootstrap filter along two axes:
`trigger` (resample only when the effective sample size falls below
`trigger`×Np, rather than at every step) and `target` (denoted β in
the accompanying correspondence: resample with probability
proportional to w^(1-β) and carry forward a residual weight w^β,
renormalized to unit mean, rather than resampling to exactly equal
weights). A grid over trigger ∈ {0.5, 1.0} × target ∈ {0.0, 0.3, 0.5}
was run against R pomp's `wpfilter`, and the two agree cell by cell.

This was originally read as validating the power-tempered resampling
weight and the ESS-triggered resampling decision beyond the
trigger=1/target=0 corner of §2. It does not. R's `wpfilter` keeps its
carried weights unnormalized and forms each conditional log likelihood
as a ratio of successive total masses, which is algebraically the same
estimator as the unit-mean convention with the normalizing constant
discarded. Both sides of this comparison therefore drop the same factor,
and the agreement is evidence that the two implement the same algorithm,
not that the algorithm is unbiased. See §5.

## 5. What was found

Three issues, in descending order of consequence. The first disturbs the
likelihood and is present in every implementation examined, including
this one until it was corrected. The second was a real defect in the
earlier upstream lineage and has since been fixed by its author. The
third remains a genuine difference.

### The principal finding: a discarded normalizing constant

Ancestors are selected with probability qᵢ = wᵢ^(1-β)/S, where
S = Σᵢ wᵢ^(1-β) and the carried weights have unit mean. Selecting from
q rather than from the weights themselves changes the sampling measure,
so the properly weighted representation assigns the particle selected
at position j the importance weight

  R_j = w_{A_j}/(J·q_{A_j}) = (S/J)·w_{A_j}^β,

whose sample mean is

  C = (1/J) Σⱼ R_j = (S/J)·m,    m = mean_j w_{A_j}^β.

C is the resampling step's contribution to the normalizing constant of
the unnormalized measure. Renormalizing the retained weights to unit
mean stores V_j = R_j/C, so C must multiply the likelihood accumulator
or it is lost. C ≡ 1 at β = 0 and β = 1, so the ordinary bootstrap
filter is unaffected; this is strictly the partially retained-weight
case.

Both Aaron's Julia versions — before and after `1d8dbc0` — drop C, and
so did this translation for a day (see the history note below). The
defence that suggests itself is that E[C | w] = 1 exactly, which it is:
E[m | w] = Σᵢ qᵢwᵢ^β = (Σᵢ wᵢ)/S = J/S. But C is a function of the
selected ancestors and therefore correlated with everything those
ancestors go on to generate, so E[C] = 1 does not give E[C·ℓ] = E[ℓ].

The demonstration is exact rather than statistical. Take J = 2,
w = (1.8, 0.2), β = ½. Then q = (0.75, 0.25), and two-particle
systematic resampling gives ancestry (1,1) or (1,2) with probability
exactly ½ each, carrying C = 1.2 and C = 0.8. With future contributions
g = (1, 0) the correctly weighted predictive quantity is
(1/J)Σᵢ wᵢgᵢ = 0.9. Dropping C gives ½·1 + ½·0.75 = 0.875; retaining it
gives ½·(1.2·1) + ½·(0.8·0.75) = 0.9, exactly. The bias is Θ(1/J) per
resampling step.

`test/iid.jl` carries this end to end. On a frozen binary latent state —
X ~ Bernoulli(½) with Xₜ = X for all t, so the surviving particle
determines the entire future — the exact likelihood is Z = 0.9, and at
Np = 2 the dropped-C estimator returns E[Ẑ] = 0.8879, a
27-standard-error discrepancy that vanishes when C is restored.

### History: why this was retracted, wrongly

The finding above was made, then retracted, then reinstated. The
retraction rested on a paired comparison of the two estimators on the
linear-Gaussian reduction of Gompertz over 400,000 replicates at each of
β ∈ {0.25, 0.5, 0.75} and J ∈ {8, 16, 64}, which found the two
expectations agreeing to within 0.003% of L̂ with no significant
difference anywhere.

That check had no power. The bias is Θ(1/J) per step, and it vanishes
identically whenever the one-step predictive density is constant across
the cloud — which is exactly what happens when the latent process has
no memory, and nearly what happens when it mixes well. The design could
resolve about 0.02% of L̂ against a bias under 0.15%. When the same
experiment was rerun with the weight disparity increased, the sign
returned as the theory requires.

The first testset in `test/iid.jl` records the degenerate case
deliberately: an i.i.d. latent process is *blind* to this issue at any
number of replicates, which is why the frozen-state testset exists
beside it.

Two lessons, both general:

- A null Monte Carlo result is worth nothing unless the test model can
  exhibit the effect and the design has power against it. State the
  standard error next to every null.
- A conditionally mean-one factor is harmless only if it is also
  independent of everything downstream.

### The retained weight, and what remains a difference

Assigning the retained weight in positional order rather than by
selected ancestor was a real defect in the pre-`1d8dbc0` upstream
lineage, and the mathematical reason stands: each weight factors as
w = w^(1-β)·w^β, the w^(1-β) factor is already expressed through the
multiplicity of selection, so the retained w^β factor must accompany
the particle it came from. Aaron fixed this himself in `1d8dbc0` on
2026-09-08, independently, and his fix agrees with this one. It is
therefore no longer a difference between the two implementations, and
is recorded here only because the comparison in §§1–4 was made against
the earlier code.

What does remain a difference is the terminal ancestry draw. At
`1d8dbc0` line 408 `trace_ancestry!` still initiates the lineage with a
uniform draw over terminal particles, with Aaron's own FIXME at line 97
questioning whether the first ancestry index is correct. Whenever
β > 0, or a `trigger < 1` step skips resampling, the terminal cloud is
unequally weighted and the uniform draw targets the wrong measure (see
`ancestry_proof.md`). This affects only the measure represented by the
stored trajectory and the reported initial ancestor, not the likelihood.

### What the likelihood comparisons could and could not catch

Of the three issues, only the discarded normalizing constant disturbs
the marginal likelihood — and the comparisons in §§2–4 still failed to
catch it, for two separate reasons. Comparing two implementations that
share the same defect produces perfect agreement, which is what §4's
trigger/target grid did. And the Kalman reference in §3 is exact but was
exercised on a model with almost no power against the effect.

The retained-weight and terminal-draw issues affect the identity and
weighting of individual particles without disturbing the average log
likelihood, so no likelihood comparison of any power would have found
them; both were found by reading the code, one of them by its own
author. The agreement documented in §§1–4 is necessary but a long way
from sufficient.
