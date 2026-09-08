# Cross-validation against R pomp 6.4.0.3

This note summarizes the validation of PartiallyObservedMarkovProcesses.jl's
particle-filtering machinery against two independent references: the
exact Kalman-filter likelihood available on a linear-Gaussian reduction
of the Gompertz model, and R pomp 6.4.0.3's own filtering output on the
same model and data. The checks proceed from the most elementary
building block (agreement of measurement densities) up through the
full trigger/target-tempered filter, and close with a list of concrete
discrepancies found between Aaron King's Julia translation and his own
R implementation — discrepancies present only in the Julia code, not
in the R code they were meant to reproduce.

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
was run against R pomp's `wpfilter`, and the two agree cell by cell —
validating both the power-tempered resampling weight and the
ESS-triggered resampling decision, separately and in combination,
rather than only at the trigger=1/target=0 corner already covered in
§2.

## 5. What was found

Comparing Aaron King's own, earlier Julia translation of the filter
against his R implementation (the C routine behind `wpfilter`) turned
up two genuine discrepancies, listed first, and one apparent
discrepancy that on closer examination was not one at all — recorded
here as well, since the reasoning that dismissed it is the more
instructive of the two outcomes.

- **Retained weight after resampling is assigned in the wrong order.**
  Aaron's Julia `systematic_resample!` selects ancestors from the
  cumulative sums of w^(1-β), then raises the *entire weight vector*
  to the power β **in its original positional order** and renormalizes
  — so the retained weight credited to output slot k is w_k^β, the
  weight of whichever particle happened to occupy position k
  *before* resampling, not the weight of the particle actually
  selected into slot k. Aaron's R implementation (`wpfilter.c`, line
  188, `wt[k] = ws[sp]`, where `sp` is the sampled ancestor index) does
  this correctly: it copies the *selected ancestor's* retained weight
  into the new slot, so the weight travels with the particle it
  belongs to. Since each original weight factors as
  w = w^(1-β)·w^β, with the w^(1-β) factor already expressed through
  the multiplicity with which a particle is selected, the accompanying
  w^β factor must travel with that same particle for the carried
  weighted representation to target the correct measure. The
  positional form is a bug; the R form is the intended design.

- **The terminal ancestry draw is uniform rather than weighted.**
  Aaron's `trace_ancestry!` initiates the stored ancestral lineage with
  a uniform draw over the terminal particle indices, regardless of
  their carried weights. Whenever the terminal cloud is unequally
  weighted — which follows whenever β > 0, or simply from a
  `trigger < 1` step that skipped resampling — this
  targets the wrong distribution (see the accompanying proof in
  `ancestry_proof.md`). No R-side implementation issue applies here
  directly, since R pomp's `wpfilter` does not itself perform lineage
  tracing to reconstruct a stored trajectory; the correct behavior is
  simply what the weighted representation already requires: a draw
  proportional to the terminal weights before tracing backward.

### An apparent third issue that was not one

The unit-mean convention used throughout the Julia routine rescales the
retained weights by a common factor at every resampling step. Writing
S' = Σ w^(1-β) and m for the mean of the retained weights before
renormalization, the properly weighted representation assigns each
selected particle the weight w^β·S'/J, so the unit-mean convention
discards the factor m·S'/J — a factor that vanishes at β = 0 and β = 1
but not in between. This was initially recorded as a third defect, on
the grounds that the credited form satisfies the proper-weighting
identity

  E[ Ẑ · (1/J) Σⱼ Wⱼ φ(xʲ) ] = γₙ(φ)

exactly, whereas the uncredited form does not, and the credit was
added to the running log likelihood.

That reasoning was mistaken. Failure of *that* identity does not imply
bias: the uncredited form satisfies a different one — the telescoping
identity, under which the normalization applied at step n cancels
against the increment at step n+1 — and is equally unbiased. The two
are alternative proofs of unbiasedness, not a correct and an incorrect
scheme.

The point was settled numerically on the linear-Gaussian reduction of
Gompertz, where the Kalman filter supplies exact truth. Both
estimators were computed from the same draws — the particle cloud is
identical under the two conventions, which differ only in a scalar
accumulator — so the comparison is exactly paired and its standard
error is far smaller than that of either mean. Over 400,000 replicates
at each of β ∈ {0.25, 0.5, 0.75} and J ∈ {8, 16, 64}, the two
expectations agree to within 0.003% of L̂, with no significant
difference at any setting. Since m·S'/J is a conditionally mean-one
factor uncorrelated with the estimate, crediting it adds variance and
nothing else. The credit was accordingly removed, and the present
translation follows the same bookkeeping as both of Aaron's versions.

### Summary

Both genuine issues are present in Aaron's Julia translation and not in
his R code — the R code implements the retained-weight assignment
correctly, and the terminal-draw issue is specific to a lineage-tracing
feature the R implementation does not provide. The present translation
adopts the R form in both cases: copying the selected ancestor's
retained weight, and drawing the terminal ancestry index proportional
to the carried weights.

Neither would have been caught by the marginal-log-likelihood
comparisons of §§2–4. Both affect the identity and weighting of
individual retained particles without disturbing the average log
likelihood, so those checks would have passed either way. The
retained-weight defect was found by direct code inspection against the
R source; the terminal-draw defect by Aaron's own marginal note in the
code questioning whether the first index in the ancestry was correct.
This is worth stating plainly: the likelihood-level agreement
documented in §§1–4 is necessary but not sufficient, and the two
defects that mattered were found by reading, not by testing.
