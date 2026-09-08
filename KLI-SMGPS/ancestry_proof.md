# The terminal draw for a stored ancestral lineage must be weighted, not uniform

## Setting

A sequential Monte Carlo filter propagates a cloud of particles
{x_1^j, ..., x_N^j}, j = 1,...,J, through the observation times
1,...,N, together with a lineage-tracing array that, at each step,
records which particle at step i-1 was the parent of particle j at
step i. When the filter resamples with equal post-resampling weights
at every step, the terminal cloud at time N is itself equally
weighted, and any one of the J terminal particles is as representative
of the filtering distribution as any other.

This is not the general case. Under a partially tempered or
ESS-triggered filter — resampling with probability proportional to
w^(1-β) and carrying forward a residual weight w^β (β > 0), or
skipping resampling altogether at some step because the effective
sample size already exceeds a trigger threshold (trigger < 1) — the
terminal particles carry unequal weights w_N^1, ..., w_N^J. The
filtering distribution at time N is then represented not by the
particle positions alone but by the weighted empirical measure

  π_N(dx) ≈ Σ_j (w_N^j / Σ_k w_N^k) δ_{x_N^j}(dx).

To extract a single ancestral trajectory from the filter's output —
for diagnostic plotting, for conditional simulation, or as a
representative draw from the smoothing distribution — one must choose
a terminal particle and then trace its lineage backward through the
recorded parentage. The question this note settles is: with what
probability should each terminal index j be chosen?

## Claim

The terminal index must be drawn with probability proportional to the
carried weight w_N^j, i.e.

  P(select j) = w_N^j / Σ_k w_N^k.

A uniform draw over the J terminal indices, P(select j) = 1/J for
every j, is correct only in the special case that all terminal weights
are equal, and is otherwise a biased draw from the wrong measure.

## Proof by counterexample

Take J = 2 terminal particles with carried weights W = (0.9, 0.1).
(Such weights arise, for instance, as the w^β factors retained under
partial tempering, or simply as the raw importance weights of two
particles that survived an ESS-triggered step without resampling.)

The filtering distribution places mass

  0.9 / (0.9 + 0.1) = 0.9

on particle 1, and mass 0.1 on particle 2. This is what "the filtering
distribution at time N" means by construction: it is the weighted
empirical measure carried by the particle cloud.

A **weighted draw** — select particle 1 with probability 0.9 and
particle 2 with probability 0.1 — reproduces this measure exactly. It
is correct.

A **uniform draw** — select each particle with probability 0.5 —
does not reproduce it. Particle 2, which should be selected only 1
time in 10, is instead selected 1 time in 2: a five-fold
overweighting. Symmetrically, particle 1 is selected only 5/9 as often
as it should be. The uniform draw is not a small perturbation of the
correct answer; it targets a different distribution entirely — the
unweighted empirical distribution of particle positions, which
coincides with the filtering distribution only when the weights
happen to be equal.

## General statement

For J terminal particles with weights w_N^1,...,w_N^J (not
identically equal), the probability that the stored trajectory
descends from terminal particle j must be

  w_N^j / Σ_k w_N^k.

Uniform selection assigns probability 1/J to every j. The two
coincide, for every j, if and only if all the w_N^j are equal. In
every other case uniform selection is a strictly biased estimator of
the filtering distribution: it systematically under-represents
particles with above-average weight and over-represents particles
with below-average weight, exactly as in the two-particle example
above, and the resulting stored trajectory is a draw from the
unweighted empirical distribution of particle positions rather than
from the filtering distribution the particle cloud was constructed to
represent. Any downstream quantity computed from such a trajectory —
smoothed-state summaries, diagnostic overlays against the data, or a
seed trajectory for further conditional simulation — inherits this
bias.

This is a special case of the elementary fact that a self-normalized
importance sample from a weighted particle cloud recovers the target
distribution only by weighted resampling; discarding the weights at
the point of the final draw discards exactly the information that
distinguishes the filtering distribution from the proposal (here, the
raw particle positions).

## Connection to the code

`src/pfilter.jl` carries two implementations of `trace_ancestry!`,
differing precisely on this point.

The three-argument form (lines 483–496) draws the terminal index
uniformly:

```julia
j::I = rand(axes(perm,2))
```

with no reference to any terminal weight. This is correct only when
the terminal cloud is known to be equally weighted — the case
following a step that actually resampled to equal weights, with no
subsequent unresampled or partially-tempered step.

The five-argument form (lines 508–544) is the weighted variant. It
builds the cumulative sum of the linear weights `w` into `work`, draws
a single uniform variate u on (0, Σw], and locates the terminal index
j by inverse-CDF search over the cumulative array:

```julia
u::W = s*rand(LogLik)
jj::I = 1
while (u > work[jj] && jj < n)
    jj += 1
end
```

— i.e., j is selected with probability exactly w_N^j / Σ_k w_N^k, as
required. (When every weight is -Inf on the log scale, `wmax` is not
finite and the routine falls back to a uniform draw, since in that
degenerate case — total filtering failure — no weight carries useful
information and every terminal particle is equally uninformative.)

Both routines then trace the lineage backward from the selected j in
the same way, walking `perm` from time N down to time 1. The two
implementations differ only in how the starting index j at time N is
chosen — and it is exactly this choice that must respect the terminal
weights whenever those weights are not uniform, i.e. whenever the
filter has run with β > 0 or with trigger < 1 and resampling was
skipped on the last step.
