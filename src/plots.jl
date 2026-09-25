## Plotting functions, implemented in the AlgebraOfGraphics extension
## (ext/PartiallyObservedMarkovProcessesAoGExt.jl): load
## `AlgebraOfGraphics` and a Makie backend (e.g., `CairoMakie`) to use them.

"""
    sliceplot(df; kwargs...)

Plots the output of [`slice`](@ref), one panel per sliced parameter,
with error bars from the `se` column.  Returns the figure drawn by
AlgebraOfGraphics; additional arguments are passed to `draw`.
Requires `AlgebraOfGraphics` and a Makie backend.
"""
function sliceplot end

"""
    mcapplot(m; kwargs...)

Plots an [`mcap`](@ref) result: the profile, the smoothed and quadratic
fits, the estimate, the confidence interval, and the cutoff.  Returns
the figure drawn by AlgebraOfGraphics; additional arguments are passed
to `draw`.  Requires `AlgebraOfGraphics` and a Makie backend.
"""
function mcapplot end

"""
    traceplot(mf; pars, monitor, kwargs...)

Plots the traces of one or a vector of [`mif`](@ref) computations: the
log likelihood and the parameters `pars` (by default, those perturbed)
against the iteration, with the output of [`monitor`](@ref) as a
further panel if given.  Returns the figure drawn by AlgebraOfGraphics;
additional arguments are passed to `draw`.  Requires
`AlgebraOfGraphics` and a Makie backend.
"""
function traceplot end
