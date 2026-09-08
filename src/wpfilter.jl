"""
    wpfilter(object; Np = 1, trigger = 1, target = 0, params, rinit,
             rprocess, logdmeasure, kwargs...)

`wpfilter` runs the weighted particle filter: systematic resampling
occurs at an observation time only when the effective sample size there
falls to `trigger*Np` or below, and upon resampling the weights are
renormalized to the power `target`. It is equivalent to calling
[`pfilter`](@ref) with the same arguments.

At least the `rinit`, `rprocess`, and `logdmeasure` basic components are
needed. `kwargs...` can be used to modify or unset additional fields.
"""
wpfilter(
    object::ValidPompData;
    trigger::Real = 1,
    target::Real = 0,
    kwargs...,
) = pfilter(object; trigger, target, kwargs...)

"""
    wpfilter(object::PfilterdPompObject; Np = object.Np,
             trigger = object.trigger, target = object.target, kwargs...)

Running `wpfilter` on a `PfilterdPompObject` re-runs the weighted
particle filter, as [`pfilter`](@ref) does.
"""
wpfilter(
    object::PfilterdPompObject;
    Np::Integer = object.Np,
    trigger::Real = object.trigger,
    target::Real = object.target,
    kwargs...,
) = pfilter(object; Np, trigger, target, kwargs...)

wpfilter(_...) = error("Incorrect call to `wpfilter`.")
