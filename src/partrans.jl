## ------------------------------------------------------------ partrans

## `PartransTo{tags,groups}`/`PartransFrom{tags,groups}` are callable
## structs, not closures: `tags` (a `NamedTuple` of per-parameter codes)
## and `groups` (a `Tuple` of `Tuple`s of `Symbol`s, the log-barycentric
## groups) are carried as type parameters, so the generated method body
## below is specialized, key by key, at the time `to`/`from` is first
## applied to a concretely typed parameter `NamedTuple`.
struct PartransTo{tags,groups} <: Function end
struct PartransFrom{tags,groups} <: Function end

@generated (::PartransTo{tags,groups})(θ::NamedTuple{ks}) where {tags,groups,ks} = begin
    exprs = Expr[]
    for k ∈ ks
        if haskey(tags,k)
            tag = getfield(tags,k)
            push!(exprs,
                tag === :log ? :(log(θ.$k)) :
                tag === :logit ? :(log(θ.$k/(1-θ.$k))) :
                :(θ.$k)
            )
        else
            found = false
            for grp ∈ groups
                if k ∈ grp
                    s = Expr(:call,:+,(:(θ.$m) for m ∈ grp)...)
                    push!(exprs,:(log(θ.$k)-log($s)))
                    found = true
                    break
                end
            end
            found || push!(exprs,:(θ.$k))
        end
    end
    :(NamedTuple{$ks}(($(exprs...),)))
end

@generated (::PartransFrom{tags,groups})(θ::NamedTuple{ks}) where {tags,groups,ks} = begin
    exprs = Expr[]
    for k ∈ ks
        if haskey(tags,k)
            tag = getfield(tags,k)
            push!(exprs,
                tag === :log ? :(exp(θ.$k)) :
                tag === :logit ? :(1/(1+exp(-θ.$k))) :
                :(θ.$k)
            )
        else
            found = false
            for grp ∈ groups
                if k ∈ grp
                    s = Expr(:call,:+,(:(exp(θ.$m)) for m ∈ grp)...)
                    push!(exprs,:(exp(θ.$k)/$s))
                    found = true
                    break
                end
            end
            found || push!(exprs,:(θ.$k))
        end
    end
    :(NamedTuple{$ks}(($(exprs...),)))
end

_logbary_groups(g::Tuple{}) = ()
_logbary_groups(g::Tuple{Vararg{Symbol}}) = (g,)
_logbary_groups(g::Tuple{Vararg{Tuple}}) = g
_logbary_groups(g) = error(
    "`logbarycentric` must be a `Tuple` of `Symbol`s (one group) or a "*
    "`Tuple` of `Tuple`s of `Symbol`s (several disjoint groups)."
)

"""
    ParameterTransform(tags::NamedTuple; logbarycentric = ())
    ParameterTransform(to::Function, from::Function)

A declarative parameter transformation for [`mif2`](@ref), mapping a
natural-scale parameter `NamedTuple` to and from an unconstrained
estimation scale.

In the first form, `tags` names, for each parameter to be transformed,
one of `:log` (``T=\\log\\theta``), `:logit`
(``T=\\log(\\theta/(1-\\theta))``), or `:identity`. Parameters not named
in `tags` pass through unchanged. `logbarycentric` names one group (a
`Tuple` of `Symbol`s) or several disjoint groups (a `Tuple` of `Tuple`s
of `Symbol`s) of parameters to be jointly transformed by the log
barycentric transformation,
``T_i=\\log(\\theta_i/\\textstyle\\sum_{j\\in g}\\theta_j)``, paired with
``\\theta_i=e^{T_i}/\\sum_{j\\in g}e^{T_j}``, so the natural-scale values
recovered by the second map always sum to one. Group members must not
also be named in `tags`.

The pair is mutually inverse in one direction only. Composing `to` then
`from` is the identity on the interior of the simplex, but the reverse
composition is not the identity on ``\\mathbb{R}^k``: the second map is
invariant under ``T\\mapsto T+c\\mathbf{1}``, so the ``k`` coordinates
carry only ``k-1`` degrees of freedom and `from` is not injective. The
image of `to` is the manifold ``\\{T:\\sum_i e^{T_i}=1\\}``, and `to` is a
bijection of the simplex interior onto *that*, not onto Euclidean space.

For iterated filtering the redundancy is harmless — a Gaussian step
leaves the manifold and the second map projects back onto the simplex,
giving a logistic-normal perturbation in which the common direction is
quotiented out — but that direction is unconstrained by the data and
random-walks freely across iterations.

In the second form, `to` and `from` are supplied directly as a mutually
inverse pair of `NamedTuple`-to-`NamedTuple` functions.
"""
struct ParameterTransform
    to::Function
    from::Function
end

ParameterTransform(tags::NamedTuple; logbarycentric::Tuple = ()) = begin
    for k ∈ keys(tags)
        v = getfield(tags,k)
        v ∈ (:log,:logit,:identity) ||
            error("`ParameterTransform` tags must be `:log`, `:logit`, or `:identity`; "*
                  "got `$v` for parameter `$k`.")
    end
    groups = _logbary_groups(logbarycentric)
    seen = Symbol[]
    for g ∈ groups
        for s ∈ g
            s ∈ keys(tags) &&
                error("log-barycentric group member `$s` must not also be named in `tags`.")
            s ∈ seen &&
                error("log-barycentric groups must be pairwise disjoint; "*
                      "`$s` appears in more than one group.")
            push!(seen,s)
        end
    end
    ParameterTransform(PartransTo{tags,groups}(),PartransFrom{tags,groups}())
end

## Recovers the declared tag/group specification from a `ParameterTransform`
## built from the declarative constructor, for use in `mif2`'s validation
## of parameter names; not defined (and so errors) for an explicit `to`
## wrapping an arbitrary function pair.
partrans_spec(pt::ParameterTransform) = partrans_spec(typeof(pt.to))
partrans_spec(::Type{<:PartransTo{tags,groups}}) where {tags,groups} = (tags,groups)
