"""
   val_array(x)

Stop makin' a fool outta me!
Why don'cha come on over...?
"""
val_array(x::Vector) = x

## A one-dimensional view is already a vector of values and must be
## passed through. It matches neither of the two methods above -- a
## `SubArray` is not an `Array`, and `Vector` is `Array{T,1}` -- so
## without this it fell through to the scalar fallback below and was
## wrapped as a single value, which silently turned a view of J
## parameter sets into one parameter set. `Vector` remains the most
## specific method, so nothing else changes.
val_array(x::AbstractVector) = x

val_array(x::Array{X,N}) where {X,N} = vec(x)

val_array(x::Array{X,N}, dim::Integer...) where {X,N} = begin
    q,r = divrem(length(x),prod(dim))
    if r != 0
        error("in `val_array`: size mismatch.")
    else
        reshape(x,dim...,q)
    end
end

val_array(x) = [x]

val_array(x, dim::Integer...) = val_array([x],dim...)
