## Paired test.  The credited and uncredited estimators are computed from
## the SAME draws, so their difference is a paired quantity whose
## standard error is far smaller than that of either mean.  If the two
## conventions really give different expectations, mean(L̂c - L̂u) is
## significantly nonzero.

using Random, Statistics, Printf

const r  = 4.5
const K  = 210.0
const X0 = 150.0
const S  = exp(-r)
const A  = (1-S)*log(K)

kalman(logy,σp,σm) = begin
    m = log(X0); v = 0.0; ll = 0.0
    for t ∈ eachindex(logy)
        if t > 1
            m = A + S*m; v = S^2*v + σp^2
        end
        f = v + σm^2; innov = logy[t] - m
        ll += -0.5*(log(2π*f) + innov^2/f)
        g = v/f; m += g*innov; v -= g^2*f
    end
    ll
end

pf(logy, J, β, σp, σm, rng) = begin
    z = fill(log(X0), J)
    w = ones(J)
    llu = 0.0; credit = 0.0
    wr = zeros(J); cum = zeros(J); p = zeros(Int,J)
    for t ∈ eachindex(logy)
        if t > 1
            @inbounds for j ∈ 1:J
                z[j] = A + S*z[j] + σp*randn(rng)
            end
        end
        @inbounds for j ∈ 1:J
            w[j] *= exp(-0.5*((logy[t]-z[j])^2/σm^2 + log(2π*σm^2)))
        end
        inc = mean(w)
        llu += log(inc)
        w ./= inc
        α = 1-β
        s = 0.0
        @inbounds for j ∈ 1:J
            s += w[j]^α; cum[j] = s
        end
        du = s/J; u = -du*rand(rng); i = 1
        @inbounds for j ∈ 1:J
            u += du
            while (u > cum[i] && i < J); i += 1; end
            p[j] = i
        end
        @inbounds for j ∈ 1:J
            wr[j] = w[p[j]]^β
        end
        z .= z[p]
        m = mean(wr)
        w .= wr ./ m
        credit += log(m*s/J)
    end
    (llu, credit)
end

Random.seed!(1234)
N = 5
σp = 0.7; σm = 0.1
logy = let z = log(X0), acc = Float64[]
    for t ∈ 1:N
        if t > 1; z = A + S*z + σp*randn(); end
        push!(acc, z + σm*randn())
    end
    acc
end
exact = kalman(logy,σp,σm)
@printf("exact loglik = %.6f   (N=%d, σp=%.2f, σm=%.2f)\n\n", exact, N, σp, σm)

@printf("%4s %5s | %-22s | %-24s | %s\n",
        "J","β","total credit (log)","paired  E[L̂c] - E[L̂u]","significant?")
println("-"^92)
for β ∈ (0.25, 0.5, 0.75), J ∈ (8, 16, 64)
    rng = MersenneTwister(20260908)
    R = 400_000
    su = zeros(R); sc = zeros(R); cr = zeros(R)
    for k ∈ 1:R
        (a,c) = pf(logy, J, β, σp, σm, rng)
        cr[k] = c
        su[k] = exp(a-exact); sc[k] = exp(a+c-exact)
    end
    d = sc .- su
    md = mean(d); sed = std(d)/sqrt(R)
    z = md/sed
    @printf("%4d %5.2f | mean %+.4f sd %.4f | %+.6f ± %.6f | z = %+6.2f %s\n",
            J, β, mean(cr), std(cr), md, sed, z, abs(z)>3 ? "***" : "")
end
