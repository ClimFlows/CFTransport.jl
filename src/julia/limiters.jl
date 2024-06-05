"""
    slope = minmod(slope1, slope2)
Minmod limiter
"""
@inline function minmod(a::F, b::F) where F
    @fastmath if a*b>=0
        aa, bb = abs(a), abs(b)
        c = (1//2)*(a+b)
        d = min(c, 2aa, 2bb)
        return d * sign(a)
    else
        return zero(F)
    end
end

"""
    slope = minmod_simd(slope1, slope2)
Minmod limiter. This implementation avoids branching and may be more suitable
for SIMD vectorization.
"""
@inline @fastmath function minmod_simd(a::F, b::F) where F
    @inline min2(x,y) = x+y-abs(x-y) # 2min(a,b) without branching
    c = a+b
    d = min2(abs(c), 2*min2(abs(a),abs(b))) # min(2(a+b), 8min(a,b)) = 4min((a+b)/2, 2a,2b)
    return d * sign(c) * (1+sign(a*b))/8 # sign(c)*min((a+b)/2, 2a,2b)
end

@inline limited_slope((i,j), q, limiter) = @fast limiter(q[i,j]-q[i-1,j], q[i+1,j]-q[i,j])
