codes = Dict(
    6=>:(out[i] = U[i]*scheme(U[i], q[i-3],q[i-2],q[i-1],q[i],q[i+1],q[i+2])),
    4=>:(out[i] = U[i]*scheme(U[i], q[i-2],q[i-1],q[i],q[i+1])),
    2=>:(out[i] = U[i]*scheme(U[i], q[i-1],q[i])),
    0=>:(out[i] = 0)
)

zero, two, four, six = codes[0], codes[2], codes[4], codes[6]


for o in [2,4,6]
    code = codes[o]
    @eval begin
        function leftpencil(out::Q, q::P, U::P, order::Val{$o}, ax::Axis{HALOED}, scheme::S) where{P,Q,S}
            for i in EDGES(ax)
                $code
            end
        end
    end
end


@eval begin
    function leftpencil(out::Q, q::P, U::P, order::Val{6}, (;n)::Axis{CLOSED}, scheme::S) where{P,Q,S}
        i = 1
        $zero
        i = 2
        $two
        i = 3
        $four
        for i in 4:n-2
             $six
        end
        i = n-1
        $four
        i = n
        $two
        i = n+1
        $zero
    end
    function leftpencil(out::Q, q::P, U::P, order::Val{4}, (;n)::Axis{CLOSED}, scheme::S) where{P,Q,S}
        i = 1
        $zero
        i = 2
        $two
        for i in 3:n-1
            $four
        end
        i = n
        $two
        i = n+1
        $zero
    end
    function leftpencil(out::Q, q::P, U::P, order::Val{2}, (;n)::Axis{CLOSED}, scheme::S) where{P,Q,S}
        i = 1
        $zero
        for i in 2:n
            $two
        end
        i = n+1
        $zero
    end
end

@eval begin
    function leftpencil(out::Q, q::P, U::P, order::Q, ax::A, scheme::S) where{P,Q,S,A<:AXIS}
        for i in EDGES(ax)
            if order[i] == 6
                $six
            elseif o[i] == 4
                $four
            elseif o[i] == 2
                $two
            else
                $zero
            end
        end
    end
end
