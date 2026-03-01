codes = Dict(
    6=>:(out[i] += scheme(U[i]+U[i+1], q[i-2],q[i-1],q[i],q[i+1],q[i+2],q[i+3])),
    4=>:(out[i] += scheme(U[i]+U[i+1], q[i-1],q[i],q[i+1],q[i+2])),
    2=>:(out[i] += scheme(U[i]+U[i+1], q[i],q[i+1])),
    0=>:(out[i] += 0)
)

zero, two, four, six = codes[0], codes[2], codes[4], codes[6]


for o in [2,4,6]
    code = codes[o]
    @eval begin
        function rightpencil(out::Q, q::R, U::P, order::Val{$o}, ax::Axis{HALOED}, scheme::S) where{P,Q,R,S}
            for i in CENTERS(ax)
                $code
            end
        end
    end
end

@eval begin
    function rightpencil(out::Q, q::R, U::P, order::Val{6}, (;n)::Axis{CLOSED}, scheme::S) where{P,Q,R,S}
        i = 1
        $two
        i = 2
        $four
        for i in 3:n-2
            $six
        end
        i = n-1
        $four
        i = n
        $two
    end
    function rightpencil(out::Q, q::R, U::P, order::Val{4}, (;n)::Axis{CLOSED}, scheme::S) where{P,Q,R,S}
        i = 1
        $two
        for i in 2:n-1
            $four
        end
        i = n
        $two
    end
    function rightpencil(out::Q, q::R, U::P, order::Val{2}, (;n)::Axis{CLOSED}, scheme::S) where{P,Q,R,S}
        for i in 1:n
            $two
        end
    end
end

@eval begin
    function rightpencil(out::Q, q::R, U::P, order::O, ax::A, scheme::S) where{O,P,Q,R,S,A<:AXIS}
        for i in CENTERS(ax)
            if o[i] == 6
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
