abstract type SCHEME end
abstract type UPWINDED <: SCHEME end
abstract type CENTERED <: SCHEME end
abstract type UPWENO <: UPWINDED end

" WJS(U, ...) : the original WENO scheme of Jiang and Shu (1996)"
struct WJS <: UPWENO end
" WZ(U, ...) : the WENO-Z scheme of Borges at al. (2007)"
struct WZ <: UPWENO end
" UP(U, ...) : the linear odd order reconstruction"
struct UP <: UPWINDED end
" CE(U, ...) : the linear even order reconstruction"
struct CE <: CENTERED end


ce(qmmm,qmm,qm,qp,qpp,qppp) =  (37*(qm+qp)-8*(qmm+qpp)+qmmm+qppp)*eltype(qm)(1/60)
ce(qmm,qm,qp,qpp) = (7(qm+qp)-(qmm+qpp))*eltype(qm)(1/12)
ce(qm,qp) = (qm+qp)*eltype(qm)(1/2)

up(qmmm,qmm,qm,qp,qpp) = (2qmmm-13qmm+47qm+27qp-3qpp)*eltype(qm)(1/60)
up(qmm,qm,qp) = (-qmm+5qm+2qp)*eltype(qm)(1/6)


@inline (::Type{UP})(u,qm,qp) = u > 0 ? qm : qp
@inline (::Type{UP})(u,qmm,qm,qp,qpp) = u > 0 ? up(qmm,qm,qp) : up(qpp,qp,qm)
@inline (::Type{UP})(u,qmmm,qmm,qm,qp,qpp,qppp) = u > 0 ? up(qmmm,qmm,qm,qp,qpp) : up(qppp,qpp,qp,qm,qmm)

@inline (::Type{WZ})(u,qm,qp) = u > 0 ? qm : qp
@inline (::Type{WZ})(u,qmm,qm,qp,qpp) = u > 0 ? wenoz(qmm,qm,qp) : wenoz(qpp,qp,qm)
@inline (::Type{WZ})(u,qmmm,qmm,qm,qp,qpp,qppp) = u > 0 ? wenoz(qmmm,qmm,qm,qp,qpp) : wenoz(qppp,qpp,qp,qm,qmm)


@inline (::Type{WJS})(u,qm,qp) = u > 0 ? qm : qp
@inline (::Type{WJS})(u,qmm,qm,qp,qpp) = u > 0 ? wenojs(qmm,qm,qp) : wenojs(qpp,qp,qm)
@inline (::Type{WJS})(u,qmmm,qmm,qm,qp,qpp,qppp) = u > 0 ? wenojs(qmmm,qmm,qm,qp,qpp) : wenojs(qppp,qpp,qp,qm,qmm)

@inline (::Type{CE})(u,qm,qp) = ce(qm,qp)
@inline (::Type{CE})(u,qmm,qm,qp,qpp) = ce(qmm,qm,qp,qpp)
@inline (::Type{CE})(u,qmmm,qmm,qm,qp,qpp,qppp) = ce(qmmm,qmm,qm,qp,qpp,qppp)

function wenojs(qmm::T,qm::T,qp::T) where T

    if (qmm==qm==qp)
        return qm
    end

    beta1 = (qm-qmm)^2
    beta2 = (qp-qm)^2

    w1 = beta2^2
    w2 = beta1^2

    q1 = -qmm +3qm
    q2 = qm + qp

    return T(1/2)*(w1*q1+w2*q2) / (w1+w2)

end

function wenoz(qmm::T,qm::T,qp::T) where T

    if (qmm==qm==qp)
        return qm
    end

    beta1 = (qm-qmm)^2
    beta2 = (qp-qm)^2

    tau = abs(beta2-beta1)

    w1 = beta2*(beta1 + tau)
    w2 = beta1*(beta2 + tau)*2

    q1 = -qmm +3qm
    q2 = qm + qp

    return T(1/2)*(w1*q1+w2*q2) / (w1+w2)

end


function wenojs(qmmm::T,qmm::T,qm::T,qp::T,qpp::T) where T

    k1, k2 = T(13/12), T(1/4)
    beta1 = k1*(qmmm -2qmm +qm)^2 + k2*(qmmm -4qmm +3qm)^2
    beta2 = k1*(qmm -2qm +qp)^2 + k2*(qmm -qp)^2
    beta3 = k1*(qm -2qp +qpp)^2 + k2*(3qm -4qp +qpp)^2

    w1 = (beta2^2*beta3^2)
    w2 = (beta1^2*beta3^2)*6
    w3 = (beta1^2*beta2^2)*3

    q1 = 2qmmm -7qmm +11qm
    q2 = -qmm +5qm +2qp
    q3 = 2qm +5qp -qpp

    return T(1/6)*(w1*q1+w2*q2+w3*q3)/(w1+w2+w3+floatmin(T))
end

function wenoz(qmmm::T,qmm::T,qm::T,qp::T,qpp::T) where T

    k1, k2 = T(13/12), T(1/4)
    beta1 = k1*(qmmm -2qmm +qm)^2 + k2*(qmmm -4qmm +3qm)^2
    beta2 = k1*(qmm -2qm +qp)^2 + k2*(qmm -qp)^2
    beta3 = k1*(qm -2qp +qpp)^2 + k2*(3qm -4qp +qpp)^2

    tau5 = abs(beta1-beta3)

    w1 = beta2*beta3*(beta1 + tau5)
    w2 = beta1*beta3*(beta2 + tau5)*6
    w3 = beta1*beta2*(beta3 + tau5)*3

    q1 = 2qmmm -7qmm +11qm
    q2 = -qmm +5qm +2qp
    q3 = 2qm +5qp -qpp

    return T(1/6)*(w1*q1+w2*q2+w3*q3)/(w1+w2+w3+floatmin(T))
end
