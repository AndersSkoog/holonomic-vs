module Orientation

import S², vec2, vec3, vec4, toVec from Primitives
export SO3, UnitQuaternion, SU2, MobiusTrans
# -----------------------w--
# SO(3)
# -------------------------
struct SO3
    mtx::Matrix{Float64}

    function SO3(dir::S², ang::Float64)
        y = dir.θ
        p = dir.ϕ
        r = ang

        yc, ys = cos(y), sin(y)
        pc, ps = cos(p), sin(p)
        rc, rs = cos(r), sin(r)

        Rz = [
            yc -ys 0.0;
            ys  yc 0.0;
            0.0 0.0 1.0
            ]

        Ry = [
            pc  0.0 ps;
            0.0 1.0 0.0;
            -ps 0.0 pc
            ]

        Rx = [
            1.0 0.0 0.0;
            0.0 rc -rs;
            0.0 rs  rc
            ]

        new(Rz * Ry * Rx)
    end
end


# -------------------------
# Unit Quaternion
# -------------------------
struct UnitQuaternion
    w::Float64
    i::Float64
    j::Float64
    k::Float64

    function UnitQuaternion(dir::S², ang::Float64)
        s = sin(ang / 2)
        w = cos(ang / 2)
        c = toVec(dir)

        i = c.x * s
        j = c.y * s
        k = c.z * s

        n = sqrt(w*w + i*i + j*j + k*k)

        new(w/n, i/n, j/n, k/n)
    end
end


# -------------------------
# SU(2) / S³ representation
# -------------------------
struct SU2
    U::Matrix{ComplexF64}

    function S³(dir::S², ang::Float64)
        c = toVec(dir)

        o1 = [0+0im 1+0im; 1+0im 0+0im]
        o2 = [0+0im 0-1im; 0+1im 0+0im]
        o3 = [1+0im 0+0im; 0+0im -1+0im]
        I  = [1+0im 0+0im; 0+0im 1+0im]

        dotsum = c.x*o1 + c.y*o2 + c.z*o3

        U = cos(ang/2)*I - im*sin(ang/2)*dotsum

        new(U)
    end
end


# -------------------------
# C² stereographic coordinates
# -------------------------
struct C²
    zeta::ComplexF64
    xi::ComplexF64

    function C²(dir::S²)
        c = toVec(dir)
        zeta = (c.x + c.y*im) / (1 - c.z)
        xi   = (c.x - c.y*im) / (1 + c.z)
        new(zeta, xi)
    end
end


# -------------------------
# Möbius rotational coefficients
# -------------------------
struct MobiusTrans
    zeta::Tuple{ComplexF64,ComplexF64,ComplexF64,ComplexF64}
    xi::Tuple{ComplexF64,ComplexF64,ComplexF64,ComplexF64}

    function MobiusRotCoef(dir::S²)
        c2 = C²(dir)

        z1, z2 = c2.zeta, c2.xi

        # first chart
        x1, y1 = real(z1), imag(z1)
        norm1 = sqrt(1 + x1*x1 + y1*y1)

        a1 = 1/norm1 + 0im
        b1 = z1/norm1
        c1 = conj(z1)/norm1
        d1 = 1/norm1 + 0im

        # second chart
        x2, y2 = real(z2), imag(z2)
        norm2 = sqrt(1 + x2*x2 + y2*y2)

        a2 = 1/norm2 + 0im
        b2 = z2/norm2
        c2 = conj(z2)/norm2
        d2 = 1/norm2 + 0im

        new((a1,b1,c1,d1), (a2,b2,c2,d2))
    end
end

end
