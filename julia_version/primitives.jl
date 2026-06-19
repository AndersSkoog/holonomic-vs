module Primitives

export S¹, B², S², B³, vec2, vec3, vec4, toVec, toArray

# -------------------------
# S¹
# -------------------------
struct S¹
    θ::Float64
    function S¹(θ::Float64)
        θ = mod(θ, 2π)
        new(θ)
    end
end


# -------------------------
# B² (disk / polar)
# -------------------------
struct B²
    r::Float64
    θ::Float64
    function B²(θ::Float64, r::Float64)
        r = clamp(r, 0.0, 1.0)
        θ = mod(θ, 2π)
        new(θ,r)
    end
end


# -------------------------
# S²
# -------------------------
struct S²
    θ::Float64
    ϕ::Float64
    function S²(θ::Float64, ϕ::Float64)
        new(mod(θ, 2π), mod(ϕ, 2π))
    end
end


struct B³
    r::Float64
    θ::Float64
    ϕ::Float64
    function B³(θ::Float64, ϕ::Float64, r::Float64)
        new(clamp(r, 0.0, 1.0),mod(θ, 2π),mod(ϕ, 2π))
    end
end


struct vec2
    x::Float64
    y::Float64
end

struct vec3
    x::Float64
    y::Float64
    z::Float64
end

struct vec4
    x::Float64
    y::Float64
    z::Float64
    w::Float64
end

toVec(v::S¹) = vec2(sin(v.θ),cos(v.θ))
toVec(v::S²) = vec2(sin(v.θ),cos(v.θ)
toVec(v::B²) = vec2(v.r * sin(v.θ),v.r * cos(v.θ))
toVec(v::S²) = vec3(cos(v.ϕ) * cos(v.θ),cos(v.ϕ) * sin(v.θ),sin(v.ϕ))
toVec(v::B³) = vec3(v.r * (cos(v.ϕ) * cos(v.θ)),v.r * (cos(v.ϕ) * sin(v.θ)),v.r * sin(v.ϕ))
toVec(v::ComplexF64) = vec2(real(v),imag(v))

toArray(v::vec2) = [v.x,v.y]
toArray(v::vec3) = [v.x,v.y,v.z]
toArray(v::vec4) = [v.x,v.y,v.z,v.w]
toArray(v::S¹) = toArray(toVec(v))
toArray(v::B²) = toArray(toVec(v))
toArray(v::S²) = toArray(toVec(v))
toArray(v::B³) = toArray(toVec(v))
toArray(v::ComplexF64) = toArray(toVec(v))


end #End Primitives module

module Orientation

import S², vec2, vec3, vec4, toVec from Primitives
export SO3, UnitQuaternion, SU2, MobiusTrans

struct Orient
  dir::S²
  ang::Float64
end

struct SO3
    mtx::Matrix{Float64}

    function SO3(dir::S²,ang::Float64)
        x, y, z = vec3(dir)
        c = cos(ang)
        s = sin(ang)
        C = 1 - c

        R = [
            c + x*x*C   x*y*C - z*s   x*z*C + y*s
            y*x*C + z*s  c + y*y*C     y*z*C - x*s
            z*x*C - y*s  z*y*C + x*s   c + z*z*C
            ]

        new(R)
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

end # End Orientation Module

module Construction
using Random
using LinearAlgebra
import S² from Primitives

function random_closed_sphere_curve(n::Int=360, k::Int=5)

    t = range(0, 2π, length=n+1)[1:end-1]

    θ = zeros(Float64, n)
    ϕ = zeros(Float64, n)

    for i in 1:k

        Aθ, Bθ = randn(), randn()
        Aϕ, Bϕ = randn(), randn()

        phaseθ = 2π * rand()
        phaseϕ = 2π * rand()

        θ .+= Aθ .* cos.(i .* t .+ phaseθ) .+
            Bθ .* sin.(i .* t .+ phaseθ)

        ϕ .+= Aϕ .* cos.(i .* t .+ phaseϕ) .+
            Bϕ .* sin.(i .* t .+ phaseϕ)
    end

    [S²(θ[j], ϕ[j]) for j in eachindex(θ)]
end


struct ArcS²
    from::S²
    to::S²
end

function toS²(v::Array{Float64})
    x,y,z = v[0],v[1],v[2]
    r = sqrt((x*x)+(y*y)+(z*z))
    theta = atan2(y,x)  # longitude
    phi = acos(z / r)
    retrun S²(theta,phi)
end

function rotation(v::ArcS²)
    v1,v1 = toArray(v.from)),toArray(v.to))
    ang = acos(clamp(dot(v1, v2), -1, 1))
    ax  = normalize(cross(v1, v2))
    dir = toS²(ax)
    return SO3(dir,ang)
end

function rolltranslation(curve::Array{S²}, i::Int, contact::vec3, R::SO3)
    r=1.0
    arc = ArcS²(curve[i],curve[i+1])
    ang = acos(clamp(dot(toArray(arc.from), toArray(arc.to)), -1, 1))
    R_inc = rotation(arc)
    R_new = R * R_inc
    n = [R_new[1,3], R_new[2,3], R_new[3,3]]
    d = [0.0,0.0,-1.0]
    move_dir = cross3(n, d)
    mdn = norm3(move_dir)

    if mdn < 1e-12
        disp = [0.0, 0.0, 0.0]
    else
        move_dir = normalize(move_dir)
        disp = [move_dir[0] * r * ang,
                move_dir[1] * r * ang,
                move_dir[2] * r * ang]
    end

    contact_new = contact .+ disp

    return contact_new, R_new
end

function inital_R(curve::Array{S²}):
    arc = ArcS²(S²(0,pi/2),curve[0])
    return rotation(arc)
end


function extract_holonomic_roll_data(curve::Array{S²})

    prev_contact = vec3(0.0, 0.0, 0.0)
    prev_R = initial_R(curve)

    contact_curve = vec3[]
    orient_curve = SO3[]
    torsion_curve = vec3[]

    for i in 1:length(curve)-2

        contact, R = rolltranslation(
            curve,
            i,
            prev_contact,
            prev_R
            )

        tor_z = R.mtx * [0.0, 0.0, 1.0]

        tor_pt = vec3(
            contact.x,
            contact.y,
            tor_z[3]
            )

        push!(contact_curve, contact)
        push!(orient_curve, R)
        push!(torsion_curve, tor_pt)

        prev_contact = contact
        prev_R = R
    end

    return contact_curve, orient_curve, torsion_curve
end

function in_unit_disc(p)


end


function holonomic_view_3(contact_curve::Array{vec3},sel_index::Int)
    stereograpic_coords = map(v-> C²(v),contact_curve)



    rot_coefs = MobiusTrans(contact_curve[sel_index])
    a1,b1,c1,d1 = rot_coefs[0]
    a2,b2,c2,d2 = rot_coefs[1]


a = 1 / np.sqrt(1 + m_sel)
b = z_sel / np.sqrt(1 + m_sel)
c = -np.conjugate(z_sel) / np.sqrt(1 + m_sel)
d = a
sphere_point = inv_stereographic_proj(z_sel)
# Möbius-transformed cap
#z_cap = [((z*a) + b)/((z*c) + d) for z in cap_circle]
#sphere_cap_circle = [inv_stereographic_proj(z) for z in z_cap]
#sphere_cap_circle = cap_circle(sphere_point)

persp_point = sphere_point * 2

view_points1 = []
view_points2 = []
for p in curve:
    z_scaled = cap_scale * to_complex(p)
z_trans = ((a*z_scaled) + b) / ((c*z_scaled) + d)
view_points1.append(inv_stereographic_proj(z_trans))

for z in disc_curve:
    z_scaled = cap_scale * z
z_trans = ((a*z_scaled) + b) / ((c*z_scaled) + d)
view_points2.append(inv_stereographic_proj(z_trans))


return {
    "persp_point": persp_point,
    "disc_curve":disc_curve,
    "view_points1": view_points1,
    "view_points2":view_points2,
    "sphere_point": sphere_point
    }







