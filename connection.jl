module Connection
using LinearAlgebra
using .S2
using .SO3
using .SU2
using .CP1
export roll_contact_point, init_roll_frame_SO3, roll_frame_SO3, roll_frame_SU2,roll_connection,roll_connection_view

function roll_contact_point(prev::ComplexF64,v::S2)
    θ,φ = theta(v),phi(v)
    r = sin(φ)
    x=real(prev) + r * cos(θ)
    y=imag(prev) + r * sin(θ)
    return Complex(x,y)
end

function init_roll_frame_SO3(sphere_curve::Array{S2})
    p1 = toArray(sphere_curve[length(sphere_curve)])
    p2 = toArray(sphere_curve[1])
    t = p2 .- p2
    u = t .- dot(t,p1) .* p1
    v = np.cross(p1,u)
    R = [p1[1] p1[2] p2[3]; u[1] u[2] u[3]; v[1] v[2] v[3]]
    return SO3(R)
end

funtion roll_frame_SO3(p1::S2,p2::S2,prev::SO3)
    pa1 = [p1.x,p1.y,p1.z]
    pa2 = [p2.x,p2.y,p2.z]
    ang = acos(clamp(dot(pa1,pa2), -1.0, 1.0))
    axis = normalize(cross(p1,p2))
    R_prev = prev.R
    R_inc = SO3_from_axis_angle(axis,ang).R
    R_new = R_prev * R_inc
    return SO3(R_new)
end

function torsion_pt(contact::ComplexF64,frame::SO3)
  x,y = real(contact),imag(contact)
  tor_vec = frame.R * [0.0,0.0,1.0]
  z = tor_vec[3]
  return [x,y,z]
end

function torsion_angle(contacts::Array{ComplexF64},frames::Array{SO3},ind::Int64)
  L = length(frames)
  Li = L-4
  diff = abs(ind-Li)
  si = ind < (L-4) ? ind : ind - diff
  t1 = torsion_pt(contacts[si],frames[si])
  t2 = torsion_pt(contacts[si+1],frames[si+1])
  t3 = torsion_pt(contacts[si+2],frames[si+2])
  t4 = torsion_pt(contacts[si+3],frames[si+3])
  d1 = t2 - t1
  d2 = t3 - t2
  d3 = t4 - t3
  n1 = normalize(cross(d1,d2))
  n2 = normalize(cross(d2,d3))
  x = dot(n1,n2)
  y = dot(b,cross(n1,n2))
  τ = atan(y,x)
  return τ
end

function torsion_angles(contacts::Array{ComplexF64},frames::Array{SO3})
  return [torsion_angle(contacts,frames,ind) for ind in 1:length(contacts)]
end


function roll_frame_SU2(prev::SU2,v::S2)
    return prev.U * SU2(UnitSpinor(v)).U
end


struct RollConnection
  contacts::Array{ComplexF64}
  framesSU2::Array{SU2}
  framesSO3::Array{SO3}
  torangles::Array{Float64}
end


#this is a heavy computation which should only run once for a chosen sphere curve,
#perhaps can be optimized by making the sphere curve fully evaluated during compile time.
#perhaps output can be stored in a file so that we dont have to run it each time we adjustments
function roll_connection(sphere_curve::Array{S2})
    prev_contact = 0+0im
    prev_frame_SO3 = init_roll_frame_SO3(sphere_curve)
    prev_frame_SU2 = SU2_from_UnitSpinor(UnitSpinor(sphere_curve[1]))
    contacts = []
    frames_SU2 = []
    frames_SO3 = []
    for ind in range 1:length(sphere_curve)
        contact = roll_contact_point(prev_contact,sphere_curve[ind])
        frame_SU2 = roll_frame_SU2(prev_frame_SU2,contact)
        frame_SO3 = roll_frame_SO3(prev_contact,contact,prev_frame_SO3)
        push!(contacts,contact)
        push!(frames_SU2,frame_SU2)
        push!(frames_SO3,frame_SO3)
        prev_contact = contact
        prev_frame_SU2 = frame_SU2
        prev_frame_SO3 = frame_SO3
    end
    torangles = torsion_angles(contacts,frames_SO3)
    return RollConnection(contacts,frames_SU2,frames_SO3,torangles)
end


#this is a bit ambigious but is saved for later reference, should probobly be moved to main holonomic.jl file
function roll_connection_view(contacts::Array{ComplexF64},sel_ind::Int64)
    s = tan(acos(4/5)/2)
    connection = roll_connection(sphere_curve)
    roll_frame = connection.roll_frames[sind].U
    a,b,c,d = roll_frame[1,1],roll_frame[1,2],roll_frame[2,1],roll_frame[2,2]
    pts = map(p-> MobiusTrans(p*s,(a,b,c,d)),connection.contacts)
    O = roll_frame
    K = connection.tor_frames[sind]
    return (pts=pts,O=O,K=K)
end











