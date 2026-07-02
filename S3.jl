module S3
using LinearAlgebra
using .CP1
using .S2
using .SU2
using .Holomorph

struct C2
  z1::ComplexF64
  z2::ComplexF64
end

function stereo_proj(p::C2)
  x1 = real(p.z1)
  y1 = imag(p.z1)
  x2 = real(p.z2)
  y2 = imag(p.z2)
  d = 1 - y2
  return R3(x1 / d,y1 / d,x2 / d)
end

struct S3
  fiber1::Vector{C2}
  fiber2::Vector{C2}
end

basefiber_xy = [C2(cis(a),0+0im) for a in range(0,2*pi,360)]

function hopf_link(h::Holomorph2, ind::Int64)
  S = InvStereoProj(h.refpoint)
  R = SO3_from_axis_angle(S,h.torangle)
  a = InvStereoProj(h.result[ind])
  b = R ⊗ a
  U1 = SU2_from_axis_angle(a,pi)
  U2 = SU2_from_axis_angle(b,pi)
  fib1 = U1 ⊗ basefiber_xy
  fib2 = U2 ⊗ basefiber_xy
  return S3(fib1,fib2)
end

function hopf_fibration(h::Holomorph2)
  l = length(h.result)
  return [hopf_link(h,ind) for ind in 1:l]
end

function project_fiber(fiber::Vector{C2})
  return map(p-> stereo_proj(p),fiber)
end

function project_hopf_fibration(fibration::Vector{S3})
  fibers1 = map(p -> p.fiber1,fibration)
  fibers2 = map(p -> p.fiber2,fibration)
  proj_fibers1 = map(f -> project_fiber(f),fibers1)
  proj_fibers2 = map(f -> project_fiber(f),fibers2)
  return proj_fibers1,proj_fibers2
end










#ambigious but saved for later reference
function hopf_link(pts::Vector{ComplexF64},frames::Vector{SU2},ind::Int64)
  F1 = ind > 1 ? frames[ind-1] : init_frame
  F2 = frames[ind]
  K = torsion_su2(F1,F2)
  P=pts[ind] #selected Complex number returned by Mobius Transform of Roll contact points
  S=inv_stereo_proj(P) #P converted to S2
  U1 = SU2(UnitSpinor(S)) # Frame Associated to S2
  U2 = U1 * exp(K)
  fib1 = basefiber_xy * U1
  fib2 = basefiber_xy * U2
  return S3(fib1,fib2)
end

end
