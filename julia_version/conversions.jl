module Primitives

struct S²
  θ::Float64
  ϕ::Float64
  function S²(θ::Float64,ϕ::Float64)
    θ=mod(θ,2π)
    ϕ=mod(ϕ,2π)
    new(θ,ϕ)
  end
end


struct Frame3
    m::

    function Frame3(dir::S², roll::Float64)
        y, p, r = dir.θ, dir.ϕ, roll
        yc, ys = cos(y), sin(y)
        pc, ps = cos(p), sin(p)
        rc, rs = cos(r), sin(r)
        Rz = [
            yc -ys 0.0;
            ys  yc 0.0;
            0.0 0.0 1.0
        ]
        Ry = [
            pc 0.0 ps;
            0.0 1.0 0.0;
            -ps 0.0 pc;
        ]

        Rx = [
            1.0 0.0 0.0;
            0.0  rc -rs;
            0.0  rs  rc
        ]
        res=Rz*Ry*Rx

        new(Rz * Ry * Rx)
    end
end

struct Basis



end






def Rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
return np.array([[c, -s, 0],[s, c, 0],[0, 0, 1]],dtype=float)

def Ry(pitch):
    c, s = np.cos(pitch), np.sin(pitch)
return np.array([[c, 0, s],[0, 1, 0],[-s, 0, c]],dtype=float)

def Rx(roll):
    c, s = np.cos(roll), np.sin(roll)
return np.array([[1, 0, 0],[0, c, -s],[0, s, c]],dtype=float)




struct Axis
  vec::Vector{Float64}
  function Axis(i::Int8,n::Int8)
    vec=zeros(n)
    vec[i] = val
    new(vec)
  end
end





struct Frame2
  x::xAxis
  y::yAxis
  function Frame2()

  end
end

struct Frame3
  x::xAxis
  y::yAxis
  z::zAxis
  function Frame3()

  end
end

struct Frame4
  x::xAxis
  y::yAxis
  z::zAxis
  w::wAxis
  function Frame4()

  end
end






struct S¹
  θ::Float64

  function S¹
    θ = mod(θ + π, 2π) - π
    new(θ)
  end



struct B¹

end

struct B²

end

struct B³


end


struct R²
  x::float64
  y::float64
end


struct R³
  x::float64
  y::float64
  z::float64
end

struct R⁴
  x::Float64
  y::Float64
  z::Float64
  w::Float64
end

struct C¹
  z::ComplexF64
end

struct C²
  z1::ComplexF64
  z2::ComplexF64
end


end


module Conversions

function toR³(c::S²)
  r=cos(c.ϕ)
  x=r*cos(c.θ)
  y=r*sin(c.θ)
  z=sin(c.ϕ)
  R³(x,y,z)
end

end
