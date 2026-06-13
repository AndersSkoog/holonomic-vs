from math import sin, cos, pi, tau, radians
from constants import MIN_VAL
from itertools import product

right_angle = pi/2
def abs_angle(val:float): return radians(val) % right_angle

def clip(val:float,low:float,high:float):
  if val < low: return low
  elif val > high: return high
  else: return val

#pauli matrices
o1 = np.array([[0,1],[1,0]], dtype=complex)
o2 = np.array([[0,-1j],[1j,0]], dtype=complex)
o3 = np.array([[1,0],[0,-1]], dtype=complex)
#SU2 identity
I = np.eye(2, dtype=complex)

def mobius_coef_from_sphere_point(SP,angle):
  x,y,z = SP
  dotsum=(x*o1)+(y*o2)+(z*o3)
  U=cos(angle/2)*I - (1j*sin(angle/2)*dotsum)
  a=U[0][0]
  b=U[0][1]
  c=U[1][0]
  d=U[1][1]
  return a,b,c,d


class riemann_sphere_coord:

  def __init__(self,theta:float,phi:float,spart:(int,int,int)):
    assert spart in sparts, "not valid sphere part argument"
    self._spart = spart
    self._theta = radians(theta) % right_angle
    self._phi = radians(phi) % right_angle
    self._pr = cos(self._phi)
    self._x = self._pr * cos(self._theta)
    self._y = self._pr * sin(self._theta)
    self._z = sin(self._phi)

  @staticmethod
  def disc_point_constructor(p):
    x,y = p
    zeta=complex(x,y)
    r=abs(zeta)
    theta=atan2(y,x)
    phi=acos(r)
    return riemann_sphere_coord(theta,phi)

  @property
  def at_inf: return np.isclose(self._z,1.0,MIN_VAL,MIN_VAL)

  @property
  def on_equator(self): return np.isclose(self._z,0.0,MIN_VAL,MIN_VAL)

  @property
  def equator_pt(self): return np.array([cos(self._theta),sin(self._theta),0.0])
  
  @property
  def theta(self):return self._theta

  @theta.setter
  def theta(self,val:float):
    self._theta = radians(val) % right_angle
    self._x = self._pr * cos(self._theta)
    self._y = self._pr * sin(self._theta)

  @property
  def phi(self): return self._phi

  @phi.setter
  def phi(self,val:float):
    self._phi = radians(val) % right_angle
    self._pr = cos(self._phi)
    self._x = self._pr * cos(self._theta)
    self._y = self._pr * sin(self._theta)
    self._z = sin(self._phi)

  @property
  def x(self): return self._x * self._spart[0]

  @property
  def y(self): return self._y * self._spart[1]

  @property
  def z(self): return self._z * self._spart[2]

  @property
  def zeta(self): return complex(self._x,self._y) / (1 - self.z)

  @property
  def xi(self): return complex(self._x,-self._y) / (1 + self.z)

  @property
  def cmplx_disc_point(self):
    if self.at_inf: return complex(0,0)
    return self.xi if self._spart[2] == 1 else self.zeta

  @property
  def cmplx_plane_point(self):
    if self.at_inf: return complex(inf,inf)
    return self.zeta if self._spart[2] == 1 else self.xi

  @property
  def north_a(self): return np.array([inf,inf,inf]) if self.at_inf else np.array([self._x,self._y,self._z])

  @property
  def south_a(self): return np.array([0.0,0.0,-1.0)]) if self.at_inf else np.array(self._x,self._y,-self._z)

  @property
  def north_b(self): return np.array([inf,inf,inf]) if self.at_inf else np.array([-self._x,self._y,self._z])

  @property
  def south_b(self): return np.array([0.0,0.0,-1.0)]) if self.at_inf else np.array(-self._x,self._y,-self._z)

  @property
  def north_c(self): return np.array([inf,inf,inf]) if self.at_inf else np.array([-self._x,-self._y,self._z])

  @property
  def south_c(self): return np.array([0.0,0.0,-1.0)]) if self.at_inf else np.array(-self._x,self._y,-self._z)

  @property
  def north_d(self): return np.array([inf,inf,inf]) if self.at_inf else np.array([self._x,-self._y,self._z])

  @property
  def south_d(self): return np.array([0.0,0.0,-1.0)]) if self.at_inf else np.array(self._x,-self._y,-self._z)

  def mobius_coef(self,angle):
    x,y,z = self.x,self.y,self.z
    dotsum=(x*o1)+(y*o2)+(z*o3)
    U=cos(angle/2)*I - (1j*sin(angle/2)*dotsum)
    a=U[0][0]
    b=U[0][1]
    c=U[1][0]
    d=U[1][1]
    return a,b,c,d

  def mobius_trans_rot(self,c:complex):
    SP=np.array([self.x,self.y,self.z])
    a,b,c,d = mobius_coef_from_sphere_point(SP,#we need to find the rotation angle) is this the angle between the north pole and SP ?
    return ((c*a)+b)/((c*c)+d)



      


      





      


      


      
    

      
  
  

  
      
        
