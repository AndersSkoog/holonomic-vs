from typing import Tuple

cmplx_ser = Tuple[float,float] # serialized complex number
unitvec3 = Tuple[float,float,float]
vec3 = Tuple[float,float,float]
quat = Tuple[float,float,float,float]
unitquat = Tuple[float,float,float,float]
zpair = Tuple[complex,complex]
mobius_sphere_orient = Tuple[complex,complex]
mobius_coef = Tuple[complex,complex,complex,complex]
mobius_coef_ser = Tuple[cmplx_ser,cmplx_ser,cmplx_ser,cmplx_ser] #serialized mobius transform coefficents
orient_vec3 = Tuple[float,float,float]
orient_axis_ang = Tuple[unitvec3,float]


