"""sterographic projection of point in the plane to a point on the sphere centered at [0,0,r]"""
def plane_to_sphere(p, r):
  X, Y = p[0],p[1]
  X2, Y2 = pow(X,2),pow(Y,2)
  r2 = pow(r,2)
  denom = X2 + Y2 + r2
  x = (2 * r2 * X) / denom
  y = (2 * r2 * Y) / denom
  z = r * (X2 + Y2 - r2) / denom + r  # center shift up by r
  return [x, y, z]

