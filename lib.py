import numpy as np
from itertools import combinations
from math import tau, pi

angles = np.linspace(0,tau,360)
t01_100 = np.linspace(0,1,100)

def axis_pairs(dim): return list(combinations(range(dim),2))

def normalize_vector(vec):
  vec = np.array(vec)
  n = np.linalg.norm(vec)
  if n == 0: return vec
  return vec / n
