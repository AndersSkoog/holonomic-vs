from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

def proj_file_path(path:str): return str(BASE_DIR) + path

def read_file(filepath:str) -> str:
  f = open(filepath)
  txt = f.read()
  f.close()
  return txt

def adj(l): return [[l[i - 1], l[i]] for i in range(1, len(l))]
def adjacents(l): return [[l[i], l[i + 1]] for i in range(len(l) - 1)]

