from pathlib import Path
import numpy as np

def proj_path(): return Path.cwd().parent

def proj_file_path(path:str): return str(Path.cwd()) + path

def read_file(path: str) -> str:
  f = open(proj_file_path(path))
  txt = f.read()
  f.close()
  return txt

