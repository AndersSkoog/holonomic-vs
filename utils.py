from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

def proj_file_path(path:str): return str(BASE_DIR) + path

def read_file(filepath:str) -> str:
  f = open(filepath)
  txt = f.read()
  f.close()
  return txt

