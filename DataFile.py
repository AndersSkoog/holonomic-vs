import json
from pathlib import Path

#BASE_DIR = Path(__file__).resolve().parent

class DataFile:
    def __init__(self, filepath):
        self.filepath = filepath
        # load or initialize
        if self.filepath.exists():
            try:
                with open(self.filepath, "r") as f:
                    self.data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                self.data = {}
        else:
            self.data = {}

    def has_key(self,key): return key in self.data.keys()


    def save(self, name, data_dict):
        self.data[name] = data_dict
        self._write()

    def load(self, key):
        if self.has_key(key): return self.data[key]
        else: return None

    def delete(self, key):
        if self.has_key(key):
          del self.data[key]
          self._write()
          return True
        else: return False


    def _write(self):
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=2)




