import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PRESET_DIR = BASE_DIR / "parameter_presets"
DATA_DIR = BASE_DIR / "curve_data"

class PresetManager:
    def __init__(self, filename:str):
        self.filepath = PRESET_DIR / filename

        # load or initialize
        if self.filepath.exists():
            try:
                with open(self.filepath, "r") as f:
                    self.data = json.load(f)
            except:
                self.data = {}
        else:
            self.data = {}

    def save_preset(self, name, param_dict):
        """Store a preset under a name."""
        self.data[name] = param_dict
        self._write()

    def load_preset(self, name):
        """Return parameters for a preset name."""
        return self.data[name]

    def delete_preset(self, name):
        if name in list(self.data.keys()):
            del self.data[name]
            self._write()

    def list_presets(self):
        return list(self.data.keys())

    def _write(self):
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=2)


class DataManager:
    def __init__(self, filename:str):
        self.filepath = DATA_DIR / filename

        # load or initialize
        if self.filepath.exists():
            try:
                with open(self.filepath, "r") as f:
                    self.data = json.load(f)
            except:
                self.data = {}
        else:
            self.data = {}

    def save_preset(self, name, param_dict):
        """Store a preset under a name."""
        self.data[name] = param_dict
        self._write()

    def load_preset(self, name):
        """Return parameters for a preset name."""
        return self.data[name]

    def delete_preset(self, name):
        if name in list(self.data.keys()):
            del self.data[name]
            self._write()

    def list_presets(self):
        return list(self.data.keys())

    def _write(self):
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=2)




