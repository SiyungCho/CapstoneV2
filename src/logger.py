import json
import os
from datetime import datetime

class JsonLogger:
    def __init__(self, log_dir):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, "checkpoints", current_time)
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_path = os.path.join(self.log_dir, "log.txt")
        self.config_path = os.path.join(self.log_dir, "config.json")
        self.model_arguments_path = os.path.join(self.log_dir, "model_arguments.json")
        self.data_arguments_path = os.path.join(self.log_dir, "data_arguments.json")
        self.train_arguments_path = os.path.join(self.log_dir, "train_arguments.json")
        self.patchtstmodel_arguments_path = os.path.join(self.log_dir, "patchtstmodel_arguments.json")
        self.create_file(self.config_path)
        self.create_file(self.model_arguments_path)
        self.create_file(self.log_path)
        self.create_file(self.data_arguments_path)
        self.create_file(self.train_arguments_path)
        self.create_file(self.patchtstmodel_arguments_path)
    
    def create_file(self, path):
        if not os.path.exists(path):
            if path.endswith(".json"):
                with open(path, "w") as f:
                    f.write(json.dumps({}, indent=4))
            elif path.endswith(".txt"):
                with open(path, "w") as f:
                    f.write("")

    def log(self, data, *, mode="append", log_type="log", skip_if_exists=False):
        if log_type == "config":
            self._write_json(self.config_path, data, mode, skip_if_exists)
        elif log_type == "model_arguments":
            self._write_json(self.model_arguments_path, data, mode, skip_if_exists)
        elif log_type == "log":
            self._write_text_line(self.log_path, data)
        elif log_type == "data_arguments":
            self._write_json(self.data_arguments_path, data, mode, skip_if_exists)
        elif log_type == "train_arguments":
            self._write_json(self.train_arguments_path, data, mode, skip_if_exists)
        
    def _read_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _write_json(self, path, data, mode, skip_if_exists):
        existing = self._read_json(path)

        if existing == data and skip_if_exists:
            return

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def _write_text_line(self, path, data):
        if isinstance(data, dict):
            line = json.dumps(data, ensure_ascii=False)
        else:
            line = str(data)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {line}\n")