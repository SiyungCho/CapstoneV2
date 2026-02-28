import json
import os
from datetime import datetime
import time

import matplotlib.pyplot as plt

from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities import rank_zero_only

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
        elif log_type == "patchtstmodel_arguments":
            self._write_json(self.patchtstmodel_arguments_path, data, mode, skip_if_exists)
        
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


class CustomLightningLogger(Logger):
    def __init__(self, json_logger, flops_analyzer=None, total_flops=0, epoch_timer_callback=None, loss_history_callback=None):
        super().__init__()
        self.json_logger = json_logger
        self.start_time = time.time()
        self.flops_analyzer = flops_analyzer
        self.total_flops = total_flops
        self.epoch_timer_callback = epoch_timer_callback
        self.loss_history_callback = loss_history_callback

    @property
    def name(self):
        return "logger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # Log hyperparameters to the logger.
        self.json_logger.log(params, log_type="config")

    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics = {k: float(v) for k, v in metrics.items()}
        self.json_logger.log({"step": step, **metrics}, log_type="log")

    @rank_zero_only
    def save(self):
        # Save the logger state if needed.
        pass

    @rank_zero_only
    def finalize(self, status):
        end_time = time.time()
        duration = end_time - self.start_time
        self.json_logger.log(f"Experiment finalized with status: {status}. Total duration: {duration:.2f} seconds.", log_type="log")
        self.generate_train_summary(duration, self.total_flops, self.flops_analyzer, self.epoch_timer_callback)

    def generate_train_summary(self, training_duration, total_flops, flops_analyzer, epoch_timer_callback):
        hours, rem = divmod(training_duration, 3600)
        minutes, seconds = divmod(rem, 60)
        self.json_logger.log("\n--- Training Summary ---", log_type="log")
        self.json_logger.log(f"Total training runtime: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}", log_type="log")
        
        if total_flops > 0 and flops_analyzer is not None:
            self.json_logger.log(f"Total FLOPs per forward pass: {total_flops / 1e9:.2f} GFLOPs", log_type="log")
            self.json_logger.log("--- FLOPs Breakdown by Module ---", log_type="log")
            self.json_logger.log(flops_analyzer.by_module(), log_type="log")
            self.json_logger.log("---------------------------------", log_type="log")
        self.json_logger.log("------------------------\n", log_type="log")
    
        if epoch_timer_callback and epoch_timer_callback.epoch_times:
            self.json_logger.log("Generating plot for epoch training times...", log_type="log")
            plt.figure(figsize=(10, 6))
            num_epochs_completed = range(1, len(epoch_timer_callback.epoch_times) + 1)
            plt.plot(num_epochs_completed, epoch_timer_callback.epoch_times, marker='o', linestyle='-')
            plt.title('Training Time per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Time (seconds)')
            plt.grid(True)
            plt.xticks(list(num_epochs_completed))
            plt.tight_layout()
            plt.savefig(self.json_logger.log_dir + "/epoch_times.png")
            plt.close() # Close the figure to free up memory
            self.json_logger.log(f"Epoch times plot saved to epoch_times.png", log_type="log")


        lh = self.loss_history_callback
        if lh and (lh.train_losses or lh.val_losses):
            epochs = list(range(1, max(len(lh.train_losses), len(lh.val_losses)) + 1))
            train_vals = (lh.train_losses + [None] * (len(epochs) - len(lh.train_losses)))
            val_vals   = (lh.val_losses   + [None] * (len(epochs) - len(lh.val_losses)))

            plt.figure(figsize=(10, 6))
            if any(v is not None for v in train_vals):
                plt.plot(epochs, train_vals, marker="o", linestyle="-", label="train loss")
            if any(v is not None for v in val_vals):
                plt.plot(epochs, val_vals, marker="o", linestyle="-", label="val loss")
            plt.title("Train/Val Loss by Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.xticks(epochs)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.json_logger.log_dir + "/loss_curves.png")
            plt.close()