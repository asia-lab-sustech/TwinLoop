import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import threading
import os
from collections import deque

_MAX_RAW   = 50000
_MAX_AVG   = 50000

class TrainingVisualizer:
    def __init__(self, save_path='pic/training_curve.png', window_size=50):
        self.save_path   = os.path.abspath(save_path)
        self.window_size = window_size

        self.rewards     = deque(maxlen=_MAX_RAW)
        self.avg_rewards = deque(maxlen=_MAX_AVG)
        self.losses      = deque(maxlen=_MAX_RAW)
        self.avg_losses  = deque(maxlen=_MAX_AVG)

        self.lock    = threading.Lock()
        self.counter = 0

        try:
            save_dir = os.path.dirname(self.save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
        except Exception:
            pass

        print(f"Visualizer initialized. Plot will be saved to: {self.save_path}")

    def add_latency(self, reward):
        with self.lock:
            self.rewards.append(reward)
            self._update_moving_avg(self.rewards, self.avg_rewards)
            self._trigger_save()

    def add_loss(self, loss):
        with self.lock:
            self.losses.append(loss)
            self._update_moving_avg(self.losses, self.avg_losses)
            self._trigger_save()

    def _update_moving_avg(self, data_deque, avg_deque):
        n = min(len(data_deque), self.window_size)
        import itertools
        window = list(itertools.islice(
            reversed(data_deque), n))          
        avg_deque.append(float(np.mean(window)))

    def _trigger_save(self):
        self.counter += 1
        if self.counter == 1 or self.counter % 10 == 0:
            self._save_plot_thread_safe()

    def save_final(self):
        print("Saving final plot...")
        with self.lock:
            self._save_plot_thread_safe()

    def _save_plot_thread_safe(self):
        try:
            fig    = Figure(figsize=(10, 8))
            canvas = FigureCanvasAgg(fig) 
            
            raw_rewards  = list(self.rewards)
            avg_rewards_ = list(self.avg_rewards)
            raw_losses   = list(self.losses)
            avg_losses_  = list(self.avg_losses)

            # --- 1: Latency ---
            ax1 = fig.add_subplot(211)
            if raw_rewards:
                ax1.plot(raw_rewards,  color='gray', alpha=0.3,
                         label='Raw Latency', linewidth=0.5)
                ax1.plot(avg_rewards_, color='red',  linewidth=2,
                         label=f'Avg Latency (window={self.window_size})')
            ax1.set_ylabel('Latency (s)')
            ax1.set_xlabel('Total Tasks Processed (All-cars)')
            ax1.set_title('Task Response Time')
            ax1.legend(loc='upper left')
            ax1.grid(True, linestyle='--', alpha=0.5)

            # --- 2: Loss ---
            ax2 = fig.add_subplot(212)
            if raw_losses:
                ax2.plot(raw_losses,  color='gray', alpha=0.3,
                         label='Raw Loss', linewidth=0.5)
                ax2.plot(avg_losses_, color='blue', linewidth=2,
                         label=f'Avg Loss (window={self.window_size})')
            ax2.set_xlabel(
                'Steps of training (depends on train_frequency & min_memory_size in dqn agent)')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training Loss')
            ax2.legend(loc='upper right')
            ax2.grid(True, linestyle='--', alpha=0.5)

            fig.tight_layout()
            fig.savefig(self.save_path, dpi=100)
            del fig, canvas

        except Exception as e:
            print(f"Error saving plot: {e}")