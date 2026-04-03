# scripts/dt_server.py

import sys
import os
import socket
import json
import threading
import argparse
import random
import torch
from concurrent.futures import ThreadPoolExecutor
import matplotlib
matplotlib.use('Agg')
from plotter import TrainingVisualizer
import csv
from typing import Optional
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.agent_factory import AgentFactory
from src.environment.rl_parser import RLParser
from src.utils.config_loader import ConfigLoader

class DTServer:
    def __init__(self, port: int, save_path: str, init_weights_path: str, plot_path: str = "pic/dt.png"):
        self.port = port
        self.save_path = save_path
        self.init_weights_path = init_weights_path
        
        self.config_loader = ConfigLoader()
        self.rl_parser = RLParser()
        
        self.car_agents: dict = {}
        self.agents_lock = threading.Lock()
        self._departed_weights: dict = {}
        print("[DT Server] Loading agent config template...")
        self._agent_config_template = self.config_loader.load_agent_config("agent_dt")

        self.global_weights_dict = {}
        if os.path.exists(self.init_weights_path):
            try:
                loaded = torch.load(self.init_weights_path, map_location='cpu', weights_only=False)
                if isinstance(loaded, dict) and 'weights_dict' in loaded:
                    self.global_weights_dict = loaded['weights_dict']
                else:
                    self.global_weights_dict = loaded
                print(f"[DT Server] ✅ Loaded global weights dict for {len(self.global_weights_dict)} cars.")
                
            except Exception as e:
                print(f"[DT Server] ⚠️ Failed to load init weights: {e}")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', self.port))
        
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.visualizer = TrainingVisualizer(save_path=plot_path, window_size=10)
        print(f"[DT Server] 🚀 Listening on UDP port {self.port}")
        print(f"[DT Server] 💾 Models will be saved to: {self.save_path}")

    def _get_or_create_agent(self, vid: str, car_id: int):
        with self.agents_lock:
            if vid not in self.car_agents:
                new_agent = AgentFactory.create_agent(f"dt_car_{vid}", self._agent_config_template)
                
                if hasattr(new_agent, 'set_loss_callback'):
                    def make_loss_callback(cid):
                        def cb(loss_val):
                            self.visualizer.add_loss(loss_val)
                        return cb
                    new_agent.set_loss_callback(make_loss_callback(car_id))

                if self.global_weights_dict and vid in self.global_weights_dict:
                    new_agent.set_model_weights(self.global_weights_dict[vid], sync_type="sync")
                    print(f"[DT Server] Car {vid} (ID:{car_id}): 🧠 Loaded exclusive memory.")
                
                elif self.global_weights_dict and self.car_agents:
                    mentor_vid = random.choice(list(self.car_agents.keys()))
                    mentor_weights = self.car_agents[mentor_vid].get_model_weights()
                    new_agent.set_model_weights(mentor_weights, sync_type="sync")
                    print(f"[DT Server] Car {vid} (ID:{car_id}): 🧬 Cloned from '{mentor_vid}'.")               

                else:
                    print(f"[DT Server] Car {vid} (ID:{car_id}): 🐣 Cold start (independent).")
                
                self.car_agents[vid] = new_agent
                
            return self.car_agents[vid]

    def handle_message(self, data: bytes, addr):
        try:
            message = json.loads(data.decode('utf-8').strip())
        except Exception as e:
            print(f"[DT] Decode error from {addr}: {e}")
            return

        msg_type = message.get('msg_type', '').lower()
        
        if msg_type == 'save_and_sync':
            with self.agents_lock:
                agents_copy = dict(self.car_agents)

            if agents_copy:
                all_weights = {}
                for v, agent in agents_copy.items():
                    w = agent.get_model_weights()
                    if w is not None:
                        all_weights[v] = w
                
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                torch.save(all_weights, self.save_path)
                print(f"[DT Server] ✅ Super dict saved ({len(all_weights)} agents) → {self.save_path}")
            else:
                print("[DT Server] ⚠️ No weights to save")

            print(f"[DT Server] Saving plot → {self.visualizer.save_path}")
            self.visualizer.save_final()
            return

        if msg_type == 'save_scenario':
            scenario_name = message.get('scenario', 'unknown')
            with self.agents_lock:
                agents_copy = dict(self.car_agents)

            if agents_copy:
                all_weights = {}
                for v, agent in agents_copy.items():
                    w = agent.get_model_weights()
                    if w is not None:
                        all_weights[v] = w
                for v, w in self._departed_weights.items():
                    if v not in all_weights:
                        all_weights[v] = w
                print(f"[DT Server] 📦 Total: {len(all_weights)} agents "
                      f"({len(agents_copy)} online + {len(all_weights)-len(agents_copy)} departed)")

                save_dir = os.path.dirname(self.save_path)
                os.makedirs(save_dir, exist_ok=True)
                scenario_path = os.path.join(save_dir, f"dt_{scenario_name}.pth")
                
                save_data = {
                    'scenario': scenario_name,
                    'weights_dict': all_weights
                }
                # torch.save(save_data, scenario_path)
                torch.save(all_weights, scenario_path)
                print(f"[DT Server] ✅ Scenario '{scenario_name}' super dict saved ({len(all_weights)} agents) → {scenario_path}")
            else:
                print(f"[DT Server] ⚠️ No weights to save for scenario '{scenario_name}'")

            print(f"[DT Server] Saving plot → {self.visualizer.save_path}")
            self.visualizer.save_final()
            return
        
        car_id = message.get('car_id')
        if car_id is None:
            return
            
        car_id = int(car_id)
        vid = message.get('vid', f"unknown_{car_id}")
        task_id = message.get('task_id', -1)
        unique_task_key = f"{vid}_{task_id}"

        if msg_type == 'leave':
            with self.agents_lock:
                if vid in self.car_agents:
                    agent = self.car_agents[vid]
                    w = agent.get_model_weights()
                    if w is not None:
                        self._departed_weights[vid] = w
                    with agent.lock:
                        stale = [k for k in agent.pending_tasks if str(k).startswith(f"{vid}_")]
                        for k in stale:
                            del agent.pending_tasks[k]
                    del self.car_agents[vid]
                    print(f"[DT Server] Car {vid} departed, weights saved.")
            return
        agent = self._get_or_create_agent(vid, car_id)
        
        if msg_type == 'state':
            try:
                state_vec, valid_acts, _ = self.rl_parser.parse_state_message(message)
                
                action_idx, action_name = agent.select_action(
                    state_vec, valid_acts,
                    exploration=True,  
                    task_id=unique_task_key
                )
                self.sock.sendto(str(action_idx).encode('utf-8'), addr)

            except Exception as e:
                print(f"[DT State Error] car={vid}: {e}")

        elif msg_type == 'reward':
            try:
                reward = self.rl_parser.parse_reward_message(message)
                raw_latency = float(message.get('latency', 0.0))
                self.visualizer.add_latency(raw_latency)

                #
                # simtime = message.get('sim_time', message.get('task_id', None))
                # try: simtime_val = float(simtime)
                # except Exception: simtime_val = str(simtime)
                
                # task_id_val = message.get('task_id', None)
                # try: task_id_val = int(task_id_val)
                # except Exception: task_id_val = str(task_id_val)
                
                # try:
                #     scripts_dir = os.path.dirname(os.path.abspath(__file__))
                #     pic_dir = os.path.join(scripts_dir, 'pic')
                #     os.makedirs(pic_dir, exist_ok=True)
                #     csv_path = os.path.join(pic_dir, 'metrics_tasks.csv')
                # except Exception:
                #     csv_path = os.path.join('.', 'metrics_tasks.csv')

                # header = 'sim_time,vid,latency_s,task_id,action_idx,action_name'
                # try:
                #     if os.path.exists(csv_path):
                #         try:
                #             with open(csv_path, 'r', newline='') as f:
                #                 first = f.readline().strip()
                #         except Exception:
                #             first = ''
                #         if first != header:
                #             try:
                #                 with open(csv_path, 'r', newline='') as f:
                #                     content = f.read()
                #             except Exception:
                #                 content = ''
                #             tmp_path = csv_path + '.tmp'
                #             try:
                #                 with open(tmp_path, 'w', newline='') as tmpf:
                #                     tmpf.write(header + '\n')
                #                     tmpf.write(content)
                #                 os.replace(tmp_path, csv_path)
                #             except Exception as e:
                #                 print(f"[DT] Failed to normalize CSV header: {e}")
                    
                #     action_idx = ''
                #     action_name = ''
                #     try:
                #         with agent.lock:
                #             pending = agent.pending_tasks.get(unique_task_key)
                #         if pending is not None:
                #             a = pending.get('action', '')
                #             action_idx = int(a) if a != '' and a is not None else ''
                #             valid_acts = pending.get('valid_actions', []) or []
                #             if isinstance(action_idx, int) and 0 <= action_idx < len(valid_acts):
                #                 action_name = str(valid_acts[action_idx])
                #     except Exception as e:
                #         print(f"[DT] Failed to read pending action for {unique_task_key}: {e}")

                #     with open(csv_path, 'a', newline='') as csvfile:
                #         writer = csv.writer(csvfile)
                #         if os.path.getsize(csv_path) == 0:
                #             writer.writerow(['sim_time', 'vid', 'latency_s', 'task_id', 'action_idx', 'action_name'])
                #
                #         writer.writerow([simtime_val, vid, float(raw_latency), task_id_val, action_idx, action_name])
                # except Exception as e:
                #     print(f"[DT] Failed to write metrics CSV: {e}")

                next_state_vec, next_valid_acts, _ = self.rl_parser.parse_state_message(message)

                agent.update_with_reward(
                    reward,
                    task_id=unique_task_key,
                    next_state=next_state_vec,
                    done=False
                )

            except Exception as e:
                print(f"[DT Reward Error] car={vid}: {e}")

    def start(self):
        try:
            while True:
                data, addr = self.sock.recvfrom(65536)
                self.executor.submit(self.handle_message, data, addr)
        except KeyboardInterrupt:
            print("\n[DT Server] ⚠️ Interrupted! Force saving latest model before exit...")
        except Exception as e:
            print(f"\n[DT Server] ❌ Unexpected Error: {e}")
        finally:
            with self.agents_lock:
                agents_copy = dict(self.car_agents)
            if agents_copy:
                print("[DT Server] Forcing final save of super dict...")
                all_weights = {}
                for v, agent in agents_copy.items():
                    w = agent.get_model_weights()
                    if w is not None:
                        all_weights[v] = w
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                torch.save(all_weights, self.save_path)
                print(f"[DT Server] ✅ Final Super dict saved ({len(all_weights)} agents) → {self.save_path}")
            
            self.executor.shutdown(wait=False)
            self.sock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Digital Twin RL Server")
    parser.add_argument("--port", type=int, default=6000,
                        help="UDP port to listen on")
    parser.add_argument("--save_path", type=str,
                        default="/home/shuoer/marl2/scripts/models/dt_best.pth",
                        help="Path to save the best model")
    parser.add_argument("--init_weights", type=str,
                        default="/home/shuoer/marl2/scripts/models/dt_init.pth",
                        help="Path to load DT initial weights")
    parser.add_argument("--plot_path", type=str,
                        default="pic/dt.png",
                        help="Path to save the training plot for this scenario")

    args = parser.parse_args()

    server = DTServer(
        port=args.port,
        save_path=args.save_path,
        init_weights_path=args.init_weights,
        plot_path=args.plot_path,
    )
    server.start()