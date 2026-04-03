# scripts/pt_server.py

import sys
import os
import random
import socket
import json
import argparse
import threading
import time
import torch
import subprocess
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.agent_factory import AgentFactory
from src.environment.rl_parser import RLParser  
from src.agents.state_manager import StateManager
from src.utils.config_loader import ConfigLoader
import matplotlib
matplotlib.use('Agg')
from plotter import TrainingVisualizer
import csv

PT_INIT_WEIGHTS_PATH = os.path.abspath("/home/shuoer/marl2/scripts/models/pt_init.pth")
DT_INIT_WEIGHTS_PATH = os.path.abspath("/home/shuoer/marl2/scripts/models/dt_init.pth")
PT_CONTROL_PORT = 15000

# ==========================================
# (Physical Twin Server)
# ==========================================
class PTServer:
    def __init__(self, agent_host: str = '127.0.0.1', agent_port: int = 5000, buffer_size: int = 65536,
                 plot_path: str = "pic/pt.png"):
        self.agent_host    = agent_host
        self.agent_port    = agent_port
        self.buffer_size   = buffer_size

        self.config_loader = ConfigLoader()
        self.rl_parser     = RLParser()
        self.state_manager = StateManager(self.rl_parser)

        self.car_agents: dict = {}
        self.agents_lock      = threading.Lock()

        print("[PT Server] Loading agent config template...")
        self._agent_config_template = self.config_loader.load_agent_config("agent_pt")

        self.model_lock = threading.Lock()
        self.dt_lock    = threading.Lock()

        self.global_weights_dict = {}
        if os.path.exists(PT_INIT_WEIGHTS_PATH):
            try:
                # self.global_weights_dict = torch.load(PT_INIT_WEIGHTS_PATH, map_location='cpu')
                self.global_weights_dict = torch.load(PT_INIT_WEIGHTS_PATH, map_location='cpu', weights_only=False)
                print(f"[PT Server] ✅ Loaded global weights dict for {len(self.global_weights_dict)} cars.")
            except Exception as e:
                print(f"[PT Server] ⚠️ Failed to load init weights: {e}")

        # 启动 UDP 服务端
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.agent_host, self.agent_port))

        self.executor   = ThreadPoolExecutor(max_workers=10)
        self.visualizer = TrainingVisualizer(save_path=plot_path, window_size=10)

        # DT 进程与时间管理
        self.dt_running     = False
        self.should_trigger = False
        self.dt_sim_process = None   
        self.exploration    = True  # 始终开启探索

        print(f"[PT Server] 🚀 Listening on {self.agent_host}:{self.agent_port}")

    def _get_or_create_agent(self, vid: str, car_id: int):
        with self.agents_lock:
            if vid not in self.car_agents:
                new_agent = AgentFactory.create_agent(f"car_{vid}", self._agent_config_template)

                if hasattr(new_agent, 'set_loss_callback'):
                    def make_loss_callback(cid):
                        def cb(loss_val): self.visualizer.add_loss(loss_val)
                        return cb
                    new_agent.set_loss_callback(make_loss_callback(car_id))

                if self.global_weights_dict and vid in self.global_weights_dict:
                    new_agent.set_model_weights(self.global_weights_dict[vid], sync_type="sync")
                    if self.car_agents:
                        max_steps = max(a.step_count for a in self.car_agents.values())
                        new_agent.step_count = max_steps
                    print(f"[PT Server] Car {vid} (ID:{car_id}): 🧠 Loaded exclusive memory.")

                elif self.global_weights_dict and self.car_agents:
                    mentor_vid = random.choice(list(self.car_agents.keys()))
                    mentor_weights = self.car_agents[mentor_vid].get_model_weights()
                    new_agent.set_model_weights(mentor_weights, sync_type="sync")
                    new_agent.step_count = self.car_agents[mentor_vid].step_count
                    print(f"[PT Server] Car {vid} (ID:{car_id}): 🧬 Cloned from '{mentor_vid}'.")

                else:
                    new_agent.step_count = 0
                    print(f"[PT Server] Car {vid} (ID:{car_id}): 🐣 Cold start (independent).")

                self.car_agents[vid] = new_agent
            return self.car_agents[vid]

    def _perform_sync_safely(self, path: str):
        if not os.path.exists(path):
            print(f"[PT Server] ❌ Sync file not found: {path}")
            return

        with self.model_lock:
            try:
                print(f"[PT Server] 🔄 DT→PT Soft Sync Started from {path}...")
                # dt_weights = torch.load(path, map_location='cpu')
                # raw = torch.load(path, map_location='cpu')
                raw = torch.load(path, map_location='cpu', weights_only=False)
                print(f"[PT Server] raw type={type(raw)}, keys={list(raw.keys()) if isinstance(raw, dict) else 'NOT A DICT'}")

                if isinstance(raw, dict) and 'weights_dict' in raw:
                    dt_weights = raw['weights_dict']
                else:
                    dt_weights = raw

                synced = 0
                skipped = 0
                with self.agents_lock:
                    print(f"[PT Server] 🧠 PT currently has {len(self.car_agents)} agents: {list(self.car_agents.keys())}")
                
                    for vid, agent in self.car_agents.items():
                        if vid in dt_weights:
                            agent.set_model_weights(dt_weights[vid], sync_type="sync")
                            synced += 1
                        else:
                            skipped += 1
                print(f"[PT Server] ✅ Sync complete: {synced} updated, {skipped} skipped (not in DT weights)")

                proc = getattr(self, 'dt_sim_process', None)
                if proc:
                    try:
                        proc.terminate()
                        proc.wait()
                    except Exception as e:
                        print(f"[PT Server] Process kill error: {e}")
                    self.dt_sim_process = None

                with self.dt_lock:
                    self.dt_running  = False
                    # self.exploration   = False

            except Exception as e:
                print(f"[PT Server] Sync error: {e}")
            finally:
                try:
                    ctrl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    ctrl_msg = json.dumps({"msg_type": "set_time_ratio", "value": 10.0}).encode('utf-8')
                    ctrl.sendto(ctrl_msg, (self.agent_host, PT_CONTROL_PORT))
                    ctrl.close()
                    print(f"[PT Server] Sent control: set_time_ratio=10 -> {self.agent_host}:{PT_CONTROL_PORT}")
                except Exception as e:
                    print(f"[PT Server] Failed to notify PT runner of time-ratio change: {e}")

    def _check_and_trigger_dt(self):
        if not self.dt_running and self.should_trigger:
            with self.dt_lock:
                self.should_trigger = False
                self.dt_running = True
                threading.Thread(target=self.trigger_dt_process, daemon=True).start()

    def trigger_dt_process(self):
        print("[PT Server] ⚡ TRIGGERING DIGITAL TWIN...")
        try:
            try:
                ctrl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                ctrl_msg = json.dumps({"msg_type": "set_time_ratio", "value": 0.1}).encode('utf-8')
                ctrl.sendto(ctrl_msg, (self.agent_host, PT_CONTROL_PORT))
                ctrl.close()
                print(f"[PT Server] Sent control: set_time_ratio=0.1 -> {self.agent_host}:{PT_CONTROL_PORT}")
            except Exception as e:
                print(f"[PT Server] Failed to notify PT runner to slow down: {e}")

            os.makedirs(os.path.dirname(DT_INIT_WEIGHTS_PATH), exist_ok=True)
            with self.agents_lock:
                agents_snapshot = dict(self.car_agents)
            
            if agents_snapshot:
                all_weights = {v: a.get_model_weights() for v, a in agents_snapshot.items()}
                torch.save(all_weights, DT_INIT_WEIGHTS_PATH)
                print(f"[PT Server] 💾 Saved {len(all_weights)} agents for DT initialization.")
        except Exception as e:
            print(f"[PT Server] ⚠️ Failed to save init weights: {e}")

        try:
            # 1. 运行 sim_update.py，生成 DT 所需的 SUMO 环境文件
            sim_update_script = os.path.join(os.path.dirname(__file__), "sim_update.py")
            print("[PT Server] Running sim_update.py to generate DT traffic files...")
            subprocess.run([sys.executable, sim_update_script], check=True)
            print("[PT Server] sim_update complete.")

            # 2. 启动 multisce_run.py，接管 DT 全生命周期
            multisce_script = os.path.join(os.path.dirname(__file__), "multisce_run.py")
            self.dt_sim_process = subprocess.Popen([sys.executable, multisce_script])
            print("[PT Server] Multi-scenario DT training started.")
        except Exception as e:
            print(f"[PT Server] Failed to launch DT: {e}")
            self.dt_running = False

    def handle_message(self, data: bytes, addr):
        try:
            message = json.loads(data.decode('utf-8').strip())
        except Exception as e:
            print(f"Decode error from {addr}: {e}")
            return
            
        msg_type = message.get('msg_type', '').lower()
        if msg_type == 'trigger_dt':
            if not self.dt_running:
                sim_time = message.get("sim_time", "Unknown")
                print(f"\n[PT Server] 📥 Received explicit trigger signal from PT (Time: {sim_time}s). Starting DT...")
                os.system("lsof -ti :6000,7000 | xargs -r kill -9 2>/dev/null")
                
                with self.dt_lock:
                    self.should_trigger = True
                self._check_and_trigger_dt()
            return

        # 模型同步信号处理
        if msg_type == 'sync_model':
            print(f"[PT Server] 📥 Received Sync signal from DT.")
            path = message.get('model_path')
            self._perform_sync_safely(path)
            return

        car_id = message.get('car_id')
        if car_id is None: return
        car_id = int(car_id)
        
        vid = message.get('vid', f"unknown_{car_id}")
        task_id = message.get('task_id', -1)
        unique_task_key = f"{vid}_{task_id}"

        # 处理离开消息，清理 Agent 和悬挂任务
        if msg_type == 'leave':
            with self.agents_lock:
                if vid in self.car_agents:
                    agent = self.car_agents[vid]
                    with agent.lock:
                        stale = [k for k in agent.pending_tasks if str(k).startswith(f"{vid}_")]
                        for k in stale:
                            del agent.pending_tasks[k]
                    del self.car_agents[vid]
                    print(f"[Server] Car {vid} departed, agent removed.")
            return

        agent = self._get_or_create_agent(vid, car_id)

        if msg_type == 'state':
            try:
                state_vec, valid_acts, _ = self.rl_parser.parse_state_message(message)
                
                action_idx, action_name = agent.select_action(
                    state_vec, valid_acts,
                    exploration=self.exploration,
                    task_id=unique_task_key
                )
                
                self.sock.sendto(str(action_idx).encode('utf-8'), addr)

            except Exception as e:
                print(f"[state error] car={vid}: {e}")

        elif msg_type == 'reward':
            try:
                reward = self.rl_parser.parse_reward_message(message)
                raw_latency = float(message.get('latency', 0.0))
                self.visualizer.add_latency(raw_latency)

                # --- 记录每个 task 的最小信息到 CSV ---
                simtime = message.get('sim_time', message.get('task_id', None))
                try: simtime_val = float(simtime)
                except Exception: simtime_val = str(simtime)
                
                task_id_val = message.get('task_id', None)
                try: task_id_val = int(task_id_val)
                except Exception: task_id_val = str(task_id_val)
                
                try:
                    scripts_dir = os.path.dirname(os.path.abspath(__file__))
                    pic_dir = os.path.join(scripts_dir, 'pic')
                    os.makedirs(pic_dir, exist_ok=True)
                    csv_path = os.path.join(pic_dir, 'metrics_tasks.csv')
                except Exception:
                    csv_path = os.path.join('.', 'metrics_tasks.csv')

                header = 'sim_time,vid,latency_s,task_id,action_idx,action_name'
                try:
                    if os.path.exists(csv_path):
                        try:
                            with open(csv_path, 'r', newline='') as f:
                                first = f.readline().strip()
                        except Exception:
                            first = ''
                        if first != header:
                            try:
                                with open(csv_path, 'r', newline='') as f:
                                    content = f.read()
                            except Exception:
                                content = ''
                            tmp_path = csv_path + '.tmp'
                            try:
                                with open(tmp_path, 'w', newline='') as tmpf:
                                    tmpf.write(header + '\n')
                                    tmpf.write(content)
                                os.replace(tmp_path, csv_path)
                            except Exception as e:
                                print(f"[PT] Failed to normalize CSV header: {e}")
                    
                    action_idx = ''
                    action_name = ''
                    try:
                        with agent.lock:
                            pending = agent.pending_tasks.get(unique_task_key)
                        if pending is not None:
                            a = pending.get('action', '')
                            action_idx = int(a) if a != '' and a is not None else ''
                            valid_acts = pending.get('valid_actions', []) or []
                            if isinstance(action_idx, int) and 0 <= action_idx < len(valid_acts):
                                action_name = str(valid_acts[action_idx])
                    except Exception as e:
                        print(f"[PT] Failed to read pending action for {unique_task_key}: {e}")

                    with open(csv_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if os.path.getsize(csv_path) == 0:
                            writer.writerow(['sim_time', 'vid', 'latency_s', 'task_id', 'action_idx', 'action_name'])
                        writer.writerow([simtime_val, vid, float(raw_latency), task_id_val, action_idx, action_name])
                except Exception as e:
                    print(f"[PT] Failed to write metrics CSV: {e}")

                next_state_vec, next_valid_acts, _ = self.rl_parser.parse_state_message(message)

                agent.update_with_reward(
                    reward,
                    task_id=unique_task_key,
                    next_state=next_state_vec,
                    done=False
                )
                

            except Exception as e:
                print(f"[reward error] car={vid}: {e}")

    def start(self):
        print("[PT Server] System Online. Waiting for PT Environment...")
        try:
            while True:
                data, addr = self.sock.recvfrom(self.buffer_size)
                self.executor.submit(self.handle_message, data, addr)
        except KeyboardInterrupt:
            print("Shutting down")
            self.executor.shutdown(wait=False)
        finally:
            self.sock.close()


def main():
    parser = argparse.ArgumentParser(description="PT Server - Digital Twin Physical Node")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--plot_path", type=str, default="pic/pt.png",
                        help="训练曲线保存路径")
    args = parser.parse_args()

    print("[PT Server] Starting PT Server")
    server = PTServer(
        agent_port=args.port,
        plot_path=args.plot_path,
    )
    server.start()

if __name__ == '__main__':
    main()