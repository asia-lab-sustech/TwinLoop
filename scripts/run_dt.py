# scripts/run_dt.py

import json
import math
import os
import random
import select
import socket
import sys
import time
from collections import defaultdict
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import (PurePythonSimEnv, ChannelModel, generate_task, SIM_DEFAULTS)

SIM_DIR        = "/home/shuoer/marl/sumo"
SUMO_CFG       = "dt_heterogeneous.sumocfg"
SUMO_BIN       = "sumo"  
RSUS_XML       = "rsu.add.xml"
SNAPSHOT_PATH  = "snapshot.json"

AGENT_HOST     = "127.0.0.1"
AGENT_PORT     = 6000            
CAR_BASE_PORT  = 7000            
TOTAL_STEPS    = 1500           
STEP_LEN       = 0.5             
ACTION_TIMEOUT = 0.5             

_CH = ChannelModel.from_config(SIM_DEFAULTS.get("channel"))

def calc_uplink_rate(dist_m: float) -> float:
    snr = _CH.snr(max(dist_m, 1.0))
    return _CH.B * math.log2(1.0 + snr) 

class CarSocketPool:
    def __init__(self, base_port=7000):
        self.base_port = base_port
        self._socks: Dict[int, socket.socket] = {}
        self._vid_to_id: Dict[str, int] = {}
        self._next_id = 0
        
    def get_car_id(self, vid: str) -> int:
        if vid not in self._vid_to_id:
            self._vid_to_id[vid] = self._next_id
            self._next_id += 1
        return self._vid_to_id[vid]

    def _get_or_create_sock(self, car_id: int) -> socket.socket:
        if car_id not in self._socks:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = self.base_port + car_id
            try:
                s.bind(("0.0.0.0", port))
                s.setblocking(False)
                self._socks[car_id] = s
            except Exception as e:
                print(f"[SOCK] Failed to bind port {port} for car {car_id}: {e}")
        return self._socks[car_id]

    def send_to_agent(self, car_id: int, msg: dict):
        sock = self._get_or_create_sock(car_id)
        sock.sendto(json.dumps(msg).encode("utf-8"), (AGENT_HOST, AGENT_PORT))

    def recv_action(self, car_id: int) -> Optional[str]:
        sock = self._get_or_create_sock(car_id)
        ready, _, _ = select.select([sock], [], [], ACTION_TIMEOUT) 
        if ready:
            data, _ = sock.recvfrom(1024)
            return data.decode("utf-8").strip()
        return None  

    def close(self):
        for s in self._socks.values():
            try: s.close()
            except: pass

class RLAgentInterface:
    def __init__(self, pool: CarSocketPool, env: PurePythonSimEnv):
        self.pool = pool
        self.env  = env

    def build_state_msg(self, car_id: int, vid: str, obs_v: dict, task: dict) -> dict:
        v = self.env.vehicles.get(vid, {})
        px, py = obs_v["pos"]
        
        servers_json = []
        for s in obs_v["servers"]:
            uplink_bps = calc_uplink_rate(s["dist_m"])
            servers_json.append({
                "name": s["id"], 
                "current_load": s["current_load"], 
                "processing_rate": s["proc_rate"],
                "uplink_rate": uplink_bps
            })

        return {
            "msg_type": "state",
            "car_id":   car_id,
            "vid":      vid, 
            "task_id":  task["taskId"],
            "sim_time": self.env._sim_time,
            "car": {
                "local_load":      obs_v["local_load"],
                "processing_rate": v.get("local_rate", 1e9),
                "speed":           math.sqrt(obs_v["vel"][0]**2 + obs_v["vel"][1]**2),
                "pos_x":           px,
                "pos_y":           py,
            },
            "task_info": {
                "input_size": task["inputDataSize"],
                "demand":     task["demandResource"],
            },
            "servers": servers_json,
        }

    def query_action(self, vid: str, obs_v: dict, task: dict) -> str:
        car_id = self.pool.get_car_id(vid)
        msg = self.build_state_msg(car_id, vid, obs_v, task)
        self.pool.send_to_agent(car_id, msg)

        action_raw = self.pool.recv_action(car_id)
        if action_raw is None:
            servers = obs_v.get("servers", [])
            return random.choice(servers)["id"] if servers else "local"

        try:
            action_idx = int(action_raw)
            if action_idx == 0: return "local"
            servers = obs_v.get("servers", [])
            if 0 < action_idx <= len(servers):
                return servers[action_idx - 1]["id"]
        except ValueError:
            pass
        return "local"

    def send_reward(self, vid: str, task_result: dict, obs_v: dict, task: dict = None):
        car_id = self.pool.get_car_id(vid)
        v = self.env.vehicles.get(vid, {})
        px, py = obs_v["pos"]

        servers_json = []
        for s in obs_v.get("servers", []):
            uplink_bps = calc_uplink_rate(s["dist_m"])
            servers_json.append({
                "name": s["id"], 
                "current_load": s["current_load"], 
                "processing_rate": s["proc_rate"],
                "uplink_rate": uplink_bps
            })
            
        msg = {
            "msg_type": "reward",
            "car_id":   car_id,
            "vid":      vid,        # ⬅️ 確保加入了 vid
            "task_id":  task_result["taskId"],
            "latency":  task_result["total_delay"],
            "sim_time": self.env._sim_time,
            "car": {
                "local_load":      obs_v.get("local_load", 0),
                "processing_rate": v.get("local_rate", 1e9),
                "speed":           math.sqrt(obs_v["vel"][0]**2 + obs_v["vel"][1]**2),
                "pos_x":           px,
                "pos_y":           py,
            },
            "task_info": {
                "input_size": task["inputDataSize"],
                "demand":     task["demandResource"],
            },
            "servers":   servers_json,
        }
        self.pool.send_to_agent(car_id, msg)


def main():
    print("=" * 62)
    print("  Digital Twin (DT) Runner - Hyper-speed Training")
    print("=" * 62)

    _check = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _check.settimeout(0.5)
    _check.sendto(b'{"msg_type":"ping"}', (AGENT_HOST, AGENT_PORT))
    agent_online = True
    try:
        _check.recvfrom(64)
    except socket.timeout:
        pass
    except Exception:
        agent_online = False
    _check.close()
    print(f"  Agent port {AGENT_PORT}: "
          f"{'reachable' if agent_online else 'NOT reachable — will use fallback'}")
    print()

    pool = CarSocketPool(base_port=CAR_BASE_PORT)
    snapshot_path = os.path.join(SIM_DIR, "snapshot.json")
    server_states = None
    car_states    = None

    if os.path.exists(snapshot_path):
        try:
            with open(snapshot_path) as f:
                snap = json.load(f)
            
            server_states = snap.get("servers")
            car_states    = snap.get("cars")
                    
            print(f"[DT Runner] Snapshot loaded: {len(car_states or [])} cars, {len(server_states or [])} servers")
        except Exception as e:
            print(f"[DT Runner] Failed to load snapshot: {e}")

    env = PurePythonSimEnv(
        sumo_cfg      = SUMO_CFG,
        rsus_xml      = RSUS_XML,
        sumo_binary   = SUMO_BIN,
        step_length   = STEP_LEN,
        server_states = server_states,
        car_states    = car_states,
        target_vehicles  =50,
    )
    env.start()
    
    rl = RLAgentInterface(pool, env)
    prev_vehicles = set()  

    total_tasks    = 0
    total_delay    = 0.0
    t0             = time.time()

    try:
        for step_i in range(TOTAL_STEPS):
            env.advance_sumo()
            now = env._sim_time
            
            current_vehicles = set(env.vehicles.keys())
            for vid in (prev_vehicles - current_vehicles):
                car_id = pool.get_car_id(vid)
                pool.send_to_agent(car_id, {"msg_type": "leave", "car_id": car_id, "vid": vid})
            prev_vehicles = current_vehicles
            
            task_results = []
            step_task_count = 0
            
            for vid, v in list(env.vehicles.items()):
                if now < env._next_task_at.get(vid, float('inf')):
                    continue

                env._next_task_at[vid] = now + env._sample_interarrival()
                env._task_counter += 1
                task = generate_task(env._task_counter, vid)

                o = env.get_observations().get(vid)
                if not o: continue

                action = rl.query_action(vid, o, task)
                result = env._execute_offload(task, vid, v, action, now)
                task_results.append((vid, result, o, task))
                step_task_count += 1

                obs_after = env.get_observations()
                o_after = obs_after.get(vid, o)
                rl.send_reward(vid, result, o_after, task=task)

                total_tasks += 1
                total_delay += result["total_delay"]

            if (step_i + 1) % 50 == 0:
                elapsed = time.time() - t0
                speed   = (step_i + 1) / elapsed if elapsed > 0 else 0
                print(f"[DT] Step {step_i+1:4d} | Tasks {total_tasks:4d} | Speed: {speed:.1f} steps/s")

    except KeyboardInterrupt:
        print("\n[DT] Stopped by user.")
    finally:
        env.close()
        pool.close()

    elapsed = time.time() - t0
    print(f"\n{'='*62}")
    print(f"  DT TRAINING FINISHED in {elapsed:.1f}s")
    print(f"{'='*62}")

    print("[DT Runner] tell dt_server to save the model...")
    try:
        save_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        save_sock.sendto(json.dumps({"msg_type": "save_and_sync"}).encode('utf-8'), ("127.0.0.1", 6000))
        save_sock.close()
        time.sleep(3) 
        print("[DT Runner] the model is saved and synced successfully!")
    except Exception as e:
        print(f"[DT Runner] ⚠️ FAIL: {e}")

    print("[DT Runner] Sending 'sync_model' signal to PT server...")
    try:
        sync_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sync_msg = {
            "msg_type": "sync_model",
            "model_path": "/home/shuoer/marl/scripts/models/dt_best.pth"
        }
        sync_sock.sendto(json.dumps(sync_msg).encode('utf-8'), ("127.0.0.1", 5000))
        sync_sock.close()
        print("[DT Runner] Sync signal sent successfully!")
    except Exception as e:
        print(f"[DT Runner] Failed to send sync signal: {e}")

if __name__ == "__main__":
    os.chdir(SIM_DIR)
    main()