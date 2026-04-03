# scripts/run_pt.py

import json
import math
import os
import random
import select
import socket
import sys
import time
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import (PurePythonSimEnv, ChannelModel, SIM_DEFAULTS)
from pt_scenario_scheduler import (
    ScenarioScheduler, make_default_scenario,
    generate_task_dynamic, sample_interarrival_dynamic,
)
# ═══════════════════════════════════════════════════════
# PT 全局配置
# ═══════════════════════════════════════════════════════
SIM_DIR        = "/home/shuoer/marl2/sumo"
SUMO_CFG       = "city_simulation.sumocfg"
SUMO_BIN       = "sumo" 
RSUS_XML       = "rsus.add.xml"
SNAPSHOT_PATH  = "snapshot.json"

AGENT_HOST     = "127.0.0.1"
AGENT_PORT     = 5000  
CAR_BASE_PORT  = 4000 

TOTAL_STEPS    = 7000  
STEP_LEN       = 0.5              # the step length of sumo simulation
REAL_TIME_FACTOR = 15              # the simulation runs at REAL_TIME_FACTOR times real speed (e.g., 15 means 15x faster than real time)
ENABLE_DT      = True            # 纯PT实验时设为False，完整系统运行时设为True

CONTROL_PORT = 15000
# trigger the DT at specific steps (for evaluation and comparison purposes)
# DT_TRIGGER_STEPS = [1050,1500,2000,2500,3050,3500,4000,4500,5050,5500,6000,6500]
# DT_TRIGGER_STEPS = [1050,1666,2332,3050,3666,4332,5050,5666,6332]
# DT_TRIGGER_STEPS = [1050,2000,3050,4000,5050,6000]
DT_TRIGGER_STEPS = [1050,3050,5050]
# ═══════════════════════════════════════════════════════

_CH = ChannelModel.from_config(SIM_DEFAULTS.get("channel"))

def calc_uplink_rate(dist_m: float) -> float:
    snr = _CH.snr(max(dist_m, 1.0))
    theoretical = _CH.B * math.log2(1.0 + snr) 
    return theoretical

# ═══════════════════════════════════════════════════════
# 快照导出器
# ═══════════════════════════════════════════════════════
def export_snapshot(env: PurePythonSimEnv, filepath: str, scheduler=None):
    """
    write the current absolute physical state of the environment (vehicles, servers) into a JSON file.
    """
    snapshot = {
        "time": env._sim_time,
        "servers": [],
        "cars": []
    }
    
    for sid, srv in env.servers.items():
        snapshot["servers"].append({
            "id": srv.server_id,
            "px": srv.px,
            "py": srv.py,
            "processingRate": srv.processing_rate,
            "currentLoad": srv.current_load
        })
        
    for vid, v in env.vehicles.items():
        snapshot["cars"].append({
            "id": vid,
            "px": v["px"],
            "py": v["py"],
            "vx": v.get("vx", 0.0),
            "vy": v.get("vy", 0.0),
            "processingRate": v["local_rate"],
            "currentLoad": env._get_local_load(vid, env._sim_time)
        })
        
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        try:
            if scheduler is not None:
                w = getattr(scheduler, 'current_task_weights', None)
                if w is not None:
                    snapshot['task_type_weights'] = w
                arrival_mean = getattr(scheduler, 'current_arrival_mean', None)
                if arrival_mean is not None:
                    snapshot['arrival_mean'] = float(arrival_mean)
        except Exception:
            pass
        json.dump(snapshot, f, indent=2)
    os.replace(tmp_path, filepath)


class CarSocketPool:
    def __init__(self, base_port=4000):
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
                print(f"[PT-SOCK] Bind failed: {e}")
        return self._socks[car_id]

    def send_to_agent(self, car_id: int, msg: dict):
        sock = self._get_or_create_sock(car_id)
        sock.sendto(json.dumps(msg).encode("utf-8"), (AGENT_HOST, AGENT_PORT))

    def recv_action(self, car_id: int) -> str:
        sock = self._get_or_create_sock(car_id)
        ready, _, _ = select.select([sock], [], []) 
        if ready:
            data, _ = sock.recvfrom(1024)
            return data.decode("utf-8").strip()
        return "local"

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
        # servers = obs_v.get("servers", [])
        # a = 1/21
        # if random.random() > a:
        #   return random.choice(servers)["id"]
        # else:
        #   return "local"
        car_id = self.pool.get_car_id(vid)
        msg = self.build_state_msg(car_id, vid, obs_v, task)
        self.pool.send_to_agent(car_id, msg)

        action_raw = self.pool.recv_action(car_id)
        
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
            "vid":      vid, 
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


# ═══════════════════════════════════════════════════════
# physical twin loop
# ═══════════════════════════════════════════════════════
def main():
    print("=" * 62)
    print("  Physical Twin (PT) Runner - Replacing OMNeT++")
    print("=" * 62)
    global REAL_TIME_FACTOR
    
    pool = CarSocketPool(base_port=CAR_BASE_PORT)
    env = PurePythonSimEnv(sumo_cfg=SUMO_CFG, rsus_xml=RSUS_XML, sumo_binary=SUMO_BIN, step_length=STEP_LEN)
    env.start()
    original_server_rates = {sid: srv.processing_rate 
                             for sid, srv in env.servers.items()}
    scheduler = ScenarioScheduler(
        phases=make_default_scenario(),
        original_server_rates=original_server_rates,
        original_arrival_mean=float(SIM_DEFAULTS.get("arrival", {}).get("mean", 10.0)),
        seed=42,
    )
    rl = RLAgentInterface(pool, env)

    control_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        control_sock.bind(("0.0.0.0", CONTROL_PORT))
        control_sock.setblocking(False)
        print(f"[PT] Control socket listening on 0.0.0.0:{CONTROL_PORT}")
    except Exception as e:
        print(f"[PT] Warning: failed to bind control socket {CONTROL_PORT}: {e}")
        control_sock = None
    
    
    prev_vehicles = set()  
    total_tasks = 0
    total_delay = 0.0
    
    triggered_steps = set()
    dt_wait_start_sim_time = None
    
    phase_stats: Dict[str, Dict] = {}   # {phase_name: {tasks, delay_sum}}
    try:
        for step_i in range(TOTAL_STEPS):
            if control_sock:
                try:
                    ready, _, _ = select.select([control_sock], [], [], 0)
                    if ready:
                        data, _ = control_sock.recvfrom(4096)
                        try:
                            msg = json.loads(data.decode('utf-8'))
                            if msg.get('msg_type') == 'set_time_ratio':
                                val = msg.get('value')
                                try:
                                    new_ratio = float(val)
                                    REAL_TIME_FACTOR = new_ratio
                                    print(f"[PT] Control: set REAL_TIME_FACTOR -> {REAL_TIME_FACTOR}")
                                    # if PT receives a recovery signal with ratio > 1, it means DT has completed and notified PT,
                                    try:
                                        if 'env' in locals() and getattr(env, '_dt_waiting', False) and new_ratio > 1.0:
                                            env._dt_waiting = False
                                            if dt_wait_start_sim_time is not None:
                                                dt_wait_delta = env._sim_time - dt_wait_start_sim_time
                                                print(f"[PT] DT finished. PT sim_time advanced by {dt_wait_delta:.1f}s during DT window.")
                                                dt_wait_start_sim_time = None
                                            print("[PT] Control: DT finished, cleared env._dt_waiting")
                                    except Exception:
                                        pass
                                except Exception:
                                    print(f"[PT] Control: invalid value for set_time_ratio: {val}")
                        except Exception as e:
                            print(f"[PT] Control: failed to parse control message: {e}")
                except Exception:
                    pass

            step_wall_start = time.time() 

            env.advance_sumo()
            now = env._sim_time
            scheduler.apply(env, now)

            phase = scheduler.get_current_phase(now)
            pname = phase.name if phase else "unknown"

            if ENABLE_DT and DT_TRIGGER_STEPS:
                try:
                    if step_i in DT_TRIGGER_STEPS and step_i not in triggered_steps and not getattr(env, '_dt_waiting', False):
                        
                        try:
                            export_snapshot(env, SNAPSHOT_PATH, scheduler)
                        except Exception as e:
                            print(f"[PT] Warning: failed to export snapshot before scheduled trigger at step {step_i}: {e}")

                        try:
                            trigger_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                            trigger_msg = {"msg_type": "trigger_dt", "sim_time": now}
                            trigger_sock.sendto(json.dumps(trigger_msg).encode('utf-8'), (AGENT_HOST, AGENT_PORT))
                            trigger_sock.close()
                            print(f"[PT] t={now:.0f}s scheduled DT triggered at step {step_i}")
                        except Exception as e:
                            print(f"[PT] Failed to send scheduled trigger signal at step {step_i}: {e}")

                        try:
                            REAL_TIME_FACTOR = 0.05
                            print(f"[PT] Local control: REAL_TIME_FACTOR set to {REAL_TIME_FACTOR} (slowing for DT)")
                        except Exception:
                            pass

                        env._dt_waiting = True
                        dt_wait_start_sim_time = now
                        last_dt_trigger_time = now
                        triggered_steps.add(step_i)
                except Exception:
                    pass
            
            if pname not in phase_stats:
                phase_stats[pname] = {"tasks": 0, "delay_sum": 0.0}
            phase = scheduler.get_current_phase(env._sim_time)
                
            obs = env.get_observations()
            
            current_vehicles = set(env.vehicles.keys())
            for vid in (prev_vehicles - current_vehicles):
                car_id = pool.get_car_id(vid)
                pool.send_to_agent(car_id, {
                    "msg_type": "leave", 
                    "car_id": car_id,
                    "vid": vid
                })
            prev_vehicles = current_vehicles
            
            task_results = []
            for vid, v in list(env.vehicles.items()):
                if now < env._next_task_at.get(vid, float('inf')):
                    continue

                env._next_task_at[vid] = now + sample_interarrival_dynamic(scheduler)
                env._task_counter += 1
                task = generate_task_dynamic(env._task_counter, vid, scheduler, SIM_DEFAULTS)

                o = env.get_observations().get(vid)
                if not o: continue

                action = rl.query_action(vid, o, task)
                result = env._execute_offload(task, vid, v, action, now)
                task_results.append((vid, result, o, task))

                obs_after = env.get_observations()
                o_after = obs_after.get(vid, 0)
                rl.send_reward(vid, result, o_after, task=task)
                total_tasks += 1
                delay = result.get("total_delay", 0.0)
                total_delay += delay
                phase_stats[pname]["tasks"]     += 1
                phase_stats[pname]["delay_sum"] += delay

            compute_time = time.time() - step_wall_start
            target_time = STEP_LEN / REAL_TIME_FACTOR
            sleep_time = target_time - compute_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            if (step_i + 1) % 20 == 0:
                avg_ms = (total_delay / max(1, total_tasks)) * 1000
                print(f"[PT] Step {step_i+1:5d} | t={now:6.1f}s | phase={pname:22s} | "
                      f"Cars={len(obs):3d} | AvgDelay={avg_ms:.1f}ms")
                avg_delay = total_delay / max(1, total_tasks)
            
    except KeyboardInterrupt:
        print("\n[PT] Stopped by user.")
    finally:
        env.close()
        pool.close()
        scheduler.close()

    print("\n" + "=" * 62)
    print("  Per-Phase Latency Summary ")
    print("=" * 62)
    print(f"  {'Phase':<25} {'Tasks':>7} {'Avg Latency':>14}")
    print(f"  {'-'*25} {'-'*7} {'-'*14}")
    for pname, stat in phase_stats.items():
        n = stat["tasks"]
        avg_ms = (stat["delay_sum"] / max(1, n)) * 1000
        print(f"  {pname:<25} {n:>7d} {avg_ms:>12.1f}ms")
    print(f"  {'TOTAL':<25} {total_tasks:>7d} {(total_delay/max(1,total_tasks))*1000:>12.1f}ms")
    print("=" * 62)

if __name__ == "__main__":
    os.chdir(SIM_DIR)
    main()