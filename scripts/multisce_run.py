# scripts/multisce_run.py

import json
import math
import os
import random
import select
import socket
import subprocess
import sys
import time
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multisce_sim import (
    PurePythonSimEnv, ChannelModel, SIM_DEFAULTS,
    ScenarioConfig, TRAIN_SCENARIOS,
)

# ═══════════════════════════════════════════════════════
# Global config
# ═══════════════════════════════════════════════════════
SIM_DIR              = "/home/shuoer/marl2/sumo"
SUMO_CFG             = "dt_heterogeneous.sumocfg"
RSUS_XML             = "rsu.add.xml"
SUMO_BIN             = "sumo"
AGENT_HOST           = "127.0.0.1"
AGENT_PORT           = 6000
CAR_BASE_PORT        = 7000
STEP_LEN             = 0.5
ACTION_TIMEOUT       = 0.5
PT_AGENT_PORT        = 5000

DT_SERVER_SCRIPT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dt_server.py")
DT_INIT_WEIGHTS_PATH = "/home/shuoer/marl2/scripts/models/dt_init.pth"
DT_MODELS_DIR        = "/home/shuoer/marl2/scripts/models"
DT_PLOTS_DIR         = "/home/shuoer/marl2/scripts/pic"

DT_USE_SNAPSHOT_LOADS = True

_CH = ChannelModel.from_config(SIM_DEFAULTS.get("channel"))


def calc_uplink_rate(dist_m: float) -> float:
    snr = _CH.snr(max(dist_m, 1.0))
    return _CH.B * math.log2(1.0 + snr)


# ═══════════════════════════════════════════════════════
# Socket pool
# ═══════════════════════════════════════════════════════
class CarSocketPool:
    def __init__(self, base_port: int = CAR_BASE_PORT):
        self.base_port    = base_port
        self._socks: Dict[int, socket.socket] = {}
        self._vid_to_id: Dict[str, int]       = {}
        self._next_id = 0

    def get_car_id(self, vid: str) -> int:
        if vid not in self._vid_to_id:
            self._vid_to_id[vid] = self._next_id
            self._next_id += 1
        return self._vid_to_id[vid]

    def reset_mappings(self):
        """Clear vid→id mappings on scenario switch; reuse bound sockets."""
        self._vid_to_id.clear()
        self._next_id = 0

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
                print(f"[SOCK] Failed to bind port {port}: {e}")
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
            try:
                s.close()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════
# RL interface
# ═══════════════════════════════════════════════════════
class RLAgentInterface:
    def __init__(self, pool: CarSocketPool, env: PurePythonSimEnv):
        self.pool = pool
        self.env  = env

    def build_state_msg(self, car_id: int, vid: str, obs_v: dict, task: dict) -> dict:
        v      = self.env.vehicles.get(vid, {})
        px, py = obs_v["pos"]

        servers_json = []
        for s in obs_v["servers"]:
            servers_json.append({
                "name":            s["id"],
                "current_load":    s["current_load"],
                "processing_rate": s["proc_rate"],
                "uplink_rate":     calc_uplink_rate(s["dist_m"]),
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

    def query_action(self, vid: str, obs_v: dict, task: dict) -> Optional[str]:
        car_id    = self.pool.get_car_id(vid)
        msg       = self.build_state_msg(car_id, vid, obs_v, task)
        self.pool.send_to_agent(car_id, msg)

        action_raw = self.pool.recv_action(car_id)
        if action_raw is None:
            servers = obs_v.get("servers", [])
            return random.choice(servers)["id"] if servers else "local"

        try:
            action_idx = int(action_raw)
            if action_idx == 0:
                return "local"
            servers = obs_v.get("servers", [])
            if 0 < action_idx <= len(servers):
                return servers[action_idx - 1]["id"]
        except ValueError:
            pass
        return "local"

    def send_reward(self, vid: str, task_result: dict, obs_v: dict, task: dict = None):
        car_id = self.pool.get_car_id(vid)
        v      = self.env.vehicles.get(vid, {})
        px, py = obs_v["pos"]

        servers_json = []
        for s in obs_v.get("servers", []):
            servers_json.append({
                "name":            s["id"],
                "current_load":    s["current_load"],
                "processing_rate": s["proc_rate"],
                "uplink_rate":     calc_uplink_rate(s["dist_m"]),
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
            "servers": servers_json,
        }
        self.pool.send_to_agent(car_id, msg)


# ═══════════════════════════════════════════════════════
# Dynamic task generation
# ═══════════════════════════════════════════════════════
def generate_dynamic_task(
    task_id: int,
    src_id: str,
    progress: float,
    scenario: ScenarioConfig,
) -> dict:
    templates = SIM_DEFAULTS.get("task_templates")
    weights   = scenario.task_type_weights or SIM_DEFAULTS.get("task_template_weights")
    if not weights or len(weights) != len(templates):
        weights = [1] * len(templates)

    idx = random.choices(range(len(templates)), weights=weights, k=1)[0]
    t   = templates[idx]

    load_multiplier = scenario.task_load_trend
    data_multiplier = scenario.data_size_trend
    demand = max(1e5, random.gauss(
        float(t.get("demand_mean", 1e6)),
        float(t.get("demand_std",  1e5)),
    )) * load_multiplier
    data = max(1e3, random.gauss(
        float(t.get("data_mean", 1e4)),
        float(t.get("data_std",  1e3)),
    )) * data_multiplier

    return {
        "taskId":         task_id,
        "srcAddr":        src_id,
        "demandResource": float(demand),
        "inputDataSize":  float(data),
    }


# ═══════════════════════════════════════════════════════
# dt_server lifecycle
# ═══════════════════════════════════════════════════════
def start_dt_server(scenario_name: str, init_weights_path: str) -> subprocess.Popen:
    """Start a dedicated dt_server for the current scenario."""
    save_path = os.path.join(DT_MODELS_DIR, f"dt_{scenario_name}.pth")
    plot_path = os.path.join(DT_PLOTS_DIR,  f"dt_{scenario_name}.png")
    os.makedirs(DT_PLOTS_DIR, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    proc = subprocess.Popen([
        sys.executable, DT_SERVER_SCRIPT,
        "--port",         str(AGENT_PORT),
        "--init_weights", init_weights_path,
        "--save_path",    save_path,
        "--plot_path",    plot_path,
    ], env=env)
    time.sleep(1.5)
    print(f"[DT] dt_server started for '{scenario_name}'"
          f"\n     from  → {init_weights_path}"
          f"\n     model → {save_path}"
          f"\n     plot  → {plot_path}")
    return proc


def stop_dt_server(proc: Optional[subprocess.Popen]):
    """Gracefully stop dt_server and release the port."""
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    os.system(f"lsof -ti :{AGENT_PORT} | xargs -r kill -9 2>/dev/null")
    time.sleep(0.3)


# ═══════════════════════════════════════════════════════
# Main flow
# ═══════════════════════════════════════════════════════
def main():
    print("=" * 62)
    print("  Multi-Scenario DT Training  (Pure Python + SUMO)")
    print("=" * 62)

    os.makedirs(DT_MODELS_DIR, exist_ok=True)
    pool = CarSocketPool(base_port=CAR_BASE_PORT)

    # ── Load snapshot once, shared by all scenarios ─────────
    snapshot_path     = os.path.join(SIM_DIR, "snapshot.json")
    raw_server_states = None
    raw_car_states    = None

    if os.path.exists(snapshot_path):
        try:
            with open(snapshot_path) as f:
                snap = json.load(f)
            
            raw_server_states = snap.get("servers")
            raw_car_states    = snap.get("cars")
            if 'task_type_weights' in snap:
                try:
                    weights = snap.get('task_type_weights')
                    print(f"[DT Runner] Overriding scenario task_type_weights with snapshot weights: {weights}")
                    for s in TRAIN_SCENARIOS:
                        s.task_type_weights = weights
                except Exception as e:
                    print(f"[DT Runner] Failed to apply snapshot task weights: {e}")

            raw_arrival_mean = snap.get("arrival_mean", None)
            if raw_arrival_mean is not None:
                print(f"[DT Runner] Snapshot arrival_mean = {raw_arrival_mean:.2f}s")
                    
            print(
                f"[DT Runner] Snapshot loaded: {len(raw_car_states or [])} cars, "
                f"{len(raw_server_states or [])} servers"
            )
        except Exception as e:
            print(f"[DT Runner] Failed to load snapshot: {e}")

    all_scenario_stats: Dict[str, float] = {}
    dt_server_proc: Optional[subprocess.Popen] = None

    current_init_weights = DT_INIT_WEIGHTS_PATH
    last_saved_model = None

    try:
        for sce_idx, scenario in enumerate(TRAIN_SCENARIOS):
            print(f"\n{'='*20} SCENARIO {sce_idx+1}/{len(TRAIN_SCENARIOS)}: "
                  f"{scenario.name} {'='*20}")
            print(f"  {scenario.description}")

            # ── 1. Start a fresh dt_server for this scenario ─────
            stop_dt_server(dt_server_proc)
            dt_server_proc = start_dt_server(scenario.name, current_init_weights)

            # ── 2. Initialize env from the shared snapshot ──
            pool.reset_mappings()
            
            server_init_states = None if raw_server_states is None else [dict(s) for s in raw_server_states]
            car_init_states    = None if raw_car_states is None else [dict(c) for c in raw_car_states]
            if not DT_USE_SNAPSHOT_LOADS:
                if server_init_states:
                    for s in server_init_states:
                        if 'currentLoad' in s:
                            s['currentLoad'] = 0.0
                if car_init_states:
                    for c in car_init_states:
                        if 'currentLoad' in c:
                            c['currentLoad'] = 0.0

            env = PurePythonSimEnv(
                sumo_cfg      = SUMO_CFG,
                rsus_xml      = RSUS_XML,
                sumo_binary   = SUMO_BIN,
                step_length   = STEP_LEN,
                scenario      = scenario,
                server_states = server_init_states,
                car_states    = car_init_states,
                arrival_mean  = raw_arrival_mean,
            )
            env.start()
            rl = RLAgentInterface(pool, env)

            n_steps        = scenario.n_episodes
            total_tasks    = 0
            total_delay    = 0.0
            prev_vehicles: set = set()

            # ── 3. Scenario loop ──────────────────────────────────────
            for step_i in range(n_steps):
                ramped_progress = 1.0

                env.advance_sumo(progress=ramped_progress)
                now = env._sim_time

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

                    env._next_task_at[vid] = now + env._sample_interarrival()
                    env._task_counter += 1

                    task = generate_dynamic_task(
                        task_id  = env._task_counter,
                        src_id   = vid,
                        progress = ramped_progress,
                        scenario = scenario,
                    )

                    o = env.get_observations().get(vid)
                    if o is None:
                        continue

                    action = rl.query_action(vid, o, task)
                    result = env._execute_offload(task, vid, v, action, now)
                    task_results.append((vid, result, o, task))
                    
                    obs_after = env.get_observations()
                    o_after = obs_after.get(vid, o)
                    rl.send_reward(vid, result, o_after, task=task)
                    total_tasks += 1
                    total_delay += result["total_delay"]

                if (step_i + 1) % 100 == 0:
                    avg_d = (total_delay / max(1, total_tasks)) * 1000
                    print(f"  [{scenario.name}] Step {step_i+1:4d}/{n_steps} | "
                          f"Tasks: {total_tasks} | "
                          f"Delay: {avg_d:.1f}ms | "
                          f"Vehicles: {len(env.vehicles)}")

            # ── 4. Scenario end: save and close ─────
            try:
                sce_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sce_sock.sendto(
                    json.dumps({"msg_type": "save_scenario",
                                "scenario": scenario.name}).encode(),
                    ("127.0.0.1", AGENT_PORT)
                )
                sce_sock.close()
                time.sleep(10.0)
            except Exception as e:
                print(f"[DT] save_scenario failed: {e}")

            env.close()

            last_saved_model = os.path.join(DT_MODELS_DIR, f"dt_{scenario.name}.pth")
            current_init_weights = last_saved_model

            avg_delay_ms = (total_delay / max(1, total_tasks)) * 1000
            all_scenario_stats[scenario.name] = avg_delay_ms
            print(f"[DONE] {scenario.name} | "
                  f"Avg Delay: {avg_delay_ms:.2f}ms | Tasks: {total_tasks}")

    except KeyboardInterrupt:
        print("\n[DT] Interrupted by user.")
    finally:
        stop_dt_server(dt_server_proc)
        pool.close()

    # ── 5. All scenarios complete: sync final model to PT server ──
    print(f"\n{'='*62}")
    print("  ALL SCENARIOS COMPLETE — syncing FINAL model to PT server")
    print(f"{'='*62}")
    for name, d_ms in all_scenario_stats.items():
        print(f"  - {name:25s}: {d_ms:.2f}ms")

    if all_scenario_stats and last_saved_model:
        try:
            sync_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sync_msg  = {
                "msg_type": "sync_model",
                "model_path": last_saved_model 
            }
            sync_sock.sendto(json.dumps(sync_msg).encode(), ("127.0.0.1", PT_AGENT_PORT))
            sync_sock.close()
            print("[DT] Sync signal sent to PT server.")
        except Exception as e:
            print(f"[DT] Sync signal failed: {e}")


if __name__ == "__main__":
    os.chdir(SIM_DIR)
    main()