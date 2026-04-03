# scripts/multisce_sim.py

import math
import random
import os
import yaml
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import traci
    HAS_TRACI = True
except ImportError:
    HAS_TRACI = False
    print("[WARN] traci not found")


# ============================================================
# Channel model (same as sim.py)
# ============================================================
class ChannelModel:
    def __init__(
        self,
        tx_power_dbm:    float = 30.0,
        noise_power_w:   float = 1e-13,
        path_loss_alpha: float = 4.0,
        bandwidth_hz:    float = 1e6,
        ref_dist_m:      float = 1.0,
    ):
        self.tx_power_w = 10 ** ((tx_power_dbm - 30) / 10)
        self.noise_w    = noise_power_w
        self.alpha      = path_loss_alpha
        self.B          = bandwidth_hz
        self.d0    = ref_dist_m

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None) -> 'ChannelModel':
        if not cfg:
            return cls()
        return cls(
            tx_power_dbm     = float(cfg.get('tx_power_dbm')),
            noise_power_w    = float(cfg.get('noise_power_w')),
            path_loss_alpha  = float(cfg.get('path_loss_alpha')),
            bandwidth_hz     = float(cfg.get('bandwidth_hz')),
            ref_dist_m       = float(cfg.get('ref_dist_m')),
        )

    def snr(self, dist_m: float) -> float:
        d = max(dist_m, self.d0)
        return (self.tx_power_w * (d ** (-self.alpha))) / self.noise_w

    def capacity_bps(self, dist_m: float) -> float:
        return self.B * math.log2(1.0 + self.snr(dist_m))

    def upload_time_s(self, data_bits: float, dist_m: float) -> float:
        cap = self.capacity_bps(dist_m)
        return data_bits / cap if cap > 0 else float('inf')


# ============================================================
# Load YAML config (same as sim.py)
# ============================================================
SIM_DEFAULTS = {}
try:
    cfg_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'configs', 'sim_defaults.yaml'))
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            loaded = yaml.safe_load(f) or {}
            SIM_DEFAULTS.update(loaded)
            if 'server_rates' in SIM_DEFAULTS:
                SIM_DEFAULTS['server_rates'] = [float(x) for x in SIM_DEFAULTS['server_rates']]
            if 'car_local_rate_range' in SIM_DEFAULTS:
                rng = SIM_DEFAULTS['car_local_rate_range']
                SIM_DEFAULTS['car_local_rate_range'] = (float(rng[0]), float(rng[1]))
except Exception as e:
    print(f"[WARN] failed to load sim_defaults.yaml: {e}. Using built-in defaults.")


@dataclass
class EdgeServer:
    server_id:       str
    px:              float
    py:              float
    processing_rate: float
    current_load:    float = field(default=0.0)
    _last_update:    float = field(default=0.0, repr=False)

    def update_load(self, now: float):
        dt = now - self._last_update
        if dt > 0:
            self.current_load = max(0.0, self.current_load - dt * self.processing_rate)
        self._last_update = now

    def submit_task(self, demand_cycles: float, now: float) -> float:
        self.update_load(now)
        proc_delay = (self.current_load + demand_cycles) / self.processing_rate
        self.current_load += demand_cycles
        return proc_delay


# ============================================================
# Load servers from RSU XML
# ============================================================
def load_servers_from_xml(xml_path: str) -> Dict[str, EdgeServer]:
    servers = {}
    if not os.path.exists(xml_path):
        print(f"[ERROR] RSU XML not found at {xml_path}")
        return servers

    rates = SIM_DEFAULTS.get("server_rates", [])
    tree  = ET.parse(xml_path)
    root  = tree.getroot()

    for i, poi in enumerate(root.findall('poi')):
        clean_id = f"server{i}"
        px = float(poi.get('x', 0.0))
        py = float(poi.get('y', 0.0))
        rate = rates[i] if i < len(rates) else 2.0e9
        servers[clean_id] = EdgeServer(
            server_id       = clean_id,
            px              = px,
            py              = py,
            processing_rate = rate,
            current_load    = 0.0,
        )
    return servers

def generate_task(task_id: int, src_id: str) -> dict:
    templates = SIM_DEFAULTS.get("task_templates")
    weights   = SIM_DEFAULTS.get("task_template_weights")
    if not weights or len(weights) != len(templates):
        weights = [1] * len(templates)

    idx = random.choices(range(len(templates)), weights=weights, k=1)[0]
    t   = templates[idx]

    demand = max(1e5, random.gauss(float(t.get("demand_mean", 1e6)),
                                   float(t.get("demand_std",  1e5))))
    data   = max(1e3, random.gauss(float(t.get("data_mean",   1e4)),
                                   float(t.get("data_std",    1e3))))
    return {
        "taskId":         task_id,
        "srcAddr":        src_id,
        "demandResource": float(demand),
        "inputDataSize":  float(data),
    }


# ============================================================
# Scenario config
# ============================================================
@dataclass
class ScenarioConfig:
    name:            str

    # Trend factors (1.0=constant, 1.5=+50% at end, 0.5=half at end)
    veh_count_trend: float = 1.0
    data_size_trend: float = 1.0
    speed_trend:     float = 1.0
    task_load_trend: float = 1.0
    server_rate_trend: float = 1.0

    n_episodes:      int   = 3000
    description:     str   = ""

    # Task type weights override (None = use YAML default)
    task_type_weights: Optional[List[float]] = None

    # Runtime events (-1/negative means disabled)
    server_failure_idx: int   = -1
    failure_time:       float = 9999.0
    burst_start_time:   float = -1.0
    burst_multiplier:   float = 1.0


TRAIN_SCENARIOS = [
    # ScenarioConfig(
    #     name="dt_slight_traffic",
    #     speed_trend=0.95,
    #     task_load_trend=1.05,
    #     n_episodes=1000,
    #     task_type_weights=None,
    # ),
    
    # ScenarioConfig(
    #     name="dt_compute_surge",
    #     data_size_trend=1.05,
    #     n_episodes=1000,
    #     task_type_weights=None,
    # ),

    ScenarioConfig(
        name="dt_baseline",
        n_episodes=1000,
        task_type_weights=None,
    ),
]


# ============================================================
# Main simulation env
# ============================================================

# Snapshot vehicle protection window (seconds)
WARMUP_SECONDS = 5.0


class PurePythonSimEnv:

    def __init__(
        self,
        sumo_cfg:      str,
        rsus_xml:      str                  = "rsu.add.xml",
        scenario:      ScenarioConfig       = None,
        sumo_binary:   str                  = "sumo",
        step_length:   float               = 0.5,
        channel:       ChannelModel        = None,
        server_states: Optional[List[dict]] = None,
        car_states:    Optional[List[dict]] = None,
        arrival_mean:  Optional[float]      = None,
    ):
        self.sumo_cfg    = sumo_cfg
        self.sumo_binary = sumo_binary
        self.step_length = step_length
        self.scenario    = scenario if scenario is not None else ScenarioConfig(name="basic_baseline")
        self.channel     = channel or ChannelModel.from_config(SIM_DEFAULTS.get("channel"))

        # Load servers from XML, then override with snapshot state
        self.servers: Dict[str, EdgeServer] = load_servers_from_xml(rsus_xml)
        self._sorted_sids = sorted(
            self.servers.keys(), key=lambda s: int(s.replace("server", "")))

        if server_states:
            for s in server_states:
                sid = s.get("id")
                if sid in self.servers:
                    self.servers[sid].current_load = float(s.get("currentLoad", 0.0))
                    if "processingRate" in s:
                        self.servers[sid].processing_rate = float(s["processingRate"])
 
        self._server_base_rates: Dict[str, float] = {
            sid: srv.processing_rate for sid, srv in self.servers.items()
        }
        # Initial vehicle state (from snapshot)
        self._car_initial_rates: Dict[str, float] = {}
        self._car_initial_loads: Dict[str, float] = {}
        self._initial_veh_count = 0
        self._snapshot_vids: set = set()

        if car_states:
            self._initial_veh_count = len(car_states)
            for c in car_states:
                vid = c.get("id", "")
                self._car_initial_rates[vid] = float(c.get("processingRate", 1e9))
                self._car_initial_loads[vid] = float(c.get("currentLoad",    0.0))
                self._snapshot_vids.add(vid)

        self.vehicles:       Dict[str, dict]  = {}
        self._task_counter:  int              = 0
        self._next_task_at:  Dict[str, float] = {}
        self._sim_time:      float            = 0.0
        self._traci                           = None
        self._injected_cnt:  int              = 0
        yaml_mean = float(SIM_DEFAULTS.get("arrival", {}).get("mean", 10.0))
        self._arrival_mean: float = arrival_mean if arrival_mean is not None else yaml_mean
        self._events_fired = {"failure": False, "burst": False}

    # ------------------------------------------------------------------
    # Start / close
    # ------------------------------------------------------------------
    def start(self):
        if not HAS_TRACI:
            raise RuntimeError("TraCI not found.")
        traci.start([self.sumo_binary, "-c", self.sumo_cfg,
                     "--step-length", str(self.step_length),
                     "--collision.action", "none",
                     "--ignore-route-errors", "true",
                     "--no-warnings", "true"])
        self._traci    = traci
        self._sim_time = 0.0
        print(f"[SIM] Scenario '{self.scenario.name}' started | "
              f"{len(self.servers)} servers loaded")

    def close(self):
        if self._traci:
            try:
                traci.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # SUMO advance
    # ------------------------------------------------------------------
    def advance_sumo(self, progress: float = 0.0):
        # Switch to target values immediately (no ramp)
        rate_scale = self.scenario.server_rate_trend
        for sid, srv in self.servers.items():
            base_rate = self._server_base_rates.get(sid, srv.processing_rate)
            srv.processing_rate = base_rate * rate_scale    

        speed_factor = self.scenario.speed_trend
        for vid in traci.vehicle.getIDList():
            try:
                traci.vehicle.setSpeedFactor(vid, speed_factor)
            except Exception:
                pass

        self._adjust_vehicle_count()

        traci.simulationStep()
        self._sim_time = traci.simulation.getTime()
        active = set(traci.vehicle.getIDList())

        # Remove departed vehicles
        for vid in list(self.vehicles.keys()):
            if vid not in active:
                del self.vehicles[vid]
                self._next_task_at.pop(vid, None)

        # Update/add vehicles
        for vid in active:
            try:
                px, py = traci.vehicle.getPosition(vid)
                speed  = traci.vehicle.getSpeed(vid)
                angle  = math.radians(traci.vehicle.getAngle(vid))
            except traci.exceptions.TraCIException:
                continue

            if vid not in self.vehicles:
                rate_range = SIM_DEFAULTS.get("car_local_rate_range", (0.1e9, 0.2e9))
                local_rate = self._car_initial_rates.pop(
                    vid, random.uniform(*rate_range))
                local_load = self._car_initial_loads.pop(vid, 0.0)
                self.vehicles[vid] = {
                    "local_rate":      local_rate,
                    "local_load":      local_load,
                    "local_load_time": self._sim_time,
                }
                self._next_task_at[vid] = self._sim_time + self._sample_interarrival()

            self.vehicles[vid].update({
                "px": px, "py": py,
                "vx": speed * math.sin(angle),
                "vy": speed * math.cos(angle),
            })

    # ------------------------------------------------------------------
    # Queue check + strict vehicle removal policy (multi-scenario)
    # ------------------------------------------------------------------
    def _adjust_vehicle_count(self):

        if self._sim_time < WARMUP_SECONDS:
            return
        if self._initial_veh_count == 0:
            return

        target  = int(self._initial_veh_count * self.scenario.veh_count_trend)
        # Count vehicles on road + pending
        on_road = set(traci.vehicle.getIDList())
        try:
            pending = set(traci.simulation.getMinExpectedNumber())
        except Exception:
            pending = set()

        current_total = len(on_road) + len(pending)
        diff = target - current_total

        if diff > 0:
            for _ in range(diff):
                self._inject_vehicle()
                
        elif diff < 0:
            over_count = abs(diff)
            on_road_list = list(on_road)

            # Strict priority removal to protect snapshot vehicles
            # In multisce_sim, injected vehicles use dyn_v_ prefix
            dyn_vehs = [v for v in on_road_list if v.startswith("dyn_v_") or v.startswith("sim_dyn_")]
            
            other_vehs = [v for v in on_road_list if v not in self._snapshot_vids and v not in dyn_vehs]
            snapshot_vehs = [v for v in on_road_list if v in self._snapshot_vids]

            to_remove = []

            if len(dyn_vehs) >= over_count:
                to_remove = dyn_vehs[:over_count]
            else:
                to_remove.extend(dyn_vehs)
                rem_needed = over_count - len(to_remove)

                if len(other_vehs) >= rem_needed:
                    to_remove.extend(other_vehs[:rem_needed])
                else:
                    to_remove.extend(other_vehs)
                    rem_needed = over_count - len(to_remove)

                    if rem_needed > 0:
                        print(f"[SIM WARN] Target too low, forced to remove {rem_needed} snapshot vehicles!")
                        to_remove.extend(snapshot_vehs[:rem_needed])

            for vid in to_remove:
                try:
                    traci.vehicle.remove(vid)
                except Exception:
                    pass

    def _inject_vehicle(self):
        """Method B: inject a vehicle with a planned route via findRoute."""
        all_edges = [e for e in traci.edge.getIDList() if not e.startswith(':')]
        if len(all_edges) < 2:
            return
        try:
            best_edges = []
            best_len = 0

            for _ in range(12):
                start_edge = random.choice(all_edges)
                end_edge   = random.choice(all_edges)
                if start_edge == end_edge:
                    continue

                waypoints = []
                if len(all_edges) >= 4 and random.random() < 0.7:
                    waypoint_count = 15
                    candidates = [e for e in all_edges if e not in {start_edge, end_edge}]
                    if candidates:
                        random.shuffle(candidates)
                        waypoints = candidates[:min(waypoint_count, len(candidates))]

                edges = []
                path_ok = True
                prev_edge = start_edge
                for wp in waypoints + [end_edge]:
                    result = traci.simulation.findRoute(prev_edge, wp)
                    segment = list(result.edges)
                    if not segment:
                        path_ok = False
                        break
                    if edges and segment and edges[-1] == segment[0]:
                        edges.extend(segment[1:])
                    else:
                        edges.extend(segment)
                    prev_edge = wp

                if path_ok and len(edges) > best_len:
                    best_edges = edges
                    best_len = len(edges)

            if not best_edges:
                return
            route_id = f"dyn_r_{self._injected_cnt}"
            vid      = f"dyn_v_{self._injected_cnt}"
            self._injected_cnt += 1
            traci.route.add(route_id, best_edges)
            traci.vehicle.add(vid, route_id, typeID="car",
                              depart="now", departSpeed="0")
        except Exception as e:
            pass

    # ------------------------------------------------------------------
    # Offload execution
    # ------------------------------------------------------------------
    def _execute_offload(self, task, vid, v, server_id, now) -> dict:
        px, py = v["px"], v["py"]

        if server_id is None or server_id == "local":
            local_rate = v["local_rate"]
            local_load = self._get_local_load(vid, now)
            delay      = (local_load + task["demandResource"]) / local_rate
            self.vehicles[vid]["local_load"] = local_load + task["demandResource"]
            return {"taskId": task["taskId"], "vehicle": vid, "server": "local",
                    "dist_m": 0.0, "upload_time": 0.0,
                    "proc_delay": delay, "total_delay": delay}

        srv = self.servers.get(server_id)
        if srv is None:
            return self._execute_offload(task, vid, v, None, now)

        dist_m   = self._dist(px, py, srv.px, srv.py)
        t_upload = self.channel.upload_time_s(task["inputDataSize"], dist_m)
        t_proc   = srv.submit_task(task["demandResource"], now)

        return {"taskId": task["taskId"], "vehicle": vid, "server": server_id,
                "dist_m": dist_m, "upload_time": t_upload,
                "proc_delay": t_proc, "total_delay": t_upload + t_proc}

    # ------------------------------------------------------------------
    # Local load decay
    # ------------------------------------------------------------------
    def _get_local_load(self, vid: str, now: float) -> float:
        v    = self.vehicles[vid]
        dt   = now - v.get("local_load_time", now)
        load = max(0.0, v.get("local_load", 0.0) - dt * v["local_rate"])
        v["local_load"]      = load
        v["local_load_time"] = now
        return load

    # ------------------------------------------------------------------
    # Task interarrival
    # ------------------------------------------------------------------
    def _sample_interarrival(self) -> float:
        mean = self._arrival_mean
        if mean is None:
            arrival = SIM_DEFAULTS.get("arrival", {})
            mean = float(arrival.get("mean", 10.0))
        return random.expovariate(1.0 / mean)

    # ------------------------------------------------------------------
    # RL observation API
    # ------------------------------------------------------------------
    def get_observations(self) -> Dict[str, dict]:
        now = self._sim_time
        obs = {}

        for vid, v in self.vehicles.items():
            local_load = self._get_local_load(vid, now)
            valid_sids = self.get_valid_servers(self.servers)

            srv_obs = []
            for sid in valid_sids:
                srv = self.servers[sid]
                srv.update_load(now)
                dist = self._dist(v["px"], v["py"], srv.px, srv.py)
                srv_obs.append({
                    "id":           sid,
                    "dist_m":       dist,
                    "proc_rate":    srv.processing_rate,
                    "current_load": srv.current_load,
                })

            obs[vid] = {
                "pos":           [v["px"], v["py"]],
                "vel":           [v.get("vx", 0.0), v.get("vy", 0.0)],
                "local_rate":    v["local_rate"],
                "local_load":    local_load,
                "servers":       srv_obs,
                "valid_actions": ["local"] + valid_sids,
            }
        return obs

    def get_valid_servers(self, servers: Dict[str, EdgeServer]) -> List[str]:
        return sorted(servers.keys(), key=lambda s: int(s.replace("server", "")))

    @staticmethod
    def _dist(x1, y1, x2, y2) -> float:
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)