# scripts/sim.py

import math
import random
import os
import yaml
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import traci

# ============================================================
# ChannelModel
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
# simulation defaults configuration
# ============================================================
SIM_DEFAULTS = {}
try:
    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'sim_defaults.yaml'))
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

# ============================================================
# server model
# ============================================================
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

def load_servers_from_xml(xml_path: str) -> Dict[str, EdgeServer]:
    servers = {}
    if not os.path.exists(xml_path):
        print(f"[ERROR] RSU XML not found at {xml_path}")
        return servers

    rates = SIM_DEFAULTS.get("server_rates", [])
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    for i, poi in enumerate(root.findall('poi')):
        clean_id = f"server{i}"
        px = float(poi.get('x', 0.0))
        py = float(poi.get('y', 0.0))
        rate = rates[i] if i < len(rates) else 2.0e9
        servers[clean_id] = EdgeServer(
            server_id=clean_id, px=px, py=py,
            processing_rate=rate, current_load=0.0
        )
    return servers

def generate_task(task_id: int, src_id: str) -> dict:
    templates = SIM_DEFAULTS.get("task_templates")
    weights = SIM_DEFAULTS.get("task_template_weights")
    if not weights or len(weights) != len(templates):
        weights = [1] * len(templates)
        
    idx = random.choices(range(len(templates)), weights=weights, k=1)[0]
    t = templates[idx]
    
    demand = max(1e5, random.gauss(float(t.get("demand_mean")), float(t.get("demand_std"))))
    data   = max(1e3, random.gauss(float(t.get("data_mean")), float(t.get("data_std"))))

    return {
        "taskId":         task_id,
        "srcAddr":        src_id,
        "demandResource": float(demand),
        "inputDataSize":  float(data),
    }

# ============================================================
# simulation environment
# ============================================================
WARMUP_SECONDS = 5.0


class PurePythonSimEnv:
    def __init__(
        self,
        sumo_cfg:       str           = "city_simulation.sumocfg",
        rsus_xml:       str           = "rsus.add.xml",
        sumo_binary:    str           = "sumo",
        step_length:    float         = 0.5,
        channel:        Optional['ChannelModel'] = None,
        server_states:  Optional[List[dict]] = None,
        car_states:     Optional[List[dict]] = None,
        target_vehicles: Optional[int] = None,
    ):
        self.sumo_cfg    = sumo_cfg
        self.sumo_binary = sumo_binary
        self.step_length = step_length
        self.channel     = channel if channel else ChannelModel.from_config(SIM_DEFAULTS.get("channel"))
        self.servers: Dict[str, EdgeServer] = load_servers_from_xml(rsus_xml)
        self._sorted_sids = sorted(self.servers.keys(), key=lambda s: int(s.replace("server", "")))

        if server_states:
            for s in server_states:
                sid = s.get("id")
                if sid in self.servers:
                    self.servers[sid].current_load = float(s.get("currentLoad", 0.0))
                    if "processingRate" in s:
                        self.servers[sid].processing_rate = float(s["processingRate"])

        self._car_initial_rates: Dict[str, float] = {}
        self._car_initial_loads: Dict[str, float] = {}
        self._snapshot_vids: set = set()
        if car_states:
            for c in car_states:
                vid = c.get("id")
                self._car_initial_rates[vid] = float(c.get("processingRate", 1e9))
                self._car_initial_loads[vid] = float(c.get("currentLoad", 0.0))
                self._snapshot_vids.add(vid)

        self.vehicles:      Dict[str, dict]  = {}
        self._task_counter: int              = 0
        self._next_task_at: Dict[str, float] = {}
        self._sim_time:     float            = 0.0
        self._traci                          = None
        self._target_vehicles: Optional[int] = target_vehicles
        self._injected_cnt:    int           = 0

    def start(self):
        traci.start([
            self.sumo_binary, "-c", self.sumo_cfg,
            "--step-length", str(self.step_length),
            "--collision.action", "none",
            "--ignore-route-errors", "true",
            "--no-warnings", "true",            
        ])                                      
        self._traci = traci
        self._sim_time = 0.0

        n_snap = len(self._snapshot_vids)
        print(f"[SIM] Started | {len(self.servers)} servers | "
              f"{n_snap} snapshot vehicles | target={self._target_vehicles}")

    def close(self):
        if self._traci:
            try:
                traci.close()
            except Exception:
                pass

    def advance_sumo(self):
        traci.simulationStep()
        self._sim_time = traci.simulation.getTime()

        self._adjust_vehicle_count()
        active = set(traci.vehicle.getIDList())
        for vid in list(self.vehicles.keys()):
            if vid not in active:
                del self.vehicles[vid]
                self._next_task_at.pop(vid, None)

        for vid in active:
            try:
                px, py = traci.vehicle.getPosition(vid)
                speed  = traci.vehicle.getSpeed(vid)
                angle_rad = math.radians(traci.vehicle.getAngle(vid))
            except traci.exceptions.TraCIException:
                continue

            if vid not in self.vehicles:
                rate_range = SIM_DEFAULTS.get("car_local_rate_range", (0.1e9, 0.2e9))
                local_rate = self._car_initial_rates.pop(vid, random.uniform(*rate_range))
                local_load = self._car_initial_loads.pop(vid, 0.0)

                self.vehicles[vid] = {
                    "local_rate": local_rate,
                    "local_load": local_load,
                    "local_load_time": self._sim_time,
                }
                self._next_task_at[vid] = self._sim_time + self._sample_interarrival()

            self.vehicles[vid].update({
                "px": px, "py": py,
                "vx": speed * math.sin(angle_rad),
                "vy": speed * math.cos(angle_rad),
            })

    def _adjust_vehicle_count(self):
        """
        maintains the total number of vehicles close to self._target_vehicles by injecting or removing vehicles as needed. The removal process prioritizes non-snapshot vehicles to preserve the integrity
        """
        if self._target_vehicles is None:
            return
        if self._sim_time < WARMUP_SECONDS:
            return
        on_road = set(traci.vehicle.getIDList())
        try:
            pending = set(traci.simulation.getMinExpectedNumber())
        except Exception:
            pending = set()

        current_total = len(on_road) + len(pending)
        diff = self._target_vehicles - current_total

        if diff > 0:
            for _ in range(diff):
                self._inject_vehicle()
                
        elif diff < 0:
            over_count = abs(diff)
            on_road_list = list(on_road)
            # remove has to be done carefully to avoid deleting snapshot vehicles. The priority is:
            # 1. dynamically injected vehicles (sim_dyn_*, pt_dyn_*)
            # 2. other non-snapshot vehicles            
            # 3. snapshot vehicles (only if absolutely necessary)
            dyn_vehs = [v for v in on_road_list if v.startswith("sim_dyn_") or v.startswith("pt_dyn_")]
            
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
                        print(f"[SIM WARN] 目标车辆数过低，被迫删除 {rem_needed} 辆 PT 快照映射车！")
                        to_remove.extend(snapshot_vehs[:rem_needed])

            for vid in to_remove:
                try:
                    traci.vehicle.remove(vid)
                except Exception:
                    pass

    def _inject_vehicle(self):
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
                if random.random() < 0.7:
                    waypoint_count = 5
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
            route_id = f"sim_dyn_r_{self._injected_cnt}"
            vid      = f"sim_dyn_{self._injected_cnt}"
            self._injected_cnt += 1
            traci.route.add(route_id, best_edges)
            traci.vehicle.add(vid, route_id, typeID="car",
                              depart="now", departSpeed="0")
        except Exception as e:
            print(f"[SIM] Failed to inject vehicle: {e}")

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
        total    = t_upload + t_proc

        return {"taskId": task["taskId"], "vehicle": vid, "server": server_id,
                "dist_m": dist_m, "upload_time": t_upload,
                "proc_delay": t_proc, "total_delay": total}
    
    def _get_local_load(self, vid: str, now: float) -> float:
        v  = self.vehicles[vid]
        dt = now - v.get("local_load_time", now)
        load = max(0.0, v.get("local_load") - dt * v["local_rate"])
        v["local_load"]      = load
        v["local_load_time"] = now
        return load

    @staticmethod
    def _dist(x1, y1, x2, y2) -> float:
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    def _sample_interarrival(self) -> float:
        arrival = SIM_DEFAULTS.get("arrival", {})
        mean = float(arrival.get("mean", 8.0))
        return random.expovariate(1.0 / mean)

    def get_observations(self) -> Dict[str, dict]:
        now = self._sim_time
        obs = {}

        for vid, v in self.vehicles.items():
            local_load = self._get_local_load(vid, now)
            valid_sids = self.get_valid_servers(self.servers)

            srv_obs = []
            for sid in valid_sids:
                srv  = self.servers[sid]
                srv.update_load(now)  
                dist = self._dist(v["px"], v["py"], srv.px, srv.py)
                srv_obs.append({
                    "id":           sid,
                    "dist_m":       dist,
                    "proc_rate":    srv.processing_rate,
                    "current_load": srv.current_load,
                })

            obs[vid] = {
                "pos":              [v["px"], v["py"]],
                "vel":              [v.get("vx"), v.get("vy")],
                "local_rate":       v["local_rate"],
                "local_load":       local_load,       
                "servers":          srv_obs,
                "valid_actions":    ["local"] + valid_sids,
            }
        return obs
    
    def get_valid_servers(self, servers: Dict[str, EdgeServer]) -> List[str]:
        return sorted(servers.keys(), key=lambda s: int(s.replace("server", "")))