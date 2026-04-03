# scripts/pt_scenario_scheduler.py

import math
import random
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
 
try:
    import traci
except ImportError:
    traci = None
 
WARMUP_SECONDS = 5.0

# ═══════════════════════════════════════════════════════════════
# Phase 定义：每个 phase 描述一段时间内环境应该是什么样的
# ═══════════════════════════════════════════════════════════════
@dataclass
class Phase:
    name:           str
    start_time:     float   
    end_time:       float   
    
    target_vehicles: Optional[int] = None
    speed_factor:    float = 1.0
    server_overrides: Optional[Dict[str, float]] = None
    
    demand_multiplier: float = 1.0    
    data_multiplier:   float = 1.0    
    arrival_multiplier: float = 1.0   
    task_type_weights: Optional[List[float]] = None

    description: str = ""
 
def make_default_scenario(total_time: float = 1800.0) -> List[Phase]:
    T = {
        "06h":    0,
        "08h":  500,
        "09h":  1500,
        "11h":  2500,
        "end":  3500,
    }

    return [
        Phase(
            name="morning_light",
            start_time=T["06h"], end_time=T["08h"],
            target_vehicles=45,
            speed_factor=1.0,
            arrival_multiplier=0.8,
            demand_multiplier=1.0,
            # [offload_medium, boundary_A, boundary_B, boundary_C, light_local, navigation]
            task_type_weights=[2, 2, 2, 2, 7, 8],
        ),

        Phase(
            name="morning_peak_rise",
            start_time=T["08h"], end_time=T["09h"],
            target_vehicles=65,
            speed_factor=0.7,
            arrival_multiplier=1.0,
            demand_multiplier=1.0,
            data_multiplier=1.0,
            task_type_weights=[1, 1, 2, 3, 3, 10],
        ),

        Phase(
            name="morning_peak_decay",
            start_time=T["09h"], end_time=T["11h"],
            target_vehicles=65,
            speed_factor=0.85,
            server_overrides={
                "server0":  1.2e9,
                "server1":  0.9e9,
                "server2":  0.8e9,
                "server3":  0.7e9,
                
                "server12": 0.7e9,
                "server13": 0.8e9,
                "server14": 0.9e9,
                "server15": 1.2e9,
            },
            arrival_multiplier=1.2,
            demand_multiplier=1.0,
            task_type_weights=[2, 3, 3, 3, 4, 6],
        ),

        Phase(
            name="midday",
            start_time=T["11h"], end_time=T["end"],
            target_vehicles=55,
            speed_factor=1.0,
            server_overrides={
                "server0":  2.0e9,
                "server1":  2.0e9,
                "server2":  2.0e9,
                "server3":  2.0e9,
                
                "server12": 2.5e9,
                "server13": 2.5e9,
                "server14": 2.5e9,
                "server15": 2.5e9,
            },
            arrival_multiplier=1.0,
            demand_multiplier=0.8,
            data_multiplier=1.5,
            task_type_weights=[1, 2, 2, 3, 10, 3],
        ),
    ]
 
 
class ScenarioScheduler:
    
    def __init__(
        self,
        phases: List[Phase],
        original_server_rates: Dict[str, float],
        original_arrival_mean: float = 10.0,
        seed: int = 42,
    ):
        self.phases = sorted(phases, key=lambda p: p.start_time)
        self.original_rates = dict(original_server_rates)  # {sid: rate}
        self.original_arrival_mean = original_arrival_mean
        self.seed = seed
        self._rng = random.Random(seed)
        self._injected_cnt = 0
        self._step_counter = 0
        
        self.current_demand_mult = 1.0
        self.current_data_mult = 1.0
        self.current_arrival_mean = original_arrival_mean
        self.current_task_weights: Optional[List[float]] = None  
        
        self._last_phase_name = None
        self._active_overrides: Dict[str, float] = {}
        
        print(f"[Scheduler] Initialized with {len(phases)} phases, seed={seed}")
        for p in self.phases:
            print(f"  [{p.start_time:.0f}-{p.end_time:.0f}s] {p.name}: {p.description}")

    def get_current_phase(self, sim_time: float) -> Optional[Phase]:
        for p in self.phases:
            if p.start_time <= sim_time < p.end_time:
                return p
        return self.phases[-1] if self.phases else None
    
    def apply(self, env, sim_time: float):
        phase = self.get_current_phase(sim_time)
        if phase is None:
            return

        # ── Phase change detect ──
        if phase.name != self._last_phase_name:
            print(
                f"\n[Scheduler] ═══ PHASE CHANGE: {self._last_phase_name} → {phase.name} "
                f"(t={sim_time:.0f}s) ═══"
            )
            print(f"  {phase.description}")
            self._last_phase_name = phase.name

        # ── 1. server processing rate change ──
        self._apply_server_changes(env, phase, sim_time)

        # ── 2. the number of car ──
        if phase.target_vehicles is not None:
            self._adjust_vehicles(env, phase.target_vehicles)

        # ── 3. the speed ──
        speed_factor = phase.speed_factor
        if traci and abs(speed_factor - 1.0) > 0.01:
            for vid in traci.vehicle.getIDList():
                try:
                    traci.vehicle.setSpeedFactor(vid, speed_factor)
                except:
                    pass

        # ── 4. the task ──
        self.current_demand_mult = phase.demand_multiplier
        self.current_data_mult = phase.data_multiplier
        arrival_mult = max(phase.arrival_multiplier, 1e-6)
        self.current_arrival_mean = self.original_arrival_mean / arrival_mult
        self.current_task_weights = phase.task_type_weights
    
    def _apply_server_changes(self, env, phase: Phase, sim_time: float):
        """应用服务器 override，并在 override 解除时恢复原始值"""
        new_overrides = phase.server_overrides or {}
        
        for sid in list(self._active_overrides.keys()):
            if sid not in new_overrides and sid in env.servers:
                original = self.original_rates.get(sid, 2.0e9)
                env.servers[sid].processing_rate = original
                print(f"  [Scheduler] Server {sid} RESTORED to {original/1e9:.1f} GHz")
                del self._active_overrides[sid]
        
        for sid, rate in new_overrides.items():
            if sid in env.servers:
                if sid not in self._active_overrides or self._active_overrides[sid] != rate:
                    env.servers[sid].processing_rate = rate
                    if rate == 0:
                        print(f"  [Scheduler] Server {sid} CRASHED!")
                    else:
                        print(f"  [Scheduler] Server {sid} rate → {rate/1e9:.1f} GHz")
                    self._active_overrides[sid] = rate

    def _adjust_vehicles(self, env, target: int):
        if not traci:
            return
        if traci.simulation.getTime() < WARMUP_SECONDS:
            return

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
            dyn_vehs = [v for v in on_road_list if v.startswith("sim_dyn_") or v.startswith("pt_dyn_")]
            snapshot_vids = getattr(env, '_snapshot_vids', set())
            
            other_vehs = [v for v in on_road_list if v not in snapshot_vids and v not in dyn_vehs]
            snapshot_vehs = [v for v in on_road_list if v in snapshot_vids]

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
                        print(f"[Scheduler WARN] Target too low, forced to remove {rem_needed} PT mapped vehicles!")
                        to_remove.extend(snapshot_vehs[:rem_needed])

            for vid in to_remove:
                try:
                    traci.vehicle.remove(vid)
                except Exception:
                    pass
    
    def _inject_vehicle(self):
        """inject a new vehicle with a random route (only when we need more vehicles than current)"""
        if not traci:
            return
        all_edges = [e for e in traci.edge.getIDList() if not e.startswith(':')]
        if len(all_edges) < 2:
            return
        try:
            best_edges: List[str] = []
            best_len = 0

            for _ in range(12):
                start_edge = self._rng.choice(all_edges)
                end_edge = self._rng.choice(all_edges)
                if start_edge == end_edge:
                    continue

                waypoints = []
                if len(all_edges) >= 4 and self._rng.random() < 0.7:
                    waypoint_count = 15
                    candidates = [e for e in all_edges if e not in {start_edge, end_edge}]
                    if candidates:
                        self._rng.shuffle(candidates)
                        waypoints = candidates[:min(waypoint_count, len(candidates))]

                edges: List[str] = []
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
            route_id = f"pt_dyn_r_{self._injected_cnt}"
            vid = f"pt_dyn_{self._injected_cnt}"
            self._injected_cnt += 1
            traci.route.add(route_id, best_edges)
            traci.vehicle.add(vid, route_id, typeID="DEFAULT_VEHTYPE", depart="now", departSpeed="0")
        except Exception:
            pass
    
    def close(self):
        pass
 
 
def generate_task_dynamic(task_id: int, src_id: str, scheduler: ScenarioScheduler,
                          sim_defaults: dict) -> dict:
    templates = sim_defaults.get("task_templates")

    weights = scheduler.current_task_weights
    if weights is None or len(weights) != len(templates):
        weights = sim_defaults.get("task_template_weights")
    if not weights or len(weights) != len(templates):
        weights = [1] * len(templates)

    idx = random.choices(range(len(templates)), weights=weights, k=1)[0]
    t = templates[idx]

    demand = max(1e5, random.gauss(
        float(t.get("demand_mean")), float(t.get("demand_std"))
    )) * scheduler.current_demand_mult

    data = max(1e3, random.gauss(
        float(t.get("data_mean")), float(t.get("data_std"))
    )) * scheduler.current_data_mult

    return {
        "taskId":         task_id,
        "srcAddr":        src_id,
        "demandResource": float(demand),
        "inputDataSize":  float(data),
    }
 
 
def sample_interarrival_dynamic(scheduler: ScenarioScheduler) -> float:
    """替代 env._sample_interarrival()，使用动态到达率"""
    mean = scheduler.current_arrival_mean
    return random.expovariate(1.0 / mean)

if __name__ == "__main__":
    phases = make_default_scenario()
    print("=" * 60)
    print("  PT Dynamic Scenario Definition")
    print("=" * 60)
    for p in phases:
        print(f"\n  [{p.start_time:.0f}s - {p.end_time:.0f}s] {p.name}")
        print(f"    {p.description}")
        print(f"    Vehicles: {p.target_vehicles or 'auto'}")
        print(f"    Speed: ×{p.speed_factor}")
        print(f"    Demand: ×{p.demand_multiplier}, Data: ×{p.data_multiplier}")
        print(f"    Arrival: ×{p.arrival_multiplier} (mean → {10.0/p.arrival_multiplier:.1f}s)")
        if p.server_overrides:
            for sid, rate in p.server_overrides.items():
                print(f"    Server {sid}: {'CRASHED' if rate == 0 else f'{rate/1e9:.1f} GHz'}")
    print(f"{'='*60}")