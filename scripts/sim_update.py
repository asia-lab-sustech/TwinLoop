import json
import math
import os
import random
import subprocess
import time
import sumolib
import traci
SIM_DIR      = "/home/shuoer/marl2/sumo"
DT_NET_FILE  = "city_2km.net.xml"
DT_ROU_FILE  = "dt_heterogeneous.rou.xml"
DT_SUMO_CFG  = "dt_heterogeneous.sumocfg"
TEMP_CFG     = "temp_mapping.sumocfg"
RSU_ADD_FILE = "rsu.add.xml"
SUMO_BIN     = "sumo"
PREPROC_PORT = 19999

MIN_ROUTE_EDGES   = 40
TARGET_WAYPOINTS  = 8
MAX_RETRY         = 15
MAX_DEPART_SPEED  = 8.0


class SnapshotProcessor:
    def __init__(self):
        if os.path.exists(DT_NET_FILE):
            try:
                self.net = sumolib.net.readNet(DT_NET_FILE)
                b = self.net.getBBoxXY()
                try: 
                    self.map_h = b[1][1] - b[0][1]
                except: 
                    self.map_h = float(b[3]) - float(b[1])
            except Exception as e:
                print(f"[!] Error loading net with sumolib: {e}")
                self.map_h = 5000.0
        else:
            self.map_h = 5000.0
            self.net = None
        print(f"[*] Map initialized for DT environment.")

    def load_local_snapshot(self, filepath="snapshot.json"):
        if not os.path.exists(filepath):
            print(f"[!] Snapshot file not found at {filepath}")
            return None
        print(f"[*] Loading snapshot from {filepath}...")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                snap = json.load(f)
            print(f"[*] Successfully loaded snapshot: {len(snap.get('cars',[]))} cars, {len(snap.get('servers',[]))} servers")
            return snap
        except Exception as e:
            print(f"[!] Failed to parse snapshot: {e}")
            return None

    def transform_coordinates(self, x, y):
        return x, y

    def _build_long_route_traci(self, start_edge, all_normal_edges):
        edges = []
        prev_edge = start_edge
        wp_candidates = [e for e in all_normal_edges if e != start_edge]
        random.shuffle(wp_candidates)
        wp_used = 0
        for wp in wp_candidates:
            if wp_used >= TARGET_WAYPOINTS:
                break
            if wp == prev_edge:
                continue
            try:
                result = traci.simulation.findRoute(prev_edge, wp)
                segment = list(result.edges)
                if not segment or len(segment) < 2:
                    continue
                if edges and segment[0] == edges[-1]:
                    edges.extend(segment[1:])
                else:
                    edges.extend(segment)
                prev_edge = wp
                wp_used += 1
            except Exception:
                continue
        return edges

    def _snap_to_safe_edge(self, px, py, all_normal_edges):
        safe_px = max(50.0, min(px, self.map_h - 50.0))
        safe_py = max(50.0, min(py, self.map_h - 50.0))
        try:
            edge_id, pos, lane_idx = traci.simulation.convertRoad(
                safe_px, safe_py, isGeo=False, vClass="passenger")
            if edge_id and not edge_id.startswith(":"):
                return edge_id, pos, lane_idx
        except Exception:
            pass
        if all_normal_edges:
            return random.choice(all_normal_edges), 0.0, 0
        return None, 0.0, 0

    def generate_mapped_route_file(self, snapshot):
        if not snapshot or "cars" not in snapshot: 
            return False
        print("[*] Generating SUMO route file (findRoute multi-waypoint)...")
        
        with open(TEMP_CFG, "w") as f:
            f.write(f'<configuration><input><net-file value="{DT_NET_FILE}"/></input>'
                    f'<time><begin value="0"/><step-length value="0.1"/></time>'
                    f'<report><verbose value="false"/><no-step-log value="true"/></report></configuration>')
        
        proc = subprocess.Popen([SUMO_BIN, "-c", TEMP_CFG, "--remote-port", str(PREPROC_PORT)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.0)
        
        mapped = []
        skipped = 0
        try:
            traci.init(PREPROC_PORT)
            all_normal_edges = [e for e in traci.edge.getIDList() if not e.startswith(':')]
            print(f"[*] Network has {len(all_normal_edges)} normal edges.")

            for i, car in enumerate(snapshot['cars']):
                vid = car.get("id", f"veh{i}")
                try:
                    px = float(car.get('px', 0))
                    py = float(car.get('py', 0))

                    edge_id, pos, lane_idx = None, 0.0, 0
                    try:
                        edge_id, pos, lane_idx = traci.simulation.convertRoad(
                            px, py, isGeo=False, vClass="passenger")
                    except Exception:
                        pass

                    if not edge_id or edge_id.startswith(":"):
                        edge_id, pos, lane_idx = self._snap_to_safe_edge(
                            px, py, all_normal_edges)
                        if not edge_id:
                            skipped += 1
                            continue

                    try:
                        lane_id = f"{edge_id}_{int(lane_idx)}"
                        lane_len = traci.lane.getLength(lane_id)
                        safe_pos = max(0.0, min(pos, lane_len - 20.0))
                        if safe_pos < 0.0:
                            safe_pos = 0.0
                    except Exception:
                        safe_pos = 0.0

                    vx = float(car.get('vx', 0))
                    vy = float(car.get('vy', 0))
                    speed = math.sqrt(vx**2 + vy**2)

                    best_edges = []
                    for retry in range(MAX_RETRY):
                        candidate = self._build_long_route_traci(edge_id, all_normal_edges)
                        if len(candidate) > len(best_edges):
                            best_edges = candidate
                        if len(best_edges) >= MIN_ROUTE_EDGES:
                            break

                    if len(best_edges) < 2:
                        skipped += 1
                        continue

                    route_str = " ".join(best_edges)
                    safe_speed = min(speed, MAX_DEPART_SPEED)

                    mapped.append({
                        "id": vid,
                        "route": route_str,
                        "pos": safe_pos,
                        "speed": float(safe_speed),
                        "edge_count": len(best_edges),
                    })
                except Exception as e:
                    print(f"[!] Failed to process vehicle {vid}: {e}")
                    skipped += 1
        except Exception as e:
            print(f"[!] TraCI mapping error: {e}")
            return False
        finally:
            try: traci.close()
            except: pass
            if proc.poll() is None: proc.terminate()

        if skipped > 0:
            print(f"[!] Skipped {skipped} vehicles total.")
        if not mapped: 
            print("[!] No vehicles mapped successfully.")
            return False

        edge_counts = [m["edge_count"] for m in mapped]
        print(f"[*] Route statistics: min={min(edge_counts)} max={max(edge_counts)} "
              f"avg={sum(edge_counts)/len(edge_counts):.0f} edges")

        with open(DT_ROU_FILE, "w") as f:
            f.write('<routes>\n')
            f.write('  <vType id="car" accel="2.5" decel="4.5" length="4" minGap="2.5" '
                    'maxSpeed="50.0" guiShape="passenger"/>\n')
            for c in mapped:
                f.write(f'  <vehicle id="{c["id"]}" type="car" '
                        f'depart="0" '
                        f'departPos="{c["pos"]:.2f}" '
                        f'departSpeed="{c["speed"]:.2f}">\n')
                f.write(f'    <route edges="{c["route"]}"/>\n  </vehicle>\n')
            f.write('</routes>\n')
        print(f"[*] Route file written with {len(mapped)} vehicles.")
        return True

    def generate_rsu_add_from_snapshot(self, snapshot):
        if not snapshot: 
            return False
        servers = snapshot.get("servers", [])
        print(f"[*] Generating RSU additional file ({len(servers)} servers)...")
        try:
            servers = sorted(servers, key=lambda s: int(s.get("id", "server0").replace("server", "")))
        except Exception:
            pass
        with open(RSU_ADD_FILE, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<additional>\n')
            for s in servers:
                sid = s.get("id", f"rsu_{random.randint(0,9999)}")
                tx, ty = self.transform_coordinates(float(s.get("px", 0)), float(s.get("py", 0)))
                desc = s.get("desc", "RSU")
                f.write(f'  <poi id="{sid}" x="{tx:.2f}" y="{ty:.2f}" type="rsu" color="red" layer="100" desc="{desc}"/>\n')
            f.write('</additional>\n')
        return True

    def generate_initial_sumocfg(self, outfile=DT_SUMO_CFG):
        print(f"[*] Generating SUMO config: {outfile}")
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(f'''<?xml version="1.0" encoding="UTF-8"?>
            <configuration>
                <input>
                    <net-file value="{DT_NET_FILE}"/>
                    <route-files value="{DT_ROU_FILE}"/>
                    <additional-files value="{RSU_ADD_FILE}"/>
                </input>
                <time>
                    <begin value="0"/>
                    <step-length value="1.0"/>
                </time>
                <report>
                    <verbose value="false"/>
                    <no-warnings value="true"/>
                </report>
            </configuration>''')
        return outfile

if __name__ == "__main__":
    print("="*60)
    print("  Step 1: Digital Twin Environment Fast Update (Pure Python)")
    print("="*60)
    os.chdir(SIM_DIR)
    processor = SnapshotProcessor()
    try:
        snap = processor.load_local_snapshot("snapshot.json")
        if snap:
            processor.generate_mapped_route_file(snap)
            processor.generate_rsu_add_from_snapshot(snap)
            processor.generate_initial_sumocfg()
            print("\n✅ Environment files successfully generated! Ready to run DT simulation.")
        else:
            print("\n❌ Setup failed: Could not load or parse snapshot.json.")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")