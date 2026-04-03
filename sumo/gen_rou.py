"""
运行方式：cd /home/shuoer/marl2/sumo && python gen_rou.py
只需运行一次，生成 vehicles_2km.rou.xml
"""
import sumolib, random

NET_FILE = "city_2km.net.xml"
OUT_FILE = "vehicles_2km.rou.xml"
N_VEHICLES = 50        # 初始车辆数（TraCI 会动态调整，这里只需能启动 SUMO）
HOPS = 5000             # 每辆车随机路由的跳数（越大路线越长）

net = sumolib.net.readNet(NET_FILE)
edges = [e for e in net.getEdges() if not e.getID().startswith(':')]

def random_route(start_edge, hops=600):
    cur = start_edge
    route = [cur.getID()]
    for _ in range(hops):
        out = list(cur.getOutgoing().keys())
        if not out:
            # 如果当前边走到尽头，随机换一个非内部边继续接着走，避免短路
            candidates = [e for e in edges if e.getOutgoing()]
            if not candidates:
                break
            cur = random.choice(candidates)
            route.append(cur.getID())
            continue
        # 优先从较多后继的边中继续，减少太早走到死胡同的概率
        if len(out) > 1 and random.random() < 0.75:
            out = sorted(out, key=lambda e: len(e.getOutgoing()), reverse=True)
        cur = random.choice(out)
        route.append(cur.getID())
    return route

random.seed(42)
with open(OUT_FILE, "w") as f:
    f.write('<routes>\n')
    f.write('  <vType id="car" accel="2.5" decel="4.5" length="4" '
            'minGap="2.5" maxSpeed="50.0" guiShape="passenger"/>\n\n')
    for i in range(N_VEHICLES):
        start = random.choice(edges)
        route = random_route(start)
        if len(route) < 2:
            continue
        f.write(f'  <vehicle id="init_{i}" type="car" depart="{0}" '
                f'departSpeed="0" departPos="random_free">\n')
        f.write(f'    <route edges="{" ".join(route)}"/>\n')
        f.write('  </vehicle>\n')
    f.write('</routes>\n')

print(f"[OK] {OUT_FILE} generated with {N_VEHICLES} seed vehicles.")
print("SUMO will use TraCI to manage actual vehicle count at runtime.")