import os
import glob
import json
from datetime import datetime
import argparse
import csv
import math
import statistics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIC9_DIR = BASE_DIR
CSV_PATTERNS = ['metrics*.csv', '*.csv']

def find_csv_files(directory):
    files = []
    for p in CSV_PATTERNS:
        files.extend(glob.glob(os.path.join(directory, p)))
    return sorted(list(dict.fromkeys(files)))

def get_scenario_name(sim_time):
    if 0 <= sim_time < 500:
        return "Scenario_1_(0-500s)"
    elif 500 <= sim_time < 1500:
        return "Scenario_2_(500-1500s)"
    elif 1500 <= sim_time < 2500:
        return "Scenario_3_(1500-2500s)"
    elif 2500 <= sim_time <= 3500:
        return "Scenario_4_(2500-3500s)"
    else:
        return "Other_(Out_of_Bounds)"

def calculate_distribution(val_list):
    if not val_list:
        return None
    count = len(val_list)
    val_list.sort()
    
    mean_val = sum(val_list) / count
    stdev_val = statistics.stdev(val_list) if count > 1 else 0.0
    
    return {
        'count': count,
        'mean': mean_val,
        'stdev': stdev_val,
        'min': val_list[0],
        'max': val_list[-1],
        'p50': val_list[int(count * 0.50)],
        'p90': val_list[int(count * 0.90)],
        'p99': val_list[int(count * 0.99)]
    }

def analyze_scenarios(csv_files):
    scenarios_data = {}

    for f in csv_files:
        try:
            with open(f, 'r', newline='') as fh:
                reader = csv.reader(fh)
                try:
                    header = next(reader)
                except StopIteration:
                    continue

                hdr = [h.strip().lower() for h in header]
                
                time_idx, vid_idx, lat_idx = None, None, None
                for i, name in enumerate(hdr):
                    if name in ('sim_time', 'time'): time_idx = i
                    if name in ('vid', 'car_id'): vid_idx = i
                    if name in ('latency_s', 'latency'): lat_idx = i

                if None in (time_idx, vid_idx, lat_idx):
                    print(f"[WARN] 文件 {os.path.basename(f)} 缺少必要的列(sim_time, vid, latency_s)，已跳过。")
                    continue

                for row in reader:
                    if len(row) <= max(time_idx, vid_idx, lat_idx):
                        continue
                    
                    try:
                        sim_time = float(row[time_idx].strip())
                        lat = float(row[lat_idx].strip())
                        vid = row[vid_idx].strip()
                    except ValueError:
                        continue
                    
                    if not math.isfinite(lat) or not math.isfinite(sim_time):
                        continue
                    
                    scenario_name = get_scenario_name(sim_time)
                    
                    if scenario_name not in scenarios_data:
                        scenarios_data[scenario_name] = {"all_latencies": [], "unique_cars": set()}
                    
                    scenarios_data[scenario_name]["all_latencies"].append(lat)
                    scenarios_data[scenario_name]["unique_cars"].add(vid)

        except Exception as e:
            print(f"[WARN] 处理文件出错 {f}: {e}")

    results = {}
    for scenario_name, data in scenarios_data.items():
        if not data["all_latencies"]:
            continue
            
        dist_metrics = calculate_distribution(data["all_latencies"])

        results[scenario_name] = {
            "distribution": dist_metrics,
            "car_metrics": {
                "total_unique_cars": len(data["unique_cars"]),
                "mean_s": dist_metrics['mean']
            }
        }
        
    return results

def main():
    parser = argparse.ArgumentParser(description='Analyze multi-scenario metrics.')
    parser.add_argument('file', nargs='?', help='CSV filename (relative to this directory) to process')
    args = parser.parse_args()

    if args.file:
        csv_path = os.path.join(PIC9_DIR, args.file)
        if not os.path.exists(csv_path):
            print(f"文件不存在：{csv_path}")
            return
        csvs = [csv_path]
    else:
        csvs = find_csv_files(PIC9_DIR)

    if not csvs:
        print("未找到 CSV 文件。")
        return

    for fpath in csvs:
        print(f"正在分析: {os.path.basename(fpath)}")
        scenario_results = analyze_scenarios([fpath])
        
        out = {
            'computed_at': datetime.utcnow().isoformat() + 'Z',
            'file': os.path.basename(fpath),
            'scenarios': scenario_results
        }
        
        base = os.path.splitext(os.path.basename(fpath))[0]
        out_path = os.path.join(PIC9_DIR, f"{base}_scenario_metrics.json")
        with open(out_path, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        print(f"  -> 分析完成，结果已保存至: {out_path}\n")

if __name__ == '__main__':
    main()