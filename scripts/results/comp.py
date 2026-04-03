import os
import glob
import json
import csv

def main():
    json_files = glob.glob('*_scenario_metrics.json')
    if not json_files:
        print("未找到 *_scenario_metrics.json 文件，请先运行 analyze_metrics.py")
        return

    print(f"找到 {len(json_files)} 个结果文件，开始生成分块对比报告...")

    scenario_blocks = {}
    
    for j_file in sorted(json_files):
        exp_name = j_file.replace('_scenario_metrics.json', '')
        try:
            with open(j_file, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
            scenarios = data.get('scenarios', {})
            
            for sc_name, sc_data in scenarios.items():
                if sc_name not in scenario_blocks:
                    scenario_blocks[sc_name] = {}
                    
                dist = sc_data.get('distribution', {}) or {}
                
                scenario_blocks[sc_name][exp_name] = {
                    "mean": round(dist.get('mean', 0), 3) if dist.get('mean') is not None else "",
                    "p50": round(dist.get('p50', 0), 3) if dist.get('p50') is not None else "",
                    "p90": round(dist.get('p90', 0), 3) if dist.get('p90') is not None else "",
                    "p99": round(dist.get('p99', 0), 3) if dist.get('p99') is not None else "",
                    "stdev": round(dist.get('stdev', 0), 3) if dist.get('stdev') is not None else ""
                }
        except Exception as e:
            print(f"[WARN] 读取文件 {j_file} 失败: {e}")

    out_file = "comparison_report_blocks.csv"
    
    headers = [
        "实验名称(File)", 
        "全局均值(Mean_s)", 
        "P50(中位数)", 
        "P90(尾部)", 
        "P99(长尾)", 
        "标准差(StdDev)"
    ]

    with open(out_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        
        for sc_name in sorted(scenario_blocks.keys()):
            writer.writerow([f"========== {sc_name} =========="])
            writer.writerow(headers)
            
            exp_data_dict = scenario_blocks[sc_name]
            for exp_name in sorted(exp_data_dict.keys()):
                metrics = exp_data_dict[exp_name]
                writer.writerow([
                    exp_name,
                    metrics["mean"],
                    metrics["p50"],
                    metrics["p90"],
                    metrics["p99"],
                    metrics["stdev"]
                ])
                
            writer.writerow([])
            writer.writerow([])

    print(f"分块对比报告已生成: {out_file}")

if __name__ == '__main__':
    main()