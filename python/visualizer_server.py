import os
import sys
import time
import json
import random
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# Shared global simulation state
def load_real_metrics():
    metrics = {
        "case1": {
            "throughput": 3.5,
            "hit_rate": 58.3,
            "ram_usage_mb": 12200.0,
            "measured_pcie_bw": 1500.0
        },
        "case2": {
            "throughput": 45.0,
            "hit_rate": 99.2,
            "ram_usage_mb": 18400.0,
            "measured_pcie_bw": 3200.0
        }
    }
    
    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "profiling_report.txt")
    if not os.path.exists(report_path):
        report_path = "profiling_report.txt"
        
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            scenarios = content.split("Scenario: ")
            for sc in scenarios[1:]:
                lines = sc.split("\n")
                name = lines[0].strip()
                
                def get_float(label):
                    for l in lines:
                        if label in l:
                            parts = l.split(":")
                            if len(parts) > 1:
                                val_str = "".join(c for c in parts[1] if c.isdigit() or c == '.')
                                if val_str:
                                    return float(val_str)
                    return None
                
                if "Cold Cache" in name:
                    tput = get_float("Token Throughput")
                    hr = get_float("Cache Hit Rate")
                    ram = get_float("Peak RAM Usage")
                    bw = get_float("Measured PCIe Bandwidth")
                    if tput is not None: metrics["case1"]["throughput"] = tput
                    if hr is not None: metrics["case1"]["hit_rate"] = hr
                    if ram is not None: metrics["case1"]["ram_usage_mb"] = ram
                    if bw is not None: metrics["case1"]["measured_pcie_bw"] = bw
                elif "High Expert Reuse" in name or "Warm Cache" in name or "Scenario C" in name:
                    tput = get_float("Token Throughput")
                    hr = get_float("Cache Hit Rate")
                    ram = get_float("Peak RAM Usage")
                    bw = get_float("Measured PCIe Bandwidth")
                    if tput is not None: metrics["case2"]["throughput"] = tput
                    if hr is not None: metrics["case2"]["hit_rate"] = hr
                    if ram is not None: metrics["case2"]["ram_usage_mb"] = ram
                    if bw is not None: metrics["case2"]["measured_pcie_bw"] = bw
            print(f"[Visualizer] Successfully loaded real metrics from {report_path}:")
            print(f"  Case 1: Throughput={metrics['case1']['throughput']:.2f} t/s, Hit Rate={metrics['case1']['hit_rate']:.2f}%, PCIe BW={metrics['case1']['measured_pcie_bw']:.2f} MB/s")
            print(f"  Case 2: Throughput={metrics['case2']['throughput']:.2f} t/s, Hit Rate={metrics['case2']['hit_rate']:.2f}%, PCIe BW={metrics['case2']['measured_pcie_bw']:.2f} MB/s")
        except Exception as e:
            print(f"[Visualizer WARNING] Failed to parse profiling report: {e}. Using baseline values.")
    else:
        print(f"[Visualizer] profiling_report.txt not found. Using baseline values.")
        
    return metrics

# Shared global simulation state
class SimulationState:
    def __init__(self):
        self.lock = threading.Lock()
        self.metrics = load_real_metrics()
        self.case1 = {
            "active_layer": 0,
            "is_stalled": False,
            "stall_layer": 0,
            "throughput": self.metrics["case1"]["throughput"],
            "hit_rate": self.metrics["case1"]["hit_rate"],
            "ssd_read_mb": 0.0,
            "ram_usage_mb": self.metrics["case1"]["ram_usage_mb"],
            "elapsed_sec": 0.0,
            "status_message": "Initializing...",
            "frames_left": 0
        }
        self.case2 = {
            "active_layer": 0,
            "throughput": self.metrics["case2"]["throughput"],
            "hit_rate": self.metrics["case2"]["hit_rate"],
            "ssd_read_mb": self.metrics["case2"]["measured_pcie_bw"],
            "ram_usage_mb": self.metrics["case2"]["ram_usage_mb"],
            "elapsed_sec": 0.0,
            "prefetch_transfers": [],
            "status_message": "Initializing..."
        }

g_sim = SimulationState()

def run_simulation():
    start_time = time.time()
    frame_counter = 0
    
    # Case 1 specific counters
    case1_layer_frames = 0
    case1_stall_frames_left = 0
    
    # Case 2 specific counters
    case2_layer_frames = 0
    
    # Initialize Case 2 prefetch transfers
    with g_sim.lock:
        g_sim.case2["prefetch_transfers"] = [
            {"layer": 1, "expert": (1 * 3 + 7) % 8, "progress": 70.0},
            {"layer": 2, "expert": (2 * 3 + 7) % 8, "progress": 40.0},
            {"layer": 3, "expert": (3 * 3 + 7) % 8, "progress": 10.0}
        ]
        
    while True:
        time.sleep(0.1)  # 10 Hz Loop (100ms per step - human-readable speeds!)
        frame_counter += 1
        elapsed = time.time() - start_time
        
        with g_sim.lock:
            g_sim.case1["elapsed_sec"] = elapsed
            g_sim.case2["elapsed_sec"] = elapsed
            
            # ==========================================
            # UPDATE CASE 1 (Synchronous, disk stalled)
            # ==========================================
            if g_sim.case1["is_stalled"]:
                case1_stall_frames_left -= 1
                g_sim.case1["throughput"] = 0.17
                g_sim.case1["ssd_read_mb"] = g_sim.metrics["case1"]["measured_pcie_bw"] + random.uniform(-10.0, 10.0)
                g_sim.case1["hit_rate"] = g_sim.metrics["case1"]["hit_rate"] * 0.9 + random.uniform(-0.5, 0.5)
                g_sim.case1["frames_left"] = case1_stall_frames_left
                
                # Calculate SSD load progress percentage (40 frames total)
                pct = int((40 - case1_stall_frames_left) / 40 * 100)
                exp_id = (g_sim.case1["stall_layer"] * 3 + 7) % 8
                g_sim.case1["status_message"] = f"SSD STALL! Thread blocked loading Expert {exp_id} weights into RAM ({pct}% loaded)..."
                
                if case1_stall_frames_left <= 0:
                    g_sim.case1["is_stalled"] = False
                    g_sim.case1["ssd_read_mb"] = 0.0
                    case1_layer_frames = 0  # Resume layer processing
            else:
                case1_layer_frames += 1
                g_sim.case1["throughput"] = g_sim.metrics["case1"]["throughput"] + random.uniform(-0.3, 0.3)
                g_sim.case1["hit_rate"] = g_sim.metrics["case1"]["hit_rate"] + random.uniform(-0.5, 0.5)
                g_sim.case1["ssd_read_mb"] = 0.0
                g_sim.case1["frames_left"] = 0
                
                # Set step tags depending on compute sub-phase (1.5 seconds per layer)
                exp_id = (g_sim.case1["active_layer"] * 3 + 7) % 8
                if case1_layer_frames <= 5:
                    g_sim.case1["status_message"] = f"L{g_sim.case1['active_layer']}: Computing Multi-Head Self-Attention..."
                elif case1_layer_frames <= 10:
                    g_sim.case1["status_message"] = f"L{g_sim.case1['active_layer']}: Evaluating Gating Router -> Dispatching active parameters..."
                else:
                    g_sim.case1["status_message"] = f"L{g_sim.case1['active_layer']}: Executing FFN matrix logic for Expert {exp_id}..."
                
                if case1_layer_frames >= 15:
                    case1_layer_frames = 0
                    g_sim.case1["active_layer"] = (g_sim.case1["active_layer"] + 1) % 12
                    
                    # Trigger a stall periodically (layers 2, 5, 8, 11)
                    if g_sim.case1["active_layer"] in (2, 5, 8, 11):
                        g_sim.case1["is_stalled"] = True
                        g_sim.case1["stall_layer"] = g_sim.case1["active_layer"]
                        case1_stall_frames_left = 40  # 4 seconds stall
                        
            g_sim.case1["ram_usage_mb"] = g_sim.metrics["case1"]["ram_usage_mb"] + random.uniform(-5.0, 5.0)
            
            # ==========================================
            # UPDATE CASE 2 (Asynchronous prefetch)
            # ==========================================
            case2_layer_frames += 1
            g_sim.case2["throughput"] = g_sim.metrics["case2"]["throughput"] + random.uniform(-2.0, 2.0)
            g_sim.case2["hit_rate"] = g_sim.metrics["case2"]["hit_rate"] + random.uniform(-0.1, 0.1)
            g_sim.case2["ssd_read_mb"] = g_sim.metrics["case2"]["measured_pcie_bw"] + random.uniform(-15.0, 15.0)
            g_sim.case2["ram_usage_mb"] = g_sim.metrics["case2"]["ram_usage_mb"] + random.uniform(-10.0, 10.0)
            
            # Set sub-step activity message (1.5 seconds per layer)
            comp_exp = (g_sim.case2["active_layer"] * 3 + 7) % 8
            if case2_layer_frames <= 5:
                g_sim.case2["status_message"] = f"L{g_sim.case2['active_layer']}: Computing Attention & Gating concurrently..."
            else:
                g_sim.case2["status_message"] = f"L{g_sim.case2['active_layer']}: Running Expert {comp_exp} (Cache Hit: weights already resident)..."
            
            # Advance active layer every 15 frames (1.5s) to stay in sync with Case 1
            if case2_layer_frames >= 15:
                case2_layer_frames = 0
                g_sim.case2["active_layer"] = (g_sim.case2["active_layer"] + 1) % 12
                
                if g_sim.case2["prefetch_transfers"]:
                    g_sim.case2["prefetch_transfers"].pop(0)
                
                next_prefetch_layer = (g_sim.case2["active_layer"] + 3) % 12
                next_prefetch_expert = (next_prefetch_layer * 3 + 7) % 8
                g_sim.case2["prefetch_transfers"].append({
                    "layer": next_prefetch_layer,
                    "expert": next_prefetch_expert,
                    "progress": 0.0
                })
                
            # Prefetch progress bar increments over 10 frames (1.0 second)
            for idx, pf in enumerate(g_sim.case2["prefetch_transfers"]):
                if idx == 0:
                    pf["progress"] = min(100.0, pf["progress"] + 10.0)
                elif idx == 1:
                    pf["progress"] = min(100.0, pf["progress"] + 8.0)
                else:
                    pf["progress"] = min(100.0, pf["progress"] + 5.0)

# Custom HTTP request handler to handle visualizer requests
class VisualizerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.serve_index()
        elif self.path == "/stream":
            self.serve_stream()
        else:
            self.send_error(404, "File Not Found")
            
    def log_message(self, format, *args):
        # Suppress logging request frames to stdout for raw performance
        pass

    def serve_index(self):
        try:
            filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "index.html")
            if not os.path.exists(filepath):
                filepath = os.path.join("web", "index.html")
            
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content.encode('utf-8'))))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        except Exception as e:
            self.send_error(500, f"Internal Server Error: {e}")

    def serve_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        
        while True:
            with g_sim.lock:
                frame = {
                    "case1": g_sim.case1.copy(),
                    "case2": {
                        "active_layer": g_sim.case2["active_layer"],
                        "throughput": g_sim.case2["throughput"],
                        "hit_rate": g_sim.case2["hit_rate"],
                        "ssd_read_mb": g_sim.case2["ssd_read_mb"],
                        "ram_usage_mb": g_sim.case2["ram_usage_mb"],
                        "elapsed_sec": g_sim.case2["elapsed_sec"],
                        "status_message": g_sim.case2["status_message"],
                        "lookahead": [{"layer": pf["layer"], "expert": pf["expert"]} for pf in g_sim.case2["prefetch_transfers"]],
                        "prefetch_transfers": g_sim.case2["prefetch_transfers"],
                        "compute_active": {
                            "layer": g_sim.case2["active_layer"],
                            "expert": (g_sim.case2["active_layer"] * 3 + 7) % 8
                        }
                    }
                }
            
            data = f"data: {json.dumps(frame)}\n\n"
            try:
                self.wfile.write(data.encode('utf-8'))
                self.wfile.flush()
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                break
                
            time.sleep(0.1) # 10 Hz telemetry frame stream rate

def main():
    sim_thread = threading.Thread(target=run_simulation)
    sim_thread.daemon = True
    sim_thread.start()
    
    server = HTTPServer(("localhost", 8082), VisualizerHandler)
    print("Python MoE Visualizer Server running natively on Windows host...")
    print("URL: http://localhost:8082/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print("\nShutting down server.")

if __name__ == "__main__":
    main()
