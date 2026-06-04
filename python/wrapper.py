import os
import sys
import ctypes
from ctypes import c_void_p, c_char_p, c_uint32, c_uint64, c_int, c_double, POINTER

class MoeEngineWrapper:
    def __init__(self, expert_dir="./experts", num_experts=256, expert_size_bytes=100*1024*1024, num_layers=12, total_tokens=10, cache_size_bytes=8*1024*1024*1024):
        self.lib = self._load_library()
        self._declare_signatures()
        
        # Instantiate C++ Benchmark
        dir_bytes = expert_dir.encode('utf-8')
        self.benchmark_obj = self.lib.create_benchmark(
            dir_bytes,
            c_uint32(num_experts),
            c_uint64(expert_size_bytes),
            c_uint32(num_layers),
            c_uint32(total_tokens),
            c_uint64(cache_size_bytes)
        )
        
    def _load_library(self):
        # Look for the compiled library in common locations
        search_paths = [
            "./build/libmoe_engine.so",
            "../build/libmoe_engine.so",
            "./libmoe_engine.so",
            "./build/moe_engine.dll",
            "./build/Release/moe_engine.dll",
            "../build/Release/moe_engine.dll",
            "./moe_engine.dll",
            "/usr/local/lib/libmoe_engine.so"
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                try:
                    return ctypes.CDLL(os.path.abspath(path))
                except Exception as e:
                    print(f"[Python Wrapper] Failed to load library at {path}: {e}")
                    
        # Try system load
        try:
            if sys.platform == "win32":
                return ctypes.CDLL("moe_engine.dll")
            else:
                return ctypes.CDLL("libmoe_engine.so")
        except OSError:
            pass
            
        raise FileNotFoundError(
            "Could not find 'libmoe_engine.so' or 'moe_engine.dll'. Please compile the project using CMake."
        )
        
    def _declare_signatures(self):
        # void* create_benchmark(...)
        self.lib.create_benchmark.restype = c_void_p
        self.lib.create_benchmark.argtypes = [c_char_p, c_uint32, c_uint64, c_uint32, c_uint32, c_uint64]
        
        # void destroy_benchmark(void*)
        self.lib.destroy_benchmark.restype = None
        self.lib.destroy_benchmark.argtypes = [c_void_p]
        
        # void generate_assets(void*)
        self.lib.generate_assets.restype = None
        self.lib.generate_assets.argtypes = [c_void_p]
        
        # void cleanup_assets(void*)
        self.lib.cleanup_assets.restype = None
        self.lib.cleanup_assets.argtypes = [c_void_p]
        
        # void run_scenario_c(...)
        self.lib.run_scenario_c.restype = None
        self.lib.run_scenario_c.argtypes = [
            c_void_p,
            c_int,
            c_char_p,
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_uint64),
            POINTER(c_uint64),
            POINTER(c_uint64),
            POINTER(c_double)
        ]
        
    def generate_assets(self):
        print("[Python Wrapper] Triggering sparse assets generation...")
        self.lib.generate_assets(self.benchmark_obj)
        
    def cleanup_assets(self):
        print("[Python Wrapper] Triggering assets cleanup...")
        self.lib.cleanup_assets(self.benchmark_obj)
        
    def run_scenario(self, scenario_type: int, name: str):
        """
        Runs a scenario:
        0: COLD_CACHE
        1: WARM_CACHE
        2: HIGH_REUSE
        3: WORST_CASE_CHURN
        """
        exec_time = c_double(0.0)
        throughput = c_double(0.0)
        hits = c_uint64(0)
        misses = c_uint64(0)
        evictions = c_uint64(0)
        hit_rate = c_double(0.0)
        
        name_bytes = name.encode('utf-8')
        
        print(f"[Python Wrapper] Running scenario '{name}' from ctypes wrapper...")
        self.lib.run_scenario_c(
            self.benchmark_obj,
            c_int(scenario_type),
            name_bytes,
            ctypes.byref(exec_time),
            ctypes.byref(throughput),
            ctypes.byref(hits),
            ctypes.byref(misses),
            ctypes.byref(evictions),
            ctypes.byref(hit_rate)
        )
        
        return {
            "execution_time_s": exec_time.value,
            "throughput_tok_s": throughput.value,
            "hits": hits.value,
            "misses": misses.value,
            "evictions": evictions.value,
            "hit_rate_pct": hit_rate.value * 100.0
        }
        
    def __del__(self):
        if hasattr(self, "benchmark_obj") and self.benchmark_obj:
            self.lib.destroy_benchmark(self.benchmark_obj)
            self.benchmark_obj = None

if __name__ == "__main__":
    print("=== MoE Engine Python ctypes Wrapper Validation ===")
    
    # Simple check to verify we can initialize and run
    try:
        # We assume build/libmoe_engine.so exists if we run from the project root after compiling
        wrapper = MoeEngineWrapper(
            expert_dir="./experts",
            num_experts=16, # Use smaller config for quick Python verification
            expert_size_bytes=10*1024*1024, # 10 MB experts
            num_layers=4,
            total_tokens=5,
            cache_size_bytes=500*1024*1024 # 500 MB cache
        )
        
        wrapper.generate_assets()
        
        # Run Scenario C: High Reuse (Type 2)
        results = wrapper.run_scenario(2, "Scenario C (Python-invoked)")
        print("\n=== Results returned to Python ===")
        for key, val in results.items():
            print(f"  {key}: {val}")
            
        wrapper.cleanup_assets()
        print("\nPython validation complete.")
        
    except FileNotFoundError as e:
        print(f"\n[Validation Note] Shared library not loaded: {e}")
        print("Please build the C++ project using CMake first, then run python/wrapper.py.")
