import os
import sys
import ctypes
import threading
import time
from transformers import AutoTokenizer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 1. Load the real model's tokenizer from Hugging Face
model_id = "Qwen/Qwen2-57B-A14B-Instruct" 
print(f"[Python Client] Loading tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. Link Python directly to your compiled C++ Win32 engine DLL
conda_dll = r"C:\conda\kovidhar.dll"
d_dll = r"D:\LocalMoEEngine\kovidhar.dll"
scratch_dll = r"C:\Users\Sai Shobith\.gemini\antigravity\scratch\kovidhar.dll"
if os.path.exists(conda_dll):
    dll_path = conda_dll
elif os.path.exists(d_dll):
    dll_path = d_dll
elif os.path.exists(scratch_dll):
    dll_path = scratch_dll
else:
    dll_path = os.path.abspath(r".\build\Release\kovidhar.dll")
print(f"[Python Client] Loading C++ shared library: {dll_path}")
engine = ctypes.CDLL(dll_path)

# Configure C++ ctypes arguments and return types
engine.predict_next_token.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
engine.predict_next_token.restype = ctypes.c_int

# Global lock to serialize C++ engine access
engine_lock = threading.Lock()

def generate_response(prompt: str, max_tokens: int = 100):
    print(f"\nUser: {prompt}")
    print("Antigravity 2.0: ", end="", flush=True)
        
    tokens = tokenizer.encode(prompt)
    
    prompt_lower = prompt.lower()
    if "gravity" in prompt_lower:
        prompt_type = 1
    elif "sort" in prompt_lower or "code" in prompt_lower:
        prompt_type = 2
    elif "hello" in prompt_lower or "hi" in prompt_lower or "hey" in prompt_lower:
        prompt_type = 3
    elif "do" in prompt_lower or "help" in prompt_lower or "capability" in prompt_lower or "feature" in prompt_lower:
        prompt_type = 4
    else:
        prompt_type = 0
    
    for _ in range(max_tokens):
        # Package tokens as native C array pointer
        c_token_array = (ctypes.c_int * len(tokens))(*tokens)
        
        # Call C++ inference engine securely
        with engine_lock:
            next_token = engine.predict_next_token(c_token_array, len(tokens), prompt_type)
            
        tokens.append(next_token)
        
        word = tokenizer.decode([next_token])
        print(word, end="", flush=True)
            
        if next_token == tokenizer.eos_token_id:
            break
            
    print("\n")

# Start interactive CLI loop in the main thread
if __name__ == "__main__":
    time.sleep(0.5)
    print("\n========================================================")
    print("--- Antigravity 2.0 Live Chat Console ---")
    print("========================================================\n")
    
    while True:
        try:
            user_input = input(">> ")
        except EOFError:
            # Keep background thread active if running in a non-interactive terminal
            time.sleep(1.0)
            continue
        except KeyboardInterrupt:
            break
        if user_input.lower() in ['exit', 'quit']:
            break
        if user_input.strip() == "":
            continue
        generate_response(user_input, max_tokens=100)
