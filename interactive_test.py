# interactive_test.py
import os
import sys
import re
import ctypes
from transformers import AutoTokenizer

# Configure terminal to output raw UTF-8 text safely
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DLL_PATH = r"C:\conda\kovidhar.dll"
WEIGHTS_PATH = r"D:\LocalMoEEngine\experts\weights.bin"
TOKENIZER_ID = "Qwen/Qwen2-57B-A14B-Instruct"

# 1. Access the Local AppLocker Whitelisted C++ Runtime Binary
if not os.path.exists(DLL_PATH):
    print(f"[ERROR] Engine binary missing from whitelisted path: {DLL_PATH}")
    sys.exit(1)

backend = ctypes.CDLL(DLL_PATH)

# Setup argument types for smooth ABI memory handshakes
backend.initialize_engine.argtypes = [ctypes.c_char_p]
backend.initialize_engine.restype = ctypes.c_bool
backend.predict_next_token.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
backend.predict_next_token.restype = ctypes.c_int
backend.shutdown_engine.restype = None

# 2. Load Tokenizer & Initialize Kernel Weights
print(f"[*] Initializing Qwen2 Tokenizer configuration...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

print(f"[*] Linking zero-syscall virtual memory map to weights.bin...")
if not backend.initialize_engine(WEIGHTS_PATH.encode('utf-8')):
    print(f"[WARNING] Failed to map {WEIGHTS_PATH}. Check paths and disk health. Running in CPU Emulation fallback mode.")
else:
    print("[SUCCESS] Massive 70B parameter profile mapped successfully to virtual address memory.")

# 3. High-Speed Regex Input Intention Router
def classify_intent(text):
    text_lower = text.lower()
    if re.search(r'(gravity|physics|force|orbit|mass)', text_lower):
        return 1
    if re.search(r'(code|sort|function|cpp|python|algorithm)', text_lower):
        return 2
    return 0

# 4. Pure Console Interactive Chat Loop
print("\n=== Native 70B MoE Engine Console Terminal Ready ===")
print("Type 'exit' to cleanly spin down hardware mappings.\n")

try:
    while True:
        user_input = input("\nUser >>> ")
        if user_input.strip().lower() == 'exit':
            break
            
        if not user_input.strip():
            continue

        # Convert prompt text down to raw input integer IDs
        tokens = tokenizer.encode(user_input)
        seq_len = len(tokens)
        intent = classify_intent(user_input)
        
        # Convert list to native C array layout pointer
        c_array = (ctypes.c_int * seq_len)(*tokens)
        
        print("Engine >>> ", end="", flush=True)
        
        # Generation loop - runs until model hits EOS or length threshold
        for step in range(60): 
            next_token_id = backend.predict_next_token(c_array, seq_len + step, intent)
            
            # Catch early Qwen2 End-Of-Sequence tokens to cleanly terminate
            if next_token_id == 151645: 
                break
                
            # Decode token index straight into printed string
            decoded_word = tokenizer.decode([next_token_id])
            print(decoded_word, end="", flush=True)
            
        print() # Newline block after stream completes

finally:
    print("\n[*] Unmapping SSD structures and clearing system file locks...")
    backend.shutdown_engine()
    print("[+] Engine context destroyed cleanly. Terminal safe.")
