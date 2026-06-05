import os
import json
import torch
from moe_model import SimpleMoeModel
from safetensors.torch import save_file

def main():
    print("[Model Generator] Initializing Tiny Qwen2MoE configuration...")
    
    # Configure tiny model parameters (based on user request)
    config = {
        "vocab_size": 1000,
        "hidden_size": 512,
        "intermediate_size": 2048,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 8, # Keep equal to attention heads for standard attention
        "num_experts": 16,
        "num_experts_per_tok": 2,
        "rms_norm_eps": 1e-5
    }
    
    output_dir = "models/tiny-qwen2-moe"
    os.makedirs(output_dir, exist_ok=True)
    
    # Instantiate the model with a fixed seed for reproducible validation
    torch.manual_seed(42)
    model = SimpleMoeModel(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        num_layers=config["num_hidden_layers"],
        num_heads=config["num_attention_heads"],
        num_experts=config["num_experts"],
        num_experts_per_tok=config["num_experts_per_tok"]
    )
    
    # Save HF-style configuration file
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"[Model Generator] Wrote configuration to {config_path}")
    
    # Extract state dict and serialize to safetensors
    state_dict = model.state_dict()
    
    # Convert parameters to float32 to ensure high-precision baseline checks
    float32_state_dict = {k: v.float() for k, v in state_dict.items()}
    
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(float32_state_dict, safetensors_path)
    print(f"[Model Generator] Saved safetensors weights to {safetensors_path}")
    
    # Run a quick dummy forward pass to make sure model functions correctly
    dummy_input = torch.tensor([12, 45, 99, 102], dtype=torch.long)
    with torch.no_grad():
        logits, dumps = model(dummy_input)
    print(f"[Model Generator] Model verification successful. Logits shape: {logits.shape}")

if __name__ == "__main__":
    main()
