import os
import json
import torch
import numpy as np
from moe_model import SimpleMoeModel
from safetensors.torch import load_file

def main():
    print("[Weight Packer] Loading Tiny Qwen2MoE safetensors...")
    
    output_dir = "models/tiny-qwen2-moe"
    config_path = os.path.join(output_dir, "config.json")
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    
    if not os.path.exists(config_path) or not os.path.exists(safetensors_path):
        print("[Weight Packer ERROR] Model files not found. Run model_generator.py first.")
        return
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    state_dict = load_file(safetensors_path)
    
    # 1. Build weights.bin with 64-byte alignment for all tensors
    print("[Weight Packer] Packaging weights into weights.bin with 64-byte alignment...")
    binary_data = bytearray()
    current_offset = 0
    weights_meta = {}
    
    for name, tensor in state_dict.items():
        # Make tensor continuous in CPU memory
        np_arr = tensor.float().numpy().copy()
        tensor_bytes = np_arr.tobytes()
        
        # Calculate padding to 64-byte boundary
        padding = (64 - (current_offset % 64)) % 64
        if padding > 0:
            binary_data.extend(b'\0' * padding)
            current_offset += padding
            
        offset = current_offset
        size = len(tensor_bytes)
        
        binary_data.extend(tensor_bytes)
        current_offset += size
        
        weights_meta[name] = {
            "offset": offset,
            "size": size,
            "shape": list(np_arr.shape)
        }
        
    bin_path = os.path.join(output_dir, "weights.bin")
    with open(bin_path, "wb") as f:
        f.write(binary_data)
    print(f"[Weight Packer] Wrote packed binary to {bin_path} ({len(binary_data)} bytes)")
    
    meta_path = os.path.join(output_dir, "weights_meta.json")
    with open(meta_path, "w") as f:
        json.dump(weights_meta, f, indent=4)
    print(f"[Weight Packer] Wrote weights metadata to {meta_path}")
    
    # 2. Run PyTorch forward pass for a single token to generate reference dumps
    print("[Weight Packer] Instantiating model for reference run...")
    model = SimpleMoeModel(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        num_layers=config["num_hidden_layers"],
        num_heads=config["num_attention_heads"],
        num_experts=config["num_experts"],
        num_experts_per_tok=config["num_experts_per_tok"]
    )
    model.load_state_dict(state_dict)
    model.eval()
    
    # Single token forward pass using token ID 42 (seed input)
    input_token_id = 42
    input_ids = torch.tensor([input_token_id], dtype=torch.long)
    
    with torch.no_grad():
        logits, layer_dumps = model(input_ids)
        embedding_output = model.embed_tokens(input_ids)
        final_norm_output = model.norm(layer_dumps[-1]["final_output"])
        
    print("[Weight Packer] Generating intermediate layer tensor dumps...")
    dumps_dir = os.path.join(output_dir, "reference_dumps")
    os.makedirs(dumps_dir, exist_ok=True)
    
    # Dump embedding layer reference (Stage A)
    embed_dump_path = os.path.join(dumps_dir, "embedding_dump.npz")
    np.savez(embed_dump_path, 
             input_token_id=np.array([input_token_id]),
             embedding_output=embedding_output.numpy())
    
    # Dump layer-by-layer references (Stage B to F)
    for i, dump in enumerate(layer_dumps):
        dump_path = os.path.join(dumps_dir, f"layer_dump_{i}.npz")
        np.savez(dump_path,
                 hidden_state=dump["hidden_state"].numpy(),
                 attention_output=dump["attention_output"].numpy(),
                 router_logits=dump["router_logits"].numpy(),
                 topk_indices=dump["topk_indices"].numpy(),
                 expert_outputs=dump["expert_outputs"].numpy(), # Weighted expert FFN sum
                 final_output=dump["final_output"].numpy())
        
    # Dump final output reference (Norm and Logits)
    final_dump_path = os.path.join(dumps_dir, "final_dump.npz")
    np.savez(final_dump_path,
             final_norm_input=layer_dumps[-1]["final_output"].numpy(),
             final_norm_output=final_norm_output.numpy(),
             logits=logits.numpy())
             
    print(f"[Weight Packer] All reference tensor dumps written to {dumps_dir}")

if __name__ == "__main__":
    main()
