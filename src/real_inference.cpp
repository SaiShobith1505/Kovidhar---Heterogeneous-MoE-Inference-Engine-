#include <windows.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <utility>

// Structure mapping to ctypes definition for per-layer weights
struct LayerWeights {
    float* input_layernorm_weight;
    float* q_proj_weight;
    float* k_proj_weight;
    float* v_proj_weight;
    float* o_proj_weight;
    float* post_attention_layernorm_weight;
    float* gate_weight;
    
    // Expert weights (16 experts)
    float* expert_gate_proj[16];
    float* expert_up_proj[16];
    float* expert_down_proj[16];
};

extern "C" {
    // Stage A: Embedding lookup
    __declspec(dllexport) void test_embedding_lookup(
        const int* token_ids, const float* embed_table, float* output, 
        int seq_len, int hidden_size
    ) {
        for (int s = 0; s < seq_len; ++s) {
            int tid = token_ids[s];
            std::memcpy(output + s * hidden_size, embed_table + tid * hidden_size, hidden_size * sizeof(float));
        }
    }

    // Stage B: RMSNorm
    __declspec(dllexport) void test_rms_norm(
        const float* input, const float* weight, float* output, 
        int seq_len, int dim, float eps
    ) {
        for (int s = 0; s < seq_len; ++s) {
            double sum_sq = 0.0;
            for (int i = 0; i < dim; ++i) {
                double val = (double)input[s * dim + i];
                sum_sq += val * val;
            }
            double rsqrt = 1.0 / std::sqrt(sum_sq / dim + (double)eps);
            for (int i = 0; i < dim; ++i) {
                output[s * dim + i] = (float)((double)input[s * dim + i] * rsqrt * (double)weight[i]);
            }
        }
    }

    // Stage C: Gating Router (Softmax & Top-K selection)
    __declspec(dllexport) void test_gating_router(
        const float* input, const float* gate_weight, float* out_logits, 
        int* out_indices, float* out_weights, 
        int seq_len, int hidden_size, int num_experts, int top_k
    ) {
        for (int s = 0; s < seq_len; ++s) {
            std::vector<float> logits(num_experts);
            float max_logit = -1e20f;
            
            // 1. Matrix multiplication: logits = input * gate_weight^T
            for (int e = 0; e < num_experts; ++e) {
                float sum = 0.0f;
                for (int i = 0; i < hidden_size; ++i) {
                    sum += input[s * hidden_size + i] * gate_weight[e * hidden_size + i];
                }
                logits[e] = sum;
                if (sum > max_logit) max_logit = sum;
            }
            
            // Output logits for debugging
            for (int e = 0; e < num_experts; ++e) {
                out_logits[s * num_experts + e] = logits[e];
            }
            
            // 2. Compute Softmax
            std::vector<float> softmax_weights(num_experts);
            float sum_exp = 0.0f;
            for (int e = 0; e < num_experts; ++e) {
                softmax_weights[e] = std::exp(logits[e] - max_logit);
                sum_exp += softmax_weights[e];
            }
            for (int e = 0; e < num_experts; ++e) {
                softmax_weights[e] /= sum_exp;
            }
            
            // 3. Find Top-K indices and values
            std::vector<std::pair<float, int>> rank(num_experts);
            for (int e = 0; e < num_experts; ++e) {
                rank[e] = {softmax_weights[e], e};
            }
            
            // Stable sort by weights descending, using index ascending as fallback
            std::sort(rank.begin(), rank.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                if (std::abs(a.first - b.first) < 1e-9f) {
                    return a.second < b.second;
                }
                return a.first > b.first;
            });
            
            // 4. Normalize weights of top K experts
            float top_k_sum = 0.0f;
            for (int k = 0; k < top_k; ++k) {
                top_k_sum += rank[k].first;
            }
            for (int k = 0; k < top_k; ++k) {
                out_indices[s * top_k + k] = rank[k].second;
                out_weights[s * top_k + k] = rank[k].first / top_k_sum;
            }
        }
    }

    // Stage D: Expert FFN
    __declspec(dllexport) void test_expert_ffn(
        const float* input, const float* w1, const float* w2, const float* w3, 
        float* output, int seq_len, int hidden_size, int intermediate_size
    ) {
        for (int s = 0; s < seq_len; ++s) {
            std::vector<float> h1(intermediate_size);
            
            // Compute h1 = SiLU(W1 * x) * (W3 * x)
            for (int j = 0; j < intermediate_size; ++j) {
                float sum1 = 0.0f;
                float sum3 = 0.0f;
                for (int i = 0; i < hidden_size; ++i) {
                    sum1 += input[s * hidden_size + i] * w1[j * hidden_size + i];
                    sum3 += input[s * hidden_size + i] * w3[j * hidden_size + i];
                }
                float silu = sum1 * (1.0f / (1.0f + std::exp(-sum1)));
                h1[j] = silu * sum3;
            }
            
            // Compute output = W2 * h1
            for (int i = 0; i < hidden_size; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < intermediate_size; ++j) {
                    sum += h1[j] * w2[i * intermediate_size + j];
                }
                output[s * hidden_size + i] = sum;
            }
        }
    }

    // Stage E: Attention
    __declspec(dllexport) void test_attention(
        const float* input, 
        const float* q_proj, const float* k_proj, const float* v_proj, const float* o_proj, 
        float* output, int seq_len, int hidden_size, int num_heads
    ) {
        int head_dim = hidden_size / num_heads;
        
        std::vector<float> q(seq_len * hidden_size);
        std::vector<float> k(seq_len * hidden_size);
        std::vector<float> v(seq_len * hidden_size);
        
        // Compute Q, K, V projections
        for (int s = 0; s < seq_len; ++s) {
            for (int i = 0; i < hidden_size; ++i) {
                float sum_q = 0.0f, sum_k = 0.0f, sum_v = 0.0f;
                for (int j = 0; j < hidden_size; ++j) {
                    sum_q += input[s * hidden_size + j] * q_proj[i * hidden_size + j];
                    sum_k += input[s * hidden_size + j] * k_proj[i * hidden_size + j];
                    sum_v += input[s * hidden_size + j] * v_proj[i * hidden_size + j];
                }
                q[s * hidden_size + i] = sum_q;
                k[s * hidden_size + i] = sum_k;
                v[s * hidden_size + i] = sum_v;
            }
        }
        
        std::vector<float> context(seq_len * hidden_size, 0.0f);
        float scale = 1.0f / std::sqrt((float)head_dim);
        
        // Compute scaled dot-product attention per head
        for (int h = 0; h < num_heads; ++h) {
            for (int s1 = 0; s1 < seq_len; ++s1) {
                std::vector<float> scores(seq_len);
                float max_score = -1e20f;
                for (int s2 = 0; s2 < seq_len; ++s2) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        float q_val = q[s1 * hidden_size + h * head_dim + d];
                        float k_val = k[s2 * hidden_size + h * head_dim + d];
                        dot += q_val * k_val;
                    }
                    scores[s2] = dot * scale;
                    if (scores[s2] > max_score) max_score = scores[s2];
                }
                
                // Softmax
                float sum_exp = 0.0f;
                std::vector<float> attn_weights(seq_len);
                for (int s2 = 0; s2 < seq_len; ++s2) {
                    attn_weights[s2] = std::exp(scores[s2] - max_score);
                    sum_exp += attn_weights[s2];
                }
                for (int s2 = 0; s2 < seq_len; ++s2) {
                    attn_weights[s2] /= sum_exp;
                }
                
                // Weighted sum over V
                for (int d = 0; d < head_dim; ++d) {
                    float sum_c = 0.0f;
                    for (int s2 = 0; s2 < seq_len; ++s2) {
                        sum_c += attn_weights[s2] * v[s2 * hidden_size + h * head_dim + d];
                    }
                    context[s1 * hidden_size + h * head_dim + d] = sum_c;
                }
            }
        }
        
        // Output projection
        for (int s = 0; s < seq_len; ++s) {
            for (int i = 0; i < hidden_size; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < hidden_size; ++j) {
                    sum += context[s * hidden_size + j] * o_proj[i * hidden_size + j];
                }
                output[s * hidden_size + i] = sum;
            }
        }
    }

    // Stage F: Full Transformer Layer Execution
    __declspec(dllexport) void run_transformer_layer(
        const float* hidden_states, const LayerWeights* weights,
        int seq_len, int hidden_size, int intermediate_size, int num_heads, int num_experts, int top_k, float eps,
        float* out_attn_out, float* out_router_logits, int* out_topk_indices, float* out_expert_outputs, float* out_mlp_out, float* out_final
    ) {
        // 1. input_layernorm
        std::vector<float> norm_x(seq_len * hidden_size);
        test_rms_norm(hidden_states, weights->input_layernorm_weight, norm_x.data(), seq_len, hidden_size, eps);
        
        // 2. self_attn
        test_attention(norm_x.data(), weights->q_proj_weight, weights->k_proj_weight, weights->v_proj_weight, weights->o_proj_weight,
                       out_attn_out, seq_len, hidden_size, num_heads);
        
        // 3. residual add
        std::vector<float> x_temp(seq_len * hidden_size);
        for (int i = 0; i < seq_len * hidden_size; ++i) {
            x_temp[i] = hidden_states[i] + out_attn_out[i];
        }
        
        // 4. post_attention_layernorm
        std::vector<float> norm_x2(seq_len * hidden_size);
        test_rms_norm(x_temp.data(), weights->post_attention_layernorm_weight, norm_x2.data(), seq_len, hidden_size, eps);
        
        // 5. gating router
        std::vector<float> gating_weights(seq_len * top_k);
        test_gating_router(norm_x2.data(), weights->gate_weight, out_router_logits, out_topk_indices, gating_weights.data(),
                           seq_len, hidden_size, num_experts, top_k);
        
        // 6. expert_ffn
        std::fill_n(out_expert_outputs, seq_len * hidden_size, 0.0f);
        std::fill_n(out_mlp_out, seq_len * hidden_size, 0.0f);
        
        for (int s = 0; s < seq_len; ++s) {
            for (int k = 0; k < top_k; ++k) {
                int exp_idx = out_topk_indices[s * top_k + k];
                float weight = gating_weights[s * top_k + k];
                
                // Run FFN of selected expert on token s
                std::vector<float> token_input(hidden_size);
                std::memcpy(token_input.data(), norm_x2.data() + s * hidden_size, hidden_size * sizeof(float));
                
                std::vector<float> token_output(hidden_size);
                test_expert_ffn(
                    token_input.data(), 
                    (float*)weights->expert_gate_proj[exp_idx], 
                    (float*)weights->expert_down_proj[exp_idx], 
                    (float*)weights->expert_up_proj[exp_idx], 
                    token_output.data(), 
                    1, hidden_size, intermediate_size
                );
                
                // Accumulate results
                for (int i = 0; i < hidden_size; ++i) {
                    // Accumulate unweighted outputs for Stage D verification
                    out_expert_outputs[s * hidden_size + i] += token_output[i];
                    // Accumulate weighted outputs for Stage F final path
                    out_mlp_out[s * hidden_size + i] += weight * token_output[i];
                }
            }
        }
        
        // 7. residual add
        for (int i = 0; i < seq_len * hidden_size; ++i) {
            out_final[i] = x_temp[i] + out_mlp_out[i];
        }
    }

    // Final Norm & LM Head
    __declspec(dllexport) void run_final_layer(
        const float* hidden_states, const float* norm_weight, const float* lm_head_weight,
        int seq_len, int hidden_size, int vocab_size, float eps,
        float* out_norm, float* out_logits
    ) {
        // 1. norm
        test_rms_norm(hidden_states, norm_weight, out_norm, seq_len, hidden_size, eps);
        
        // 2. lm_head
        for (int s = 0; s < seq_len; ++s) {
            for (int v = 0; v < vocab_size; ++v) {
                float sum = 0.0f;
                for (int i = 0; i < hidden_size; ++i) {
                    sum += out_norm[s * hidden_size + i] * lm_head_weight[v * hidden_size + i];
                }
                out_logits[s * vocab_size + v] = sum;
            }
        }
    }
}
