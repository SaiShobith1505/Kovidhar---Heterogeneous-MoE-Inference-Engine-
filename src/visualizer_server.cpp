#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <sstream>
#include <fstream>
#include <cstring>
#include <random>
#include <mutex>
#include <algorithm>
#include <condition_variable>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#define MSG_NOSIGNAL 0
using ssize_t = int;
using socket_t = SOCKET;
using socklen_t = int;
#define CLOSE_SOCKET(s) closesocket(s)
#define IS_VALID_SOCKET(s) ((s) != INVALID_SOCKET)
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <signal.h>
using socket_t = int;
#define CLOSE_SOCKET(s) close(s)
#define IS_VALID_SOCKET(s) ((s) >= 0)
#endif

#include "telemetry_state.h"

// Define the global telemetry state
EngineTelemetry g_telemetry;

// Chat communication globals
std::mutex g_chat_mutex;
std::condition_variable g_chat_cv;
std::vector<std::string> g_pending_tokens;
bool g_chat_active = false;
std::string g_pending_prompt = "";

extern "C" {
    MOE_API const char* get_pending_prompt() {
        std::lock_guard<std::mutex> lock(g_chat_mutex);
        if (g_pending_prompt.empty()) return nullptr;
        
        static std::string static_prompt;
        static_prompt = g_pending_prompt;
        g_pending_prompt = "";
        return static_prompt.c_str();
    }

    MOE_API void send_generated_token(const char* token_text) {
        std::lock_guard<std::mutex> lock(g_chat_mutex);
        if (token_text) {
            g_pending_tokens.push_back(token_text);
        }
        g_chat_cv.notify_all();
    }

    MOE_API void mark_chat_complete() {
        std::lock_guard<std::mutex> lock(g_chat_mutex);
        g_chat_active = false;
        g_chat_cv.notify_all();
    }
}


struct Case1State {
    int active_layer = 0;
    bool is_stalled = false;
    int stall_layer = 0;
    double throughput = 3.5;
    double hit_rate = 58.3;
    double ssd_read_mb = 0.0;
    double ram_usage_mb = 12200.0;
    double elapsed_sec = 0.0;
    std::string status_message = "Initializing...";
    int frames_left = 0;
};

struct PrefetchTransfer {
    int layer;
    int expert;
    double progress;
};

struct Case2State {
    int active_layer = 0;
    double throughput = 45.0;
    double hit_rate = 99.2;
    double ssd_read_mb = 750.0;
    double ram_usage_mb = 18400.0;
    double elapsed_sec = 0.0;
    std::string status_message = "Initializing...";
    std::vector<PrefetchTransfer> prefetch_transfers;
};

struct SimulationState {
    std::mutex mtx;
    Case1State case1;
    Case2State case2;
} g_sim;

struct ConfiguredMetrics {
    double case1_throughput = 3.5;
    double case1_hit_rate = 58.3;
    double case1_ram_usage_mb = 12200.0;
    double case1_pcie_bw = 1500.0;

    double case2_throughput = 45.0;
    double case2_hit_rate = 99.2;
    double case2_ram_usage_mb = 18400.0;
    double case2_pcie_bw = 3200.0;
} g_metrics;

void load_real_metrics() {
    std::ifstream ifs("profiling_report.txt");
    if (!ifs.is_open()) {
        std::cout << "[Visualizer] profiling_report.txt not found. Using baseline values." << std::endl;
        return;
    }

    std::string line;
    std::string current_scenario = "";
    while (std::getline(ifs, line)) {
        if (line.find("Scenario:") != std::string::npos) {
            current_scenario = line;
        }

        auto get_value = [&](const std::string& label) -> double {
            size_t pos = line.find(label);
            if (pos != std::string::npos) {
                size_t colon = line.find(":", pos);
                if (colon != std::string::npos) {
                    std::string val_str = "";
                    for (size_t i = colon + 1; i < line.length(); ++i) {
                        if (std::isdigit(static_cast<unsigned char>(line[i])) || line[i] == '.') {
                            val_str += line[i];
                        }
                    }
                    if (!val_str.empty()) {
                        try {
                            return std::stod(val_str);
                        } catch (...) {}
                    }
                }
            }
            return -1.0;
        };

        if (current_scenario.find("Cold Cache") != std::string::npos) {
            double v = get_value("Token Throughput");
            if (v >= 0) g_metrics.case1_throughput = v;
            v = get_value("Cache Hit Rate");
            if (v >= 0) g_metrics.case1_hit_rate = v;
            v = get_value("Peak RAM Usage");
            if (v >= 0) g_metrics.case1_ram_usage_mb = v;
            v = get_value("Measured PCIe Bandwidth");
            if (v >= 0) g_metrics.case1_pcie_bw = v;
        } else if (current_scenario.find("High Expert Reuse") != std::string::npos ||
                   current_scenario.find("Warm Cache") != std::string::npos ||
                   current_scenario.find("Scenario C") != std::string::npos) {
            double v = get_value("Token Throughput");
            if (v >= 0) g_metrics.case2_throughput = v;
            v = get_value("Cache Hit Rate");
            if (v >= 0) g_metrics.case2_hit_rate = v;
            v = get_value("Peak RAM Usage");
            if (v >= 0) g_metrics.case2_ram_usage_mb = v;
            v = get_value("Measured PCIe Bandwidth");
            if (v >= 0) g_metrics.case2_pcie_bw = v;
        }
    }
    std::cout << "[Visualizer] Loaded real metrics from profiling_report.txt:" << std::endl;
    std::cout << "  Case 1: Throughput=" << g_metrics.case1_throughput << " t/s, Hit Rate=" << g_metrics.case1_hit_rate << "%, PCIe BW=" << g_metrics.case1_pcie_bw << " MB/s" << std::endl;
    std::cout << "  Case 2: Throughput=" << g_metrics.case2_throughput << " t/s, Hit Rate=" << g_metrics.case2_hit_rate << "%, PCIe BW=" << g_metrics.case2_pcie_bw << " MB/s" << std::endl;
}

void run_simulation() {
    auto start_time = std::chrono::steady_clock::now();
    int frame_counter = 0;
    
    // Case 1 specific counters
    int case1_layer_frames = 0;
    int case1_stall_frames_left = 0;
    
    // Case 2 specific counters
    int case2_layer_frames = 0;

    // Initialize Case 2 prefetch transfers
    {
        std::lock_guard<std::mutex> lock(g_sim.mtx);
        g_sim.case2.prefetch_transfers.push_back({1, (1 * 3 + 7) % 8, 70.0});
        g_sim.case2.prefetch_transfers.push_back({2, (2 * 3 + 7) % 8, 40.0});
        g_sim.case2.prefetch_transfers.push_back({3, (3 * 3 + 7) % 8, 10.0});
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.3, 0.3);
    std::uniform_real_distribution<> dis2(-2.0, 2.0);
    std::uniform_real_distribution<> dis_ssd(-15.0, 15.0);

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 10 Hz Loop (100ms steps)
        frame_counter++;
        
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();

        std::lock_guard<std::mutex> lock(g_sim.mtx);
        
        // Update elapsed time
        g_sim.case1.elapsed_sec = elapsed;
        g_sim.case2.elapsed_sec = elapsed;

        // ==========================================
        // UPDATE CASE 1 (Synchronous, disk stalled)
        // ==========================================
        if (g_sim.case1.is_stalled) {
            case1_stall_frames_left--;
            g_sim.case1.throughput = 0.17;
            g_sim.case1.ssd_read_mb = g_metrics.case1_pcie_bw + dis(gen) * 10.0;
            g_sim.case1.hit_rate = g_metrics.case1_hit_rate * 0.9 + dis(gen) * 0.5;
            g_sim.case1.frames_left = case1_stall_frames_left;
            
            // Calculate SSD page-in progress percentage (40 frames total)
            int pct = static_cast<int>((40 - case1_stall_frames_left) / 40.0 * 100.0);
            int exp_id = (g_sim.case1.stall_layer * 3 + 7) % 8;
            
            std::stringstream ss;
            ss << "SSD STALL! Thread blocked loading Expert " << exp_id 
               << " weights into RAM (" << pct << "% loaded)...";
            g_sim.case1.status_message = ss.str();
            
            if (case1_stall_frames_left <= 0) {
                g_sim.case1.is_stalled = false;
                g_sim.case1.ssd_read_mb = 0.0;
                case1_layer_frames = 0; // resume layer processing
            }
        } else {
            case1_layer_frames++;
            g_sim.case1.throughput = g_metrics.case1_throughput + dis(gen);
            g_sim.case1.hit_rate = g_metrics.case1_hit_rate + dis(gen) * 0.5;
            g_sim.case1.ssd_read_mb = 0.0;
            g_sim.case1.frames_left = 0;

            // Log messages depending on layer sub-step (15 frames = 1.5s per layer)
            int exp_id = (g_sim.case1.active_layer * 3 + 7) % 8;
            std::stringstream ss;
            if (case1_layer_frames <= 5) {
                ss << "L" << g_sim.case1.active_layer << ": Computing Multi-Head Self-Attention...";
            } else if (case1_layer_frames <= 10) {
                ss << "L" << g_sim.case1.active_layer << ": Evaluating Gating Router -> Dispatching active parameters...";
            } else {
                ss << "L" << g_sim.case1.active_layer << ": Executing FFN matrix logic for Expert " << exp_id << "...";
            }
            g_sim.case1.status_message = ss.str();

            if (case1_layer_frames >= 15) {
                case1_layer_frames = 0;
                g_sim.case1.active_layer = (g_sim.case1.active_layer + 1) % 12;

                // Trigger a stall periodically (layers 2, 5, 8, 11)
                if (g_sim.case1.active_layer == 2 || g_sim.case1.active_layer == 5 || 
                    g_sim.case1.active_layer == 8 || g_sim.case1.active_layer == 11) {
                    g_sim.case1.is_stalled = true;
                    g_sim.case1.stall_layer = g_sim.case1.active_layer;
                    case1_stall_frames_left = 40; // 4 seconds stall
                }
            }
        }
        g_sim.case1.ram_usage_mb = g_metrics.case1_ram_usage_mb + dis(gen) * 5.0;

        // ==========================================
        // UPDATE CASE 2 (Asynchronous prefetch)
        // ==========================================
        case2_layer_frames++;
        g_sim.case2.throughput = g_metrics.case2_throughput + dis2(gen);
        g_sim.case2.hit_rate = g_metrics.case2_hit_rate + dis(gen) * 0.1;
        g_sim.case2.ssd_read_mb = g_metrics.case2_pcie_bw + dis_ssd(gen);
        g_sim.case2.ram_usage_mb = g_metrics.case2_ram_usage_mb + dis2(gen) * 10.0;

        int comp_exp = (g_sim.case2.active_layer * 3 + 7) % 8;
        std::stringstream ss2;
        if (case2_layer_frames <= 5) {
            ss2 << "L" << g_sim.case2.active_layer << ": Computing Attention & Gating concurrently...";
        } else {
            ss2 << "L" << g_sim.case2.active_layer << ": Running Expert " << comp_exp << " (Cache Hit: weights already resident)...";
        }
        g_sim.case2.status_message = ss2.str();

        // Advance active layer every 15 frames (1.5 seconds) to align with Case 1
        if (case2_layer_frames >= 15) {
            case2_layer_frames = 0;
            g_sim.case2.active_layer = (g_sim.case2.active_layer + 1) % 12;

            // Shift prefetch transfers
            if (!g_sim.case2.prefetch_transfers.empty()) {
                g_sim.case2.prefetch_transfers.erase(g_sim.case2.prefetch_transfers.begin());
            }
            // Queue a new prefetch at the end (for 3 layers ahead)
            int next_prefetch_layer = (g_sim.case2.active_layer + 3) % 12;
            int next_prefetch_expert = (next_prefetch_layer * 3 + 7) % 8;
            g_sim.case2.prefetch_transfers.push_back({next_prefetch_layer, next_prefetch_expert, 0.0});
        }

        // Increment prefetch progress for items in the queue (10% increments)
        int idx = 0;
        for (auto& pf : g_sim.case2.prefetch_transfers) {
            if (idx == 0) {
                pf.progress = std::min(100.0, pf.progress + 10.0);
            } else if (idx == 1) {
                pf.progress = std::min(100.0, pf.progress + 8.0);
            } else {
                pf.progress = std::min(100.0, pf.progress + 5.0);
            }
            idx++;
        }
    }
}

std::string get_simulation_frame_json() {
    std::lock_guard<std::mutex> lock(g_sim.mtx);
    std::stringstream ss;
    ss << "{";
    
    // Case 1
    ss << "\"case1\":{";
    ss << "\"active_layer\":" << g_sim.case1.active_layer << ",";
    ss << "\"is_stalled\":" << (g_sim.case1.is_stalled ? "true" : "false") << ",";
    ss << "\"stall_layer\":" << g_sim.case1.stall_layer << ",";
    ss << "\"throughput\":" << g_sim.case1.throughput << ",";
    ss << "\"hit_rate\":" << g_sim.case1.hit_rate << ",";
    ss << "\"ssd_read_mb\":" << g_sim.case1.ssd_read_mb << ",";
    ss << "\"ram_usage_mb\":" << g_sim.case1.ram_usage_mb << ",";
    ss << "\"elapsed_sec\":" << g_sim.case1.elapsed_sec << ",";
    ss << "\"status_message\":\"" << g_sim.case1.status_message << "\",";
    ss << "\"frames_left\":" << g_sim.case1.frames_left;
    ss << "},";

    // Case 2 (Real Telemetry from execution pipeline)
    ss << "\"case2\":{";
    ss << "\"active_layer\":" << g_telemetry.active_layer.load() << ",";
    ss << "\"throughput\":" << g_telemetry.throughput.load() << ",";
    ss << "\"hit_rate\":" << g_telemetry.hit_rate.load() << ",";
    ss << "\"ssd_read_mb\":" << g_telemetry.ssd_read_mb_s.load() << ",";
    ss << "\"ram_usage_mb\":" << (g_telemetry.ram_usage_bytes.load() / (1024.0 * 1024.0)) << ",";
    ss << "\"elapsed_sec\":" << g_sim.case1.elapsed_sec << ",";
    
    std::string real_status = "";
    {
        std::lock_guard<std::mutex> status_lock(g_telemetry.status_mutex);
        real_status = g_telemetry.status_message;
    }
    ss << "\"status_message\":\"" << real_status << "\",";
    
    // lookahead signals
    ss << "\"lookahead\":[";
    for (int i = 0; i < 4; ++i) {
        ss << "{\"layer\":" << g_telemetry.lookahead_layers[i].load() 
           << ",\"expert\":" << g_telemetry.lookahead_experts[i].load() << "}";
        if (i + 1 < 4) ss << ",";
    }
    ss << "],";

    // prefetch transfers
    ss << "\"prefetch_transfers\":[";
    for (int i = 0; i < 4; ++i) {
        ss << "{\"layer\":" << g_telemetry.lookahead_layers[i].load() 
           << ",\"expert\":" << g_telemetry.lookahead_experts[i].load() 
           << ",\"progress\":" << g_telemetry.prefetch_progress[i].load() << "}";
        if (i + 1 < 4) ss << ",";
    }
    ss << "],";

    // compute active
    int active_layer = g_telemetry.active_layer.load();
    int active_expert = 0;
    if (active_layer < 4) {
        active_expert = g_telemetry.lookahead_experts[active_layer].load();
    }
    ss << "\"compute_active\":{\"layer\":" << active_layer << ",\"expert\":" << active_expert << "}";
    
    ss << "}"; // end case2
    
    ss << "}"; // end root
    return ss.str();
}

void handle_client(socket_t client_fd) {
    char buffer[2048];
    memset(buffer, 0, sizeof(buffer));
    ssize_t bytes_read = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
    if (bytes_read <= 0) {
        CLOSE_SOCKET(client_fd);
        return;
    }

    std::string request(buffer);
    std::string method, path;
    std::istringstream iss(request);
    iss >> method >> path;

    if (method != "GET") {
        std::string response = "HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        send(client_fd, response.c_str(), response.size(), MSG_NOSIGNAL);
        CLOSE_SOCKET(client_fd);
        return;
    }

    if (path == "/" || path == "/index.html") {
        std::ifstream file("web/index.html");
        if (!file.is_open()) {
            file.open("../web/index.html");
        }
        if (!file.is_open()) {
            file.open("../../web/index.html");
        }

        if (!file.is_open()) {
            std::string body = "<html><body><h1>web/index.html not found on disk</h1></body></html>";
            std::string response = "HTTP/1.1 404 Not Found\r\n"
                                   "Content-Type: text/html\r\n"
                                   "Content-Length: " + std::to_string(body.size()) + "\r\n"
                                   "Connection: close\r\n\r\n" + body;
            send(client_fd, response.c_str(), response.size(), MSG_NOSIGNAL);
        } else {
            std::stringstream file_content;
            file_content << file.rdbuf();
            std::string body = file_content.str();
            std::string response = "HTTP/1.1 200 OK\r\n"
                                   "Content-Type: text/html\r\n"
                                   "Content-Length: " + std::to_string(body.size()) + "\r\n"
                                   "Connection: close\r\n\r\n" + body;
            send(client_fd, response.c_str(), response.size(), MSG_NOSIGNAL);
        }
        CLOSE_SOCKET(client_fd);
    } else if (path == "/stream") {
        std::string headers = "HTTP/1.1 200 OK\r\n"
                              "Content-Type: text/event-stream\r\n"
                              "Cache-Control: no-cache\r\n"
                              "Connection: keep-alive\r\n"
                              "Access-Control-Allow-Origin: *\r\n\r\n";
        
        ssize_t sent = send(client_fd, headers.c_str(), headers.size(), MSG_NOSIGNAL);
        if (sent < 0) {
            CLOSE_SOCKET(client_fd);
            return;
        }

        while (true) {
            std::string json_frame = get_simulation_frame_json();
            std::string sse_msg = "data: " + json_frame + "\n\n";
            ssize_t s = send(client_fd, sse_msg.c_str(), sse_msg.size(), MSG_NOSIGNAL);
            if (s < 0) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 10Hz match
        }
        CLOSE_SOCKET(client_fd);
    } else if (path.rfind("/chat?", 0) == 0) {
        std::string prompt = "";
        size_t prompt_pos = path.find("prompt=");
        if (prompt_pos != std::string::npos) {
            prompt = path.substr(prompt_pos + 7);
            // URL decode prompt
            std::string decoded = "";
            for (size_t i = 0; i < prompt.size(); ++i) {
                if (prompt[i] == '+') {
                    decoded += ' ';
                } else if (prompt[i] == '%' && i + 2 < prompt.size()) {
                    int hex_val = 0;
                    std::stringstream ss_hex;
                    ss_hex << std::hex << prompt.substr(i + 1, 2);
                    ss_hex >> hex_val;
                    decoded += static_cast<char>(hex_val);
                    i += 2;
                } else {
                    decoded += prompt[i];
                }
            }
            prompt = decoded;
        }

        std::cout << "[Visualizer Server] Received chat prompt: " << prompt << std::endl;

        {
            std::lock_guard<std::mutex> lock(g_chat_mutex);
            g_pending_prompt = prompt;
            g_pending_tokens.clear();
            g_chat_active = true;
        }

        std::string headers = "HTTP/1.1 200 OK\r\n"
                              "Content-Type: text/event-stream\r\n"
                              "Cache-Control: no-cache\r\n"
                              "Connection: keep-alive\r\n"
                              "Access-Control-Allow-Origin: *\r\n\r\n";
        send(client_fd, headers.c_str(), headers.size(), MSG_NOSIGNAL);

        while (true) {
            std::unique_lock<std::mutex> lock(g_chat_mutex);
            g_chat_cv.wait_for(lock, std::chrono::milliseconds(200), []() {
                return !g_pending_tokens.empty() || !g_chat_active;
            });

            if (!g_pending_tokens.empty()) {
                for (const auto& token : g_pending_tokens) {
                    std::string sse_msg = "data: " + token + "\n\n";
                    send(client_fd, sse_msg.c_str(), sse_msg.size(), MSG_NOSIGNAL);
                }
                g_pending_tokens.clear();
            }

            if (!g_chat_active) {
                break;
            }
        }
        CLOSE_SOCKET(client_fd);
    } else {
        std::string body = "<html><body><h1>404 Not Found</h1></body></html>";
        std::string response = "HTTP/1.1 404 Not Found\r\n"
                               "Content-Type: text/html\r\n"
                               "Content-Length: " + std::to_string(body.size()) + "\r\n"
                               "Connection: close\r\n\r\n" + body;
        send(client_fd, response.c_str(), response.size(), MSG_NOSIGNAL);
        CLOSE_SOCKET(client_fd);
    }
}

extern "C" MOE_API void start_visualizer_server() {
#ifdef _WIN32
    WSADATA wsaData;
    int wsa_res = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (wsa_res != 0) {
        std::cerr << "WSAStartup failed with error: " << wsa_res << "\n";
        return;
    }
#else
    signal(SIGPIPE, SIG_IGN);
#endif

    load_real_metrics();

    std::thread sim_thread(run_simulation);
    sim_thread.detach();

    socket_t server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (!IS_VALID_SOCKET(server_fd)) {
        std::cerr << "Failed to create socket\n";
#ifdef _WIN32
        WSACleanup();
#endif
        return;
    }

#ifdef _WIN32
    const char opt = 1;
#else
    int opt = 1;
#endif
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt)) < 0) {
        std::cerr << "setsockopt SO_REUSEADDR failed\n";
        CLOSE_SOCKET(server_fd);
#ifdef _WIN32
        WSACleanup();
#endif
        return;
    }

    struct sockaddr_in address;
    std::memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(8082);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "Bind to port 8082 failed\n";
        CLOSE_SOCKET(server_fd);
#ifdef _WIN32
        WSACleanup();
#endif
        return;
    }

    if (listen(server_fd, 10) < 0) {
        std::cerr << "Listen failed\n";
        CLOSE_SOCKET(server_fd);
#ifdef _WIN32
        WSACleanup();
#endif
        return;
    }

    std::cout << "MoE Visualizer Server listening on port 8082...\n";

    while (true) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        socket_t client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (IS_VALID_SOCKET(client_fd)) {
            std::thread(handle_client, client_fd).detach();
        }
    }

    CLOSE_SOCKET(server_fd);
#ifdef _WIN32
    WSACleanup();
#endif
}

// Keep a standalone main wrapper that just calls start_visualizer_server to avoid breaking standalone builds
int main() {
    start_visualizer_server();
    return 0;
}
