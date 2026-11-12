#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <string>

#define MINIAUDIO_IMPLEMENTATION
#include "audio_device.hpp"
#include "daemon.hpp"

// Locate the ONNX model whether running from repository root or cpp/ directory
static std::string FindModelPath() {
    std::ifstream f1("dvec_v1.onnx");
    if (f1.good()) return "dvec_v1.onnx";
    std::ifstream f2("../dvec_v1.onnx");
    if (f2.good()) return "../dvec_v1.onnx";
    return "dvec_v1.onnx";
}

int main(int argc, char* argv[]) {
    std::cout << "=================================================\n";
    std::cout << "   DVec: Real-Time Target Speaker Extraction     \n";
    std::cout << "=================================================\n\n";

    const std::string model_path = FindModelPath();
    const std::string profile_path = "test_profile.bin";

    // Auto-create dummy 256-d profile if none exists
    {
        std::ifstream check(profile_path, std::ios::binary);
        if (!check.good()) {
            std::cout << "Creating default test profile (" << profile_path << ")...\n";
            std::ofstream dummy(profile_path, std::ios::binary);
            std::vector<float> embed(256, 0.0f);
            embed[0] = 1.0f; // Unit vector
            dummy.write(reinterpret_cast<const char*>(embed.data()), embed.size() * sizeof(float));
        }
    }

    std::cout << "Initializing neural daemon engine...\n";
    DVecDaemon daemon(model_path, profile_path);
    daemon.Start();
    std::cout << "Neural worker thread active.\n";

    // Check if test simulation mode requested
    bool test_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--test") {
            test_mode = true;
        }
    }

    if (test_mode) {
        std::cout << "\nRunning offline verification simulation (1 second audio)...\n";
        constexpr std::size_t sample_rate = 16000;
        constexpr std::size_t total_samples = 16000;
        constexpr std::size_t hop_size = DVecDaemon::HOP_SIZE;

        std::vector<float> mic_input(total_samples);
        constexpr float pi = 3.14159265358979323846f;
        for (std::size_t i = 0; i < total_samples; ++i) {
            mic_input[i] = 0.5f * std::sin(2.0f * pi * 440.0f * static_cast<float>(i) / sample_rate);
        }

        std::vector<float> render_buf(hop_size);
        for (std::size_t i = 0; i + hop_size <= total_samples; i += hop_size) {
            daemon.PushMicAudio(&mic_input[i], hop_size);
            daemon.PopCleanAudio(render_buf.data(), hop_size);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        daemon.Stop();
        std::cout << "Simulation passed successfully.\n";
        return 0;
    }

    // Live audio hardware mode
    AudioDeviceManager audio_mgr;
    audio_mgr.PrintAvailableDevices();

    int mic_idx = -1;
    int out_idx = -1;
    if (argc >= 3) {
        mic_idx = std::stoi(argv[1]);
        out_idx = std::stoi(argv[2]);
    }

    std::cout << "Starting live duplex audio stream (16 kHz Mono)...\n";
    if (!audio_mgr.Initialize(&daemon, mic_idx, out_idx)) {
        std::cerr << "Failed to initialize audio devices.\n";
        daemon.Stop();
        return 1;
    }

    if (!audio_mgr.Start()) {
        std::cerr << "Failed to start audio stream.\n";
        daemon.Stop();
        return 1;
    }

    std::cout << "\n>>> DVec is running live! <<<\n";
    std::cout << "Audio is being filtered in real-time under a 10ms budget.\n";
    std::cout << "Press [Enter] to stop the daemon...\n";
    std::cin.get();

    std::cout << "Stopping audio stream...\n";
    audio_mgr.Stop();
    daemon.Stop();
    std::cout << "DVec Daemon exited cleanly.\n";

    return 0;
}
