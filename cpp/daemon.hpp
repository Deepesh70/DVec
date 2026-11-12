#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include "spsc_ring_buffer.hpp"
#include "audio_processor.hpp"
#include "onnx_model.hpp"

class DVecDaemon {
public:
    static constexpr std::size_t RING_CAPACITY = 8192; // Headroom for ~500ms of audio
    static constexpr std::size_t HOP_SIZE = AudioProcessor::HOP_SIZE; // 160 samples

private:
    SPSCRingBuffer<float, RING_CAPACITY> in_ring_{};
    SPSCRingBuffer<float, RING_CAPACITY> out_ring_{};

    AudioProcessor audio_proc_{};
    std::unique_ptr<DVecModel> model_{nullptr};

    std::atomic<bool> is_running_{false};
    std::thread worker_thread_{};

    void WorkerLoop() noexcept {
        std::array<float, HOP_SIZE> in_hop{};
        std::array<float, HOP_SIZE> out_hop{};
        std::array<float, AudioProcessor::FREQ_BINS> spec_r{};
        std::array<float, AudioProcessor::FREQ_BINS> spec_i{};

        while (is_running_.load(std::memory_order_relaxed)) {
            // 1. Pop 160 samples (10ms) from the input ring buffer
            if (!in_ring_.Pop(in_hop.data(), HOP_SIZE)) {
                std::this_thread::yield(); // Wait for audio callback to push samples
                continue;
            }

            // 2. Causal STFT Analysis
            audio_proc_.AnalyzeHop(in_hop.data(), spec_r.data(), spec_i.data());

            // 3. Stateful ONNX neural extraction & complex masking
            if (model_) {
                model_->ProcessSpectrum(spec_r.data(), spec_i.data());
            }

            // 4. Causal iSTFT Overlap-Add Synthesis
            audio_proc_.SynthesizeHop(spec_r.data(), spec_i.data(), out_hop.data());

            // 5. Push cleaned audio to output ring buffer
            while (!out_ring_.Push(out_hop.data(), HOP_SIZE) &&
                   is_running_.load(std::memory_order_relaxed)) {
                std::this_thread::yield();
            }
        }
    }

public:
    DVecDaemon(const std::string& model_path, const std::string& profile_path) {
        model_ = std::make_unique<DVecModel>(model_path, profile_path);
    }

    ~DVecDaemon() {
        Stop();
    }

    void Start() {
        if (is_running_.exchange(true)) {
            return; // Already running
        }
        worker_thread_ = std::thread(&DVecDaemon::WorkerLoop, this);
    }

    void Stop() {
        if (!is_running_.exchange(false)) {
            return; // Already stopped
        }
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

    // Called by physical microphone callback (Lock-free, zero allocation)
    bool PushMicAudio(const float* pcm, std::size_t count) noexcept {
        return in_ring_.Push(pcm, count);
    }

    // Called by virtual speaker callback (Lock-free, zero allocation)
    // Emits silence if output buffer underflows to avoid pops
    bool PopCleanAudio(float* pcm, std::size_t count) noexcept {
        if (!out_ring_.Pop(pcm, count)) {
            std::fill(pcm, pcm + count, 0.0f); // Underflow protection
            return false;
        }
        return true;
    }
};