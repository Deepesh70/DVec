#pragma once

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

class DVecModel {
public:
    static constexpr std::size_t FREQ_BINS = 257;
    static constexpr std::size_t EMBED_DIM = 256;
    static constexpr std::size_t LSTM_LAYERS = 2;
    static constexpr std::size_t HIDDEN_DIM = 128;
    static constexpr std::size_t STATE_SIZE = LSTM_LAYERS * 1 * HIDDEN_DIM; // 256

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "DVecDaemon"};
    Ort::SessionOptions session_options_{};
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_info_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

    // Persistent tensors
    std::array<float, EMBED_DIM> spk_embed_{};
    std::array<float, STATE_SIZE> h_state_{};
    std::array<float, STATE_SIZE> c_state_{};

    // Pre-allocated I/O buffers
    std::array<float, FREQ_BINS> input_real_{};
    std::array<float, FREQ_BINS> input_imag_{};

public:
    DVecModel(const std::string& model_path, const std::string& profile_path) {
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
        const std::wstring wide_model_path(model_path.begin(), model_path.end());
        session_ = Ort::Session(env_, wide_model_path.c_str(), session_options_);
#else
        session_ = Ort::Session(env_, model_path.c_str(), session_options_);
#endif

        LoadSpeakerProfile(profile_path);
        ResetStates();
    }

    void LoadSpeakerProfile(const std::string& profile_path) {
        std::ifstream file(profile_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open speaker profile: " + profile_path);
        }
        file.read(reinterpret_cast<char*>(spk_embed_.data()), EMBED_DIM * sizeof(float));
        if (file.gcount() != static_cast<std::streamsize>(EMBED_DIM * sizeof(float))) {
            throw std::runtime_error("Profile file corrupted or incorrect size (expected 1024 bytes).");
        }
    }

    void ResetStates() noexcept {
        h_state_.fill(0.0f);
        c_state_.fill(0.0f);
    }

    // Runs a single 10ms frame forward pass and applies the complex mask in-place
    void ProcessSpectrum(float* spec_real, float* spec_imag) {
        // Tensor shapes
        const std::array<int64_t, 4> spec_shape = {1, 1, static_cast<int64_t>(FREQ_BINS), 1};
        const std::array<int64_t, 2> embed_shape = {1, static_cast<int64_t>(EMBED_DIM)};
        const std::array<int64_t, 3> state_shape = {static_cast<int64_t>(LSTM_LAYERS), 1, static_cast<int64_t>(HIDDEN_DIM)};

        // Wrap raw buffers without allocating new memory
        std::array<Ort::Value, 5> input_tensors = {
            Ort::Value::CreateTensor<float>(memory_info_, spec_real, FREQ_BINS, spec_shape.data(), spec_shape.size()),
            Ort::Value::CreateTensor<float>(memory_info_, spec_imag, FREQ_BINS, spec_shape.data(), spec_shape.size()),
            Ort::Value::CreateTensor<float>(memory_info_, spk_embed_.data(), EMBED_DIM, embed_shape.data(), embed_shape.size()),
            Ort::Value::CreateTensor<float>(memory_info_, h_state_.data(), STATE_SIZE, state_shape.data(), state_shape.size()),
            Ort::Value::CreateTensor<float>(memory_info_, c_state_.data(), STATE_SIZE, state_shape.data(), state_shape.size())
        };

        const char* input_names[] = {"input_real", "input_imag", "spk_embed", "h_in", "c_in"};
        const char* output_names[] = {"mask_real", "mask_imag", "vad_prob", "h_out", "c_out"};

        // Execute inference
        auto output_tensors = session_.Run(
            Ort::RunOptions{nullptr},
            input_names,
            input_tensors.data(),
            input_tensors.size(),
            output_names,
            5
        );

        // Extract outputs
        const float* mask_r = output_tensors[0].GetTensorData<float>();
        const float* mask_i = output_tensors[1].GetTensorData<float>();
        const float vad_prob = *output_tensors[2].GetTensorData<float>();
        const float* h_next = output_tensors[3].GetTensorData<float>();
        const float* c_next = output_tensors[4].GetTensorData<float>();

        // Update recurrent states for the next frame
        std::copy(h_next, h_next + STATE_SIZE, h_state_.begin());
        std::copy(c_next, c_next + STATE_SIZE, c_state_.begin());

        // Soft Personal VAD gate
        const float vad_gain = std::clamp((vad_prob - 0.2f) / 0.3f, 0.02f, 1.0f);

        // Apply complex multiplication in-place: (spec_r + j*spec_i) * (mr + j*mi)
        for (std::size_t k = 0; k < FREQ_BINS; ++k) {
            float mr = mask_r[k] * vad_gain;
            float mi = mask_i[k] * vad_gain;
            float r = spec_real[k];
            float i = spec_imag[k];

            spec_real[k] = r * mr - i * mi;
            spec_imag[k] = r * mi + i * mr;
        }
    }
};