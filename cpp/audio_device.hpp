#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

// Include miniaudio
#include "miniaudio.h"
#include "daemon.hpp"

class AudioDeviceManager {
private:
    ma_context context_{};
    ma_device device_{};
    bool is_initialized_{false};
    bool is_started_{false};

    static void DataCallback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
        auto* daemon = static_cast<DVecDaemon*>(pDevice->pUserData);
        if (!daemon) return;

        // 1. Capture microphone audio
        if (pInput != nullptr) {
            daemon->PushMicAudio(static_cast<const float*>(pInput), frameCount);
        }

        // 2. Output processed cleaned audio
        if (pOutput != nullptr) {
            daemon->PopCleanAudio(static_cast<float*>(pOutput), frameCount);
        }
    }

public:
    AudioDeviceManager() {
        if (ma_context_init(nullptr, 0, nullptr, &context_) != MA_SUCCESS) {
            throw std::runtime_error("Failed to initialize miniaudio context.");
        }
    }

    ~AudioDeviceManager() {
        Stop();
        if (is_initialized_) {
            ma_device_uninit(&device_);
        }
        ma_context_uninit(&context_);
    }

    void PrintAvailableDevices() {
        ma_device_info* pPlaybackInfos = nullptr;
        ma_uint32 playbackCount = 0;
        ma_device_info* pCaptureInfos = nullptr;
        ma_uint32 captureCount = 0;

        if (ma_context_get_devices(&context_, &pPlaybackInfos, &playbackCount, &pCaptureInfos, &captureCount) != MA_SUCCESS) {
            std::cerr << "Failed to enumerate audio devices.\n";
            return;
        }

        std::cout << "\n=== Available Microphones (Capture Devices) ===\n";
        for (ma_uint32 i = 0; i < captureCount; ++i) {
            std::cout << "  [" << i << "] " << pCaptureInfos[i].name
                      << (pCaptureInfos[i].isDefault ? " (Default)" : "") << "\n";
        }

        std::cout << "\n=== Available Output Sinks (Playback Devices) ===\n";
        for (ma_uint32 i = 0; i < playbackCount; ++i) {
            std::cout << "  [" << i << "] " << pPlaybackInfos[i].name
                      << (pPlaybackInfos[i].isDefault ? " (Default)" : "") << "\n";
        }
        std::cout << "\n";
    }

    bool Initialize(DVecDaemon* daemon, int capture_index = -1, int playback_index = -1) {
        ma_device_info* pPlaybackInfos = nullptr;
        ma_uint32 playbackCount = 0;
        ma_device_info* pCaptureInfos = nullptr;
        ma_uint32 captureCount = 0;

        if (ma_context_get_devices(&context_, &pPlaybackInfos, &playbackCount, &pCaptureInfos, &captureCount) != MA_SUCCESS) {
            std::cerr << "Failed to query devices.\n";
            return false;
        }

        ma_device_config config = ma_device_config_init(ma_device_type_duplex);
        config.dataCallback = DataCallback;
        config.pUserData = daemon;

        // 16 kHz Mono Float format (matches neural model exact input contract)
        config.sampleRate = 16000;
        config.periodSizeInFrames = DVecDaemon::HOP_SIZE; // 160 samples (10ms)

        // Capture settings
        config.capture.format = ma_format_f32;
        config.capture.channels = 1;
        config.capture.shareMode = ma_share_mode_shared;
        if (capture_index >= 0 && static_cast<ma_uint32>(capture_index) < captureCount) {
            config.capture.pDeviceID = &pCaptureInfos[capture_index].id;
            std::cout << "Selected Mic: " << pCaptureInfos[capture_index].name << "\n";
        } else {
            config.capture.pDeviceID = nullptr; // Default
            std::cout << "Selected Mic: System Default\n";
        }

        // Playback settings
        config.playback.format = ma_format_f32;
        config.playback.channels = 1;
        config.playback.shareMode = ma_share_mode_shared;
        if (playback_index >= 0 && static_cast<ma_uint32>(playback_index) < playbackCount) {
            config.playback.pDeviceID = &pPlaybackInfos[playback_index].id;
            std::cout << "Selected Output: " << pPlaybackInfos[playback_index].name << "\n";
        } else {
            config.playback.pDeviceID = nullptr; // Default
            std::cout << "Selected Output: System Default\n";
        }

        if (ma_device_init(&context_, &config, &device_) != MA_SUCCESS) {
            std::cerr << "Failed to initialize duplex audio device.\n";
            return false;
        }

        is_initialized_ = true;
        return true;
    }

    bool Start() {
        if (!is_initialized_) return false;
        if (is_started_) return true;

        if (ma_device_start(&device_) != MA_SUCCESS) {
            std::cerr << "Failed to start audio device.\n";
            return false;
        }

        is_started_ = true;
        return true;
    }

    void Stop() {
        if (is_started_) {
            ma_device_stop(&device_);
            is_started_ = false;
        }
    }
};
