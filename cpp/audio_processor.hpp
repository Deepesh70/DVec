#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

class AudioProcessor {
public:
    static constexpr std::size_t FRAME_SIZE = 320; // 20 ms @ 16 kHz
    static constexpr std::size_t HOP_SIZE   = 160; // 10 ms @ 16 kHz
    static constexpr std::size_t FFT_SIZE   = 512; // Next power of 2
    static constexpr std::size_t FREQ_BINS  = (FFT_SIZE / 2) + 1; // 257 bins

private:
    std::array<float, FRAME_SIZE> in_buffer_{};
    std::array<float, FRAME_SIZE> ola_buffer_{};
    std::array<float, FRAME_SIZE> window_{};

    // Pre-computed twiddle factors for 512-point FFT (avoids runtime cos/sin)
    std::array<std::complex<float>, FFT_SIZE / 2> twiddles_{};
    std::array<std::size_t, FFT_SIZE> bit_reverse_{};

    void InitializeTables() noexcept {
        constexpr float pi = 3.14159265358979323846f;

        // 1. Square-root Hann window: w[n] = sqrt(0.5 * (1 - cos(2*pi*n / N)))
        for (std::size_t n = 0; n < FRAME_SIZE; ++n) {
            float val = 0.5f * (1.0f - std::cos(2.0f * pi * static_cast<float>(n) / static_cast<float>(FRAME_SIZE)));
            window_[n] = std::sqrt(val);
        }

        // 2. FFT Twiddle factors: W_N^k = exp(-2*pi*i*k / N)
        for (std::size_t k = 0; k < FFT_SIZE / 2; ++k) {
            float angle = -2.0f * pi * static_cast<float>(k) / static_cast<float>(FFT_SIZE);
            twiddles_[k] = std::complex<float>(std::cos(angle), std::sin(angle));
        }

        // 3. Bit-reversal permutation indices
        for (std::size_t i = 0; i < FFT_SIZE; ++i) {
            std::size_t rev = 0;
            std::size_t temp = i;
            for (std::size_t j = 0; j < 9; ++j) { // 2^9 = 512
                rev = (rev << 1) | (temp & 1);
                temp >>= 1;
            }
            bit_reverse_[i] = rev;
        }
    }

    // In-place Radix-2 Cooley-Tukey FFT
    void ForwardFFT(std::array<std::complex<float>, FFT_SIZE>& x) noexcept {
        // Bit-reversal reordering
        for (std::size_t i = 0; i < FFT_SIZE; ++i) {
            std::size_t j = bit_reverse_[i];
            if (i < j) {
                std::swap(x[i], x[j]);
            }
        }

        // Cooley-Tukey butterfly stages
        for (std::size_t len = 2; len <= FFT_SIZE; len <<= 1) {
            std::size_t half = len >> 1;
            std::size_t step = FFT_SIZE / len;
            for (std::size_t i = 0; i < FFT_SIZE; i += len) {
                for (std::size_t j = 0; j < half; ++j) {
                    std::complex<float> u = x[i + j];
                    std::complex<float> v = x[i + j + half] * twiddles_[j * step];
                    x[i + j] = u + v;
                    x[i + j + half] = u - v;
                }
            }
        }
    }

    // In-place Inverse FFT
    void InverseFFT(std::array<std::complex<float>, FFT_SIZE>& x) noexcept {
        // Conjugate input
        for (std::size_t i = 0; i < FFT_SIZE; ++i) {
            x[i] = std::conj(x[i]);
        }

        ForwardFFT(x);

        // Conjugate output and scale by 1/N
        constexpr float inv_n = 1.0f / static_cast<float>(FFT_SIZE);
        for (std::size_t i = 0; i < FFT_SIZE; ++i) {
            x[i] = std::conj(x[i]) * inv_n;
        }
    }

public:
    AudioProcessor() noexcept {
        InitializeTables();
    }

    // Transforms 160 time-domain samples to 257 complex frequency bins
    void AnalyzeHop(const float* in_pcm,
                    float* spec_real,
                    float* spec_imag) noexcept {
        // 1. Shift input buffer left by HOP_SIZE
        for (std::size_t i = 0; i < FRAME_SIZE - HOP_SIZE; ++i) {
            in_buffer_[i] = in_buffer_[i + HOP_SIZE];
        }

        // 2. Append new 160 samples
        for (std::size_t i = 0; i < HOP_SIZE; ++i) {
            in_buffer_[FRAME_SIZE - HOP_SIZE + i] = in_pcm[i];
        }

        // 3. Window and zero-pad to 512 points
        std::array<std::complex<float>, FFT_SIZE> fft_buf{};
        for (std::size_t i = 0; i < FRAME_SIZE; ++i) {
            fft_buf[i] = std::complex<float>(in_buffer_[i] * window_[i], 0.0f);
        }
        for (std::size_t i = FRAME_SIZE; i < FFT_SIZE; ++i) {
            fft_buf[i] = std::complex<float>(0.0f, 0.0f);
        }

        // 4. Compute Forward FFT
        ForwardFFT(fft_buf);

        // 5. Extract first 257 positive frequency bins
        for (std::size_t k = 0; k < FREQ_BINS; ++k) {
            spec_real[k] = fft_buf[k].real();
            spec_imag[k] = fft_buf[k].imag();
        }
    }

    // Synthesizes masked spectrum back to 160 time-domain PCM samples
    void SynthesizeHop(const float* spec_real,
                       const float* spec_imag,
                       float* out_pcm) noexcept {
        // 1. Reconstruct full 512-point Hermitian spectrum
        std::array<std::complex<float>, FFT_SIZE> ifft_buf{};
        for (std::size_t k = 0; k < FREQ_BINS; ++k) {
            ifft_buf[k] = std::complex<float>(spec_real[k], spec_imag[k]);
        }
        // Conjugate symmetric negative frequencies: X[N - k] = conj(X[k])
        for (std::size_t k = 1; k < FFT_SIZE - FREQ_BINS + 1; ++k) {
            ifft_buf[FFT_SIZE - k] = std::conj(ifft_buf[k]);
        }

        // 2. Compute Inverse FFT
        InverseFFT(ifft_buf);

        // 3. Window the time-domain frame and accumulate into overlap-add buffer
        for (std::size_t i = 0; i < FRAME_SIZE; ++i) {
            ola_buffer_[i] += ifft_buf[i].real() * window_[i];
        }

        // 4. Emit the first HOP_SIZE samples
        for (std::size_t i = 0; i < HOP_SIZE; ++i) {
            out_pcm[i] = ola_buffer_[i];
        }

        // 5. Shift overlap-add buffer left by HOP_SIZE and zero trailing region
        for (std::size_t i = 0; i < FRAME_SIZE - HOP_SIZE; ++i) {
            ola_buffer_[i] = ola_buffer_[i + HOP_SIZE];
        }
        for (std::size_t i = FRAME_SIZE - HOP_SIZE; i < FRAME_SIZE; ++i) {
            ola_buffer_[i] = 0.0f;
        }
    }
};