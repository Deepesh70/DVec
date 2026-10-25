# 01. Mathematical Pipeline & Signal Processing

This document details the mathematical operations executed on the audio signal across both the analysis (STFT) and synthesis (iSTFT) stages, as well as the neural conditioning and complex masking mechanisms.

---

## 1. Input Audio Invariants

- **Sampling Rate ($f_s$):** $16{,}000\text{ Hz}$ (Mono, 16-bit signed PCM).
- **Nyquist Limit:** $8{,}000\text{ Hz}$. Human speech energy is concentrated between $100\text{ Hz}$ and $6{,}000\text{ Hz}$ (fundamental frequencies $F_0 \approx 85\text{--}255\text{ Hz}$, formant structures $F_1\text{--}F_4 \approx 500\text{--}4{,}500\text{ Hz}$).
- **Frame Length ($N$):** $320\text{ samples}$ ($20\text{ ms}$) or $512\text{ samples}$ ($32\text{ ms}$).
- **Hop Size / Stride ($R$):** $160\text{ samples}$ ($10\text{ ms}$).
- **FFT Size ($K$):** $512\text{ points}$ (zero-padded on past samples).
- **Frequency Bins ($F$):** $K / 2 + 1 = 257\text{ bins}$.

---

## 2. Causal STFT Analysis & Windowing

To maintain strictly causal execution with zero future lookahead:
1. Every analysis window of length $N$ uses samples from the current hop $[n - R + 1, n]$ combined with the buffered past samples $[n - N + 1, n - R]$.
2. Zero padding into future samples $[n + 1, \dots]$ is strictly prohibited.
3. Analysis window $w[n]$ is typically a square-root Hann window or an asymmetric Hann-like window tailored for causal reconstruction:

$$w[n] = \sin\left(\frac{\pi (n + 0.5)}{N}\right), \quad 0 \le n < N$$

For frame index $m$, the complex Short-Time Fourier Transform $X(m, k) \in \mathbb{C}$ at frequency bin $k$ is computed as:

$$X(m, k) = \sum_{n=0}^{N-1} x[mR + n] \cdot w[n] \cdot e^{-j \frac{2\pi k n}{K}} = X_r(m, k) + j X_i(m, k)$$

The input tensor to the neural network has shape $(B, 2, 1, F)$, representing real and imaginary spectral planes for a single time frame.

---

## 3. FiLM (Feature-wise Linear Modulation) Conditioning

The extraction model is conditioned on the enrolled speaker embedding $e \in \mathbb{R}^{D}$ ($D = 256$, where $\|e\|_2 = 1$).

Rather than concatenating $e$ with spectral features (which adds parameter overhead and weak gradient flow), we apply **FiLM** at the bottleneck feature map $H \in \mathbb{R}^{B \times C \times 1 \times F}$:

$$\gamma = W_\gamma e + b_\gamma, \quad \beta = W_\beta e + b_\beta \quad (\gamma, \beta \in \mathbb{R}^{C})$$

$$\text{FiLM}(H_c) = \gamma_c \odot H_c + \beta_c$$

- $\gamma_c$ acts as an adaptive feature-scale factor, selectively amplifying filters that respond to the target speaker's vocal resonance.
- $\beta_c$ acts as an adaptive bias, shifting activation thresholds to suppress interfering speaker responses.
- Because $e$ is static throughout a session, $\gamma$ and $\beta$ can be pre-computed once during session initialization.

---

## 3.1 Recurrent Bottleneck Compression (Cache-Resident Topology)

In streaming real-time audio, model weights must reside in the CPU's fast L2/L3 cache (16–32 MB shared) to eliminate memory bus stalls at 100 FPS (10ms hops).

Directly flattening encoder features ($C=64$, $F=129$) into the LSTM would produce an input dimension of $64 \times 129 = 8{,}256$, ballooning the first LSTM layer to $4.23\text{M}$ parameters ($21.2\text{ MB}$ total ONNX size).

To resolve this, we introduce a causal 2D bottleneck compression layer prior to recurrence:
- **Down-projection**: Conv2d ($C_{in}=64 \to C_{out}=16$, kernel $3 \times 1$, stride $4 \times 1$, padding $1 \times 0$) reduces frequency from $129 \to 33$ bins.
- **Compressed LSTM input**: $16 \times 33 = 528$ dimensions.
- **Parameter footprint**: Slashes recurrent weights by $93.6\%$, dropping overall ONNX model size from $21.2\text{ MB}$ to **$2.33\text{ MB}$** (569K parameters).
- **Up-projection**: ConvTranspose2d ($C_{in}=16 \to C_{out}=64$, kernel $3 \times 1$, stride $4 \times 1$, padding $1 \times 0$) reconstructs the full $129$-bin feature map for the complex decoder.

---

## 4. Complex Ideal Ratio Masking (cIRM)

Magnitude-only masking discards phase information, leaving noisy phase intact and causing audible roughness. The network directly estimates a complex mask:

$$M(m, k) = M_r(m, k) + j M_i(m, k)$$

Applying this mask to the input spectrum $X(m, k)$:

$$\hat{S}(m, k) = X(m, k) \cdot M(m, k) = (X_r M_r - X_i M_i) + j (X_r M_i + X_i M_r)$$

To stabilize gradient dynamics and constrain unbounded outputs, the raw network outputs $Y_r, Y_i$ are bounded via hyperbolic tangent:

$$M_r(m, k) = K_m \tanh(Y_r(m, k)), \quad M_i(m, k) = K_m \tanh(Y_i(m, k))$$

where $K_m = 1.0$ (or $1.2$) ensures the mask can scale energy when needed while preventing exploding amplitudes.

---

## 5. Spectral Floor Clamping (Preventing Musical Noise)

When an extraction model aggressively suppresses interfering audio bins, it creates deep notches (isolated zero-energy islands) in the output spectrum. When these holes fluctuate rapidly frame-to-frame, the listener hears "musical noise" or watery chirping.

To resolve this, we enforce a floor $\alpha_{\min} \in [0.02, 0.05]$ ($-34\text{ dB}$ to $-26\text{ dB}$):

$$|M(m, k)| = \sqrt{M_r^2(m, k) + M_i^2(m, k)}$$

$$\text{If } |M(m, k)| < \alpha_{\min}: \quad M(m, k) \leftarrow M(m, k) \cdot \frac{\alpha_{\min}}{|M(m, k)|}$$

This preserves a smooth, imperceptible acoustic floor, preventing ear fatigue without sacrificing perceived isolation.

---

## 6. Personal VAD (pVAD) Anti-Hallucination Gate

Under pure TSE, when the target speaker is completely silent and a background talker is loud, non-linear network bleed can produce low-volume "ghost whispers."

We attach a Personal Voice Activity Detection (pVAD) head to the recurrent bottleneck:

$$P(\text{target}_m) = \sigma(W_{\text{vad}} \text{Pool}(H_m) + b_{\text{vad}})$$

The scalar probability drives a smooth gain state machine $g_m$:

$$g_m = \begin{cases}
1.0 & \text{if } P(\text{target}_m) \ge 0.6 \\
0.0 & \text{if } P(\text{target}_m) \le 0.3 \text{ and hangover elapsed} \\
\lambda g_{m-1} & \text{decay during 120ms hangover window}
\end{cases}$$

The final time-frequency representation passed to synthesis is:

$$\hat{S}_{\text{gated}}(m, k) = g_m \cdot \hat{S}(m, k)$$

---

## 7. Causal iSTFT & Overlap-Add (OLA) Synthesis

1. Calculate 512-point Inverse Fast Fourier Transform (iFFT) on $\hat{S}_{\text{gated}}(m, k)$:

$$\hat{s}_{\text{frame}}[n] = \frac{1}{K} \sum_{k=0}^{K-1} \hat{S}_{\text{gated}}(m, k) \cdot e^{j \frac{2\pi k n}{K}}, \quad 0 \le n < N$$

2. Multiply by synthesis window $w_s[n]$ (equal to analysis window $w[n]$ for square-root Hann):

$$\hat{s}_{\text{win}}[n] = \hat{s}_{\text{frame}}[n] \cdot w_s[n]$$

3. Add into synthesis overlap buffer:

$$\text{out}[mR + n] \leftarrow \text{out}[mR + n] + \hat{s}_{\text{win}}[n]$$

4. The oldest $R = 160\text{ samples}$ ($10\text{ ms}$) in the overlap buffer are fully reconstructed and ready to be pushed to the audio output queue.
