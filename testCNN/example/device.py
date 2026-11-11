"""
Memristive device model based on measured I-V characteristics.

Uses a nonlinear soft-bounds model:
  LTP: G += alpha_p * (G_max - G)    per pulse
  LTD: G -= alpha_d * (G - G_min)    per pulse

Number of effective levels is calibrated from ltp_levels / ltd_levels.
Supports gradient accumulation for sub-threshold updates.
"""
import numpy as np
import pandas as pd


class DeviceModel:
    def __init__(self, csv_path: str = "data/sample_points.csv", V_bias: float = -5.0,
                 n_G_bins: int = 200, max_pulses: int = 128,
                 ltp_levels: int = 32, ltd_levels: int = 16):
        self.V_bias = V_bias
        self.n_G_bins = n_G_bins
        self.max_pulses = max_pulses
        self._load_data(csv_path)
        self._extract_segments()
        self._calibrate(ltp_levels, ltd_levels)
        self._build_lut()

    def _load_data(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.I = df["Id"].values
        self.G_raw = np.abs(self.I / self.V_bias)
        self.G_min = self.G_raw.min()
        self.G_max = self.G_raw.max()
        self.G_span = self.G_max - self.G_min
        self.G_mid_val = (self.G_min + self.G_max) / 2.0

    def _extract_segments(self):
        dG = np.diff(self.G_raw)
        signs = np.sign(dG)
        signs[signs == 0] = 1
        ltp_segs, ltd_segs = [], []
        start = 0
        for i in range(1, len(signs)):
            if signs[i] != signs[i - 1] or i == len(signs) - 1:
                seg = self.G_raw[start:i + 1]
                if len(seg) >= 5:
                    if signs[i - 1] > 0:
                        ltp_segs.append(seg)
                    else:
                        ltd_segs.append(seg)
                start = i
        self.ltp_segments = ltp_segs
        self.ltd_segments = ltd_segs

    def _calibrate(self, ltp_levels, ltd_levels):
        """Set alpha so that N pulses cover 99% of the span from G_min (LTP) or G_max (LTD)."""
        self.ltp_levels = ltp_levels
        self.ltd_levels = ltd_levels
        self.alpha_p = 1.0 - 0.01 ** (1.0 / max(ltp_levels, 1))
        self.alpha_d = 1.0 - 0.01 ** (1.0 / max(ltd_levels, 1))
        print(f"    alpha_p={self.alpha_p:.5f} ({ltp_levels} LTP lvls), "
              f"alpha_d={self.alpha_d:.5f} ({ltd_levels} LTD lvls)")

    # ---------- forward / inverse for single-element ----------
    def _ltp_forward(self, G_start, n_pulses):
        if n_pulses <= 0:
            return G_start
        decay = (1.0 - self.alpha_p) ** n_pulses
        return self.G_max - (self.G_max - G_start) * decay

    def _ltd_forward(self, G_start, n_pulses):
        if n_pulses <= 0:
            return G_start
        decay = (1.0 - self.alpha_d) ** n_pulses
        return self.G_min + (G_start - self.G_min) * decay

    def _ltp_pulses(self, G_start, G_target):
        if G_target <= G_start + 1e-20:
            return 0.0
        gap_s = max(self.G_max - G_start, 1e-20)
        gap_t = max(self.G_max - G_target, 1e-20)
        ratio = np.clip(gap_t / gap_s, 1e-15, 1.0)
        p = np.log(ratio) / np.log(1.0 - self.alpha_p)
        return min(p, float(self.max_pulses))

    def _ltd_pulses(self, G_start, G_target):
        if G_target >= G_start - 1e-20:
            return 0.0
        gap_s = max(G_start - self.G_min, 1e-20)
        gap_t = max(G_target - self.G_min, 1e-20)
        ratio = np.clip(gap_t / gap_s, 1e-15, 1.0)
        p = np.log(ratio) / np.log(1.0 - self.alpha_d)
        return min(p, float(self.max_pulses))

    # ---------- LUT ----------
    def _build_lut(self):
        n = self.n_G_bins
        self.lut_G = np.linspace(self.G_min, self.G_max, n)
        self.lut_pulses_ltp = np.zeros((n, n), dtype=np.float32)
        self.lut_pulses_ltd = np.zeros((n, n), dtype=np.float32)
        self.lut_new_G_ltp  = np.zeros((n, n), dtype=np.float32)
        self.lut_new_G_ltd  = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            Gi = self.lut_G[i]
            for j in range(i, n):
                Gj = self.lut_G[j]
                p = self._ltp_pulses(Gi, Gj)
                pi = int(np.clip(np.round(p), 0, self.max_pulses))
                self.lut_pulses_ltp[i, j] = pi
                self.lut_new_G_ltp[i, j] = self._ltp_forward(Gi, pi)
            for j in range(0, i + 1):
                Gj = self.lut_G[j]
                p = self._ltd_pulses(Gi, Gj)
                pi = int(np.clip(np.round(p), 0, self.max_pulses))
                self.lut_pulses_ltd[i, j] = pi
                self.lut_new_G_ltd[i, j] = self._ltd_forward(Gi, pi)

    # ---------- vectorized update ----------
    def apply_pulses_vectorized(self, G_current, delta_G):
        shape = G_current.shape
        G_flat = G_current.ravel().astype(np.float64)
        dG_flat = delta_G.ravel().astype(np.float64)
        G_target = np.clip(G_flat + dG_flat, self.G_min, self.G_max)

        edges = np.linspace(self.G_min, self.G_max, self.n_G_bins + 1)
        bi = np.clip(np.digitize(G_flat, edges) - 1, 0, self.n_G_bins - 1)
        tj = np.clip(np.digitize(G_target, edges) - 1, 0, self.n_G_bins - 1)

        new_G = np.zeros_like(G_flat)
        pulses = np.zeros_like(G_flat, dtype=np.int32)

        m_ltp = dG_flat > 0
        m_ltd = dG_flat < 0
        m_zero = ~m_ltp & ~m_ltd

        if m_ltp.any():
            pulses[m_ltp] = self.lut_pulses_ltp[bi[m_ltp], tj[m_ltp]].astype(np.int32)
            new_G[m_ltp] = self.lut_new_G_ltp[bi[m_ltp], tj[m_ltp]]
        if m_ltd.any():
            pulses[m_ltd] = self.lut_pulses_ltd[bi[m_ltd], tj[m_ltd]].astype(np.int32)
            new_G[m_ltd] = self.lut_new_G_ltd[bi[m_ltd], tj[m_ltd]]
        if m_zero.any():
            new_G[m_zero] = G_flat[m_zero]

        return new_G.reshape(shape), pulses.reshape(shape)

    def apply_pulses(self, G_current, delta_G_target):
        new_G, pulses = self.apply_pulses_vectorized(
            np.array([G_current]), np.array([delta_G_target]))
        return float(new_G[0]), int(pulses[0])

    # ---------- G <-> weight mapping ----------
    def G_to_weight(self, G, G_ref=None):
        if G_ref is None:
            G_ref = self.G_mid_val
        return (G - G_ref) / (self.G_max - self.G_min) * 2.0

    def weight_to_G(self, w, G_ref=None):
        if G_ref is None:
            G_ref = self.G_mid_val
        return np.clip(G_ref + w * (self.G_max - self.G_min) / 2.0,
                       self.G_min, self.G_max)

    @property
    def G_range(self):
        return self.G_min, self.G_max

    @property
    def G_mid(self):
        return self.G_mid_val
