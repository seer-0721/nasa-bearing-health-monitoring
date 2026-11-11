from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DEFAULT_TAU_MS = 20.0
DEFAULT_A_PLUS = 80.0
DEFAULT_A_MINUS = -60.0


def load_current_delta(
    filepath: str | Path,
    baseline_frac: float = 0.1,
    tail_frac: float = 0.1,
) -> float:
    data = pd.read_csv(filepath, sep="\t", comment="#", header=0)
    current = data["Id"].values

    baseline_count = max(int(len(current) * baseline_frac), 1)
    tail_count = max(int(len(current) * tail_frac), 1)
    baseline = np.mean(current[:baseline_count])
    steady = np.mean(current[-tail_count:])
    return float((steady - baseline) * 1e9)


def stdp_window(
    dt: np.ndarray,
    A_plus: float,
    tau_plus: float,
    A_minus: float,
    tau_minus: float,
) -> np.ndarray:
    result = np.zeros_like(dt, dtype=float)
    pos = dt > 0
    neg = dt < 0
    result[pos] = A_plus * np.exp(-dt[pos] / tau_plus)
    result[neg] = A_minus * np.exp(dt[neg] / tau_minus)
    return result


def _fit_side(
    dt_ms: np.ndarray,
    dw_nA: np.ndarray,
    *,
    sign: int,
    fixed_tau_ms: float,
    min_tau_ms: float,
    max_tau_ms: float,
    sparse_amplitude_mode: str,
) -> tuple[float, float]:
    if len(dt_ms) == 0:
        if sign > 0:
            return DEFAULT_A_PLUS, fixed_tau_ms
        return DEFAULT_A_MINUS, fixed_tau_ms

    magnitude = dw_nA if sign > 0 else -dw_nA
    magnitude = np.maximum(magnitude, 0.0)

    if len(dt_ms) == 1:
        if sparse_amplitude_mode == "observed":
            amp = magnitude[0]
            if sign < 0:
                amp = -amp
            return float(amp), fixed_tau_ms

        if sign > 0:
            amp = magnitude[0] / np.exp(-dt_ms[0] / fixed_tau_ms)
            return float(amp), fixed_tau_ms
        amp = magnitude[0] / np.exp(dt_ms[0] / fixed_tau_ms)
        return float(-amp), fixed_tau_ms

    def side_window(dt: np.ndarray, amp: float, tau: float) -> np.ndarray:
        if sign > 0:
            return amp * np.exp(-dt / tau)
        return amp * np.exp(dt / tau)

    amp0 = max(float(np.max(magnitude)), 1e-9)
    amp_upper = max(amp0 * 100.0, 1.0)
    amp, tau = curve_fit(
        side_window,
        dt_ms,
        magnitude,
        p0=[amp0, fixed_tau_ms],
        bounds=([0.0, min_tau_ms], [amp_upper, max_tau_ms]),
        maxfev=10000,
    )[0]
    if sign < 0:
        amp = -amp
    return float(amp), float(tau)


def fit_stdp_params(
    dt_ms: np.ndarray,
    dw_nA: np.ndarray,
    *,
    fixed_tau_ms: float,
    min_tau_ms: float,
    max_tau_ms: float,
    sparse_amplitude_mode: str,
) -> tuple[float, float, float, float]:
    pos = dt_ms > 0
    neg = dt_ms < 0

    if np.count_nonzero(pos) < 2 or np.count_nonzero(neg) < 2:
        print(
            "Not enough points for full STDP tau fitting; "
            f"using fixed tau={fixed_tau_ms:g} ms for sparse side(s)."
        )

    A_plus, tau_plus = _fit_side(
        dt_ms[pos],
        dw_nA[pos],
        sign=1,
        fixed_tau_ms=fixed_tau_ms,
        min_tau_ms=min_tau_ms,
        max_tau_ms=max_tau_ms,
        sparse_amplitude_mode=sparse_amplitude_mode,
    )
    A_minus, tau_minus = _fit_side(
        dt_ms[neg],
        dw_nA[neg],
        sign=-1,
        fixed_tau_ms=fixed_tau_ms,
        min_tau_ms=min_tau_ms,
        max_tau_ms=max_tau_ms,
        sparse_amplitude_mode=sparse_amplitude_mode,
    )
    return A_plus, tau_plus, A_minus, tau_minus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit STDP parameters from device current traces."
    )
    parser.add_argument("--ltp-file", type=str, default="20cyc.txt")
    parser.add_argument("--ltd-file", type=str, default="8cyc.txt")
    parser.add_argument("--ltp-dt-ms", type=float, default=10.0)
    parser.add_argument("--ltd-dt-ms", type=float, default=-30.0)
    parser.add_argument("--output", type=str, default="stdp_params.npz")
    parser.add_argument("--fixed-tau-ms", type=float, default=DEFAULT_TAU_MS)
    parser.add_argument("--min-tau-ms", type=float, default=1.0)
    parser.add_argument("--max-tau-ms", type=float, default=200.0)
    parser.add_argument(
        "--sparse-amplitude-mode",
        choices=["observed", "extrapolate"],
        default="observed",
        help=(
            "How to set A_plus/A_minus when a side has only one data point. "
            "'observed' uses the measured dI directly; 'extrapolate' estimates "
            "the dt=0 amplitude with the fixed tau."
        ),
    )
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = [
        {"file": args.ltp_file, "dt_ms": args.ltp_dt_ms, "name": "LTP"},
        {"file": args.ltd_file, "dt_ms": args.ltd_dt_ms, "name": "LTD"},
    ]

    dt_list = []
    dw_list = []
    for exp in experiments:
        dw_nA = load_current_delta(exp["file"])
        dt_list.append(exp["dt_ms"])
        dw_list.append(dw_nA)
        print(f"{exp['file']}: dt={exp['dt_ms']:+.1f} ms, dI={dw_nA:.2f} nA")

    dt_arr = np.array(dt_list)
    dw_arr = np.array(dw_list)

    try:
        A_plus, tau_plus, A_minus, tau_minus = fit_stdp_params(
            dt_arr,
            dw_arr,
            fixed_tau_ms=args.fixed_tau_ms,
            min_tau_ms=args.min_tau_ms,
            max_tau_ms=args.max_tau_ms,
            sparse_amplitude_mode=args.sparse_amplitude_mode,
        )
    except Exception as exc:
        print(f"Fit failed, using fallback parameters: {exc}")
        A_plus, tau_plus, A_minus, tau_minus = (
            DEFAULT_A_PLUS,
            DEFAULT_TAU_MS,
            DEFAULT_A_MINUS,
            DEFAULT_TAU_MS,
        )

    print("RESULT")
    print(f"  A_plus={A_plus:.6g} nA")
    print(f"  tau_plus={tau_plus:.6g} ms")
    print(f"  A_minus={A_minus:.6g} nA")
    print(f"  tau_minus={tau_minus:.6g} ms")

    np.savez(
        args.output,
        A_plus=A_plus,
        tau_plus=tau_plus,
        A_minus=A_minus,
        tau_minus=tau_minus,
    )
    print(f"Saved STDP parameters to {args.output}")

    if args.plot:
        dt_smooth = np.linspace(-50, 50, 200)
        dw_smooth = stdp_window(dt_smooth, A_plus, tau_plus, A_minus, tau_minus)
        plt.scatter(dt_arr, dw_arr, label="data")
        plt.plot(dt_smooth, dw_smooth, "r-", label="fit")
        plt.xlabel("dt (ms)")
        plt.ylabel("dI (nA)")
        plt.axhline(0, color="k", linestyle="--")
        plt.axvline(0, color="k", linestyle="--")
        plt.legend()
        plt.title("Device STDP Window")
        plt.show()


if __name__ == "__main__":
    main()
