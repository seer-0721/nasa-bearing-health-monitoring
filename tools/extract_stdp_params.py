from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit STDP parameters from device current traces.")
    parser.add_argument("--ltp-file", type=str, default="20cyc.txt")
    parser.add_argument("--ltd-file", type=str, default="8cyc.txt")
    parser.add_argument("--ltp-dt-ms", type=float, default=10.0)
    parser.add_argument("--ltd-dt-ms", type=float, default=-30.0)
    parser.add_argument("--output", type=str, default="stdp_params.npz")
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
    p0 = [100.0, 20.0, -50.0, 20.0]

    try:
        A_plus, tau_plus, A_minus, tau_minus = curve_fit(
            stdp_window,
            dt_arr,
            dw_arr,
            p0=p0,
            maxfev=10000,
        )[0]
    except Exception as exc:
        print(f"Fit failed, using fallback parameters: {exc}")
        A_plus, tau_plus, A_minus, tau_minus = 80.0, 20.0, -60.0, 20.0

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
