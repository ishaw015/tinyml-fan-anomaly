"""
Isha's part: data -> windows -> 64-bin FFT features -> RMS rule baseline.

Run:  python isha_pipeline.py
Outputs (in ./data): train_healthy.npy, val_healthy.npy, test_X.npy, test_y.npy,
                     test_speed.npy, norm.npy, rms_threshold.npy, sample_windows.npy

IMPORTANT: the data here is SYNTHETIC, for debugging the pipeline only.
Replace make_dataset() with a loader for a real dataset (e.g. MaFaulDa,
downsampled to 1 kHz) before reporting any results.
"""
import os
import numpy as np

FS = 1000          # sample rate (Hz)  -- agreed with the team
WIN = 256          # window length
HOP = 128          # 50% overlap
NBINS = 64         # model input size
RNG = np.random.default_rng(0)

SPEEDS = {"low": 16.0, "med": 24.0, "high": 33.0}   # rotation freq (Hz)


# ---------------------------------------------------------------- data
def synth_run(speed_hz, fault, session_gain, seconds=40):
    """One recording. Imbalance force grows with speed^2, like real machines."""
    t = np.arange(int(seconds * FS)) / FS
    f0 = speed_hz * (1 + RNG.normal(0, 0.01))            # small speed wobble
    amp1 = 0.02 * (speed_hz / 16.0) ** 2                 # healthy 1x amplitude
    if fault == "imbalance":
        amp1 *= 2.5
    x = amp1 * np.sin(2 * np.pi * f0 * t + RNG.uniform(0, 6.28))
    x += 0.3 * amp1 * np.sin(2 * np.pi * 2 * f0 * t)     # 2x harmonic
    if fault == "looseness":
        x += 1.2 * amp1 * np.sin(2 * np.pi * 3 * f0 * t)
        x += 0.8 * amp1 * np.sin(2 * np.pi * 0.5 * f0 * t)
    x *= session_gain                                    # mounting changes
    x += RNG.normal(0, 0.01, size=t.shape)               # sensor noise
    x += 1.0                                             # gravity offset
    return x


def make_dataset():
    """Returns list of (signal, label, speed_name, session). label 0=healthy."""
    runs = []
    for session in range(4):
        gain = RNG.uniform(0.85, 1.15)
        for sname, hz in SPEEDS.items():
            runs.append((synth_run(hz, "healthy", gain), 0, sname, session))
            if session >= 2:  # faults only appear in val/test sessions
                runs.append((synth_run(hz, "imbalance", gain), 1, sname, session))
                runs.append((synth_run(hz, "looseness", gain), 1, sname, session))
    return runs


# ---------------------------------------------------------------- features
HANN = np.hanning(WIN)


def windows(sig):
    n = (len(sig) - WIN) // HOP + 1
    return np.stack([sig[i * HOP:i * HOP + WIN] for i in range(n)])


def rms(w):
    w = w - w.mean(axis=1, keepdims=True)                # remove gravity
    return np.sqrt((w ** 2).mean(axis=1))


def spectrum(w):
    """64-bin log-magnitude spectrum. MUST be mirrored exactly in C."""
    w = w - w.mean(axis=1, keepdims=True)
    mag = np.abs(np.fft.rfft(w * HANN, axis=1))[:, :128]  # 0..~496 Hz
    mag = mag.reshape(len(w), 64, 2).mean(axis=2)         # average bin pairs
    return np.log1p(mag * 100.0)


# ---------------------------------------------------------------- main
def main():
    os.makedirs("data", exist_ok=True)
    runs = make_dataset()

    tr, va, te = [], [], []
    te_y, te_speed, te_rms_all = [], [], []
    for sig, label, sname, session in runs:
        w = windows(sig)
        if session < 2:
            tr.append(w)
        elif session == 2 and label == 0:
            va.append(w)
        else:  # test = session 3 (healthy + faults)
            if session == 3:
                te.append(w)
                te_y += [label] * len(w)
                te_speed += [sname] * len(w)

    tr, va, te = np.concatenate(tr), np.concatenate(va), np.concatenate(te)
    te_y, te_speed = np.array(te_y), np.array(te_speed)

    # ---- features + normalisation (stats from healthy TRAIN only)
    Xtr, Xva, Xte = spectrum(tr), spectrum(va), spectrum(te)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-6
    norm = lambda X: ((X - mu) / sd).astype(np.float32)
    np.save("data/train_healthy.npy", norm(Xtr))
    np.save("data/val_healthy.npy", norm(Xva))
    np.save("data/test_X.npy", norm(Xte))
    np.save("data/test_y.npy", te_y)
    np.save("data/test_speed.npy", te_speed)
    np.save("data/norm.npy", np.stack([mu, sd]))          # Pranavi needs these

    # small replay set for the ESP32 / Wokwi (raw windows, 8 per case)
    replay = np.concatenate([te[(te_y == 0) & (te_speed == "med")][:8],
                             te[(te_y == 0) & (te_speed == "high")][:8],
                             te[(te_y == 1) & (te_speed == "med")][:8]])
    np.save("data/sample_windows.npy", replay.astype(np.float32))

    # ---- RMS rule baseline: threshold = 99th percentile of healthy val RMS
    thr = np.percentile(rms(va), 99)
    np.save("data/rms_threshold.npy", np.array([thr]))
    pred = (rms(te) > thr).astype(int)

    print(f"windows: train {len(tr)}, val {len(va)}, test {len(te)}")
    print(f"RMS threshold (99th pct of healthy val): {thr:.4f}\n")
    tp = ((pred == 1) & (te_y == 1)).sum()
    fn = ((pred == 0) & (te_y == 1)).sum()
    fp = ((pred == 1) & (te_y == 0)).sum()
    tn = ((pred == 0) & (te_y == 0)).sum()
    print(f"Overall: detection {tp/(tp+fn):.2f}, false alarm {fp/(fp+tn):.2f}")
    print("\nRMS rule by speed:")
    for s in SPEEDS:
        m = te_speed == s
        det = pred[m & (te_y == 1)].mean()
        fa = pred[m & (te_y == 0)].mean()
        print(f"  {s:5s} fault detected {det:.2f} | false alarm {fa:.2f}")


if __name__ == "__main__":
    main()
