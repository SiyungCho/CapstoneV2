import argparse
import glob
import os
import time
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

CHANNEL_COLORS = [
    (255, 99, 71),
    (60, 179, 113),
    (30, 144, 255),
    (238, 130, 238),
    (255, 215, 0),
    (64, 224, 208),
    (205, 92, 92),
    (173, 255, 47),
]


def find_latest_multimodal_pickle() -> str:
    candidates = glob.glob("multimodal_data_*.pkl")
    if not candidates:
        raise FileNotFoundError("No files matching multimodal_data_*.pkl found in current directory.")
    return max(candidates, key=os.path.getmtime)


def load_multimodal_pickle(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = pd.read_pickle(path)
    if not isinstance(payload, dict):
        raise ValueError("Pickle must contain a dict with 'hand_data' and 'eit_data'.")
    if "hand_data" not in payload or "eit_data" not in payload:
        raise ValueError("Pickle dict is missing 'hand_data' and/or 'eit_data'.")

    hand_df = payload["hand_data"].copy()
    eit_df = payload["eit_data"].copy()

    if hand_df.empty:
        raise ValueError("'hand_data' is empty.")
    if eit_df.empty:
        raise ValueError("'eit_data' is empty.")

    return hand_df, eit_df


def preprocess_hand(hand_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [f"lm{i}_{axis}" for i in range(21) for axis in ("x", "y")]
    missing = [c for c in required_cols if c not in hand_df.columns]
    if missing:
        raise ValueError(f"Missing hand landmark columns: {missing[:4]} ...")

    hand_df = hand_df.dropna(subset=["timestamp", "lm0_x", "lm0_y"]).copy()
    hand_df = hand_df.sort_values("timestamp").reset_index(drop=True)

    if hand_df.empty:
        raise ValueError("No valid hand landmark rows found after filtering.")

    return hand_df


def preprocess_eit(eit_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    required_cols = ["timestamp", "Frame_ID", "Inj", "Magnitude"]
    missing = [c for c in required_cols if c not in eit_df.columns]
    if missing:
        raise ValueError(f"Missing EIT columns: {missing}")

    valid = eit_df.dropna(subset=required_cols).copy()

    if "Phase" in valid.columns:
        measurement = valid[valid["Phase"] == "Measurement"]
        if not measurement.empty:
            valid = measurement

    valid["Frame_ID"] = valid["Frame_ID"].astype(int)
    valid["Inj"] = valid["Inj"].astype(int)

    frame_ts = valid.groupby("Frame_ID", as_index=True)["timestamp"].mean()
    inj_means = valid.groupby(["Frame_ID", "Inj"], as_index=True)["Magnitude"].mean().unstack("Inj")
    inj_means = inj_means.reindex(columns=range(8))

    merged = inj_means.join(frame_ts, how="inner")
    merged = merged.sort_values("timestamp").dropna(subset=["timestamp"])

    if merged.empty:
        raise ValueError("No valid per-frame EIT signal data found.")

    eit_timestamps = merged["timestamp"].to_numpy(dtype=float)
    eit_values = merged[range(8)].to_numpy(dtype=float)

    return eit_timestamps, eit_values


def draw_hand_panel(canvas: np.ndarray, hand_row: pd.Series) -> None:
    h, w = canvas.shape[:2]
    points = []

    for idx in range(21):
        x = float(hand_row[f"lm{idx}_x"])
        y = float(hand_row[f"lm{idx}_y"])
        px = int(np.clip(x * w, 0, w - 1))
        py = int(np.clip(y * h, 0, h - 1))
        points.append((px, py))

    for start, end in HAND_CONNECTIONS:
        cv2.line(canvas, points[start], points[end], (80, 180, 255), 2, cv2.LINE_AA)

    for p in points:
        cv2.circle(canvas, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, p, 4, (40, 40, 40), 1, cv2.LINE_AA)


def draw_signal_panel(canvas: np.ndarray, signal_history: deque, current_values: np.ndarray) -> None:
    h, w = canvas.shape[:2]
    margin_left, margin_right = 60, 130
    margin_top, margin_bottom = 20, 35
    x0, y0 = margin_left, margin_top
    x1, y1 = w - margin_right, h - margin_bottom

    cv2.rectangle(canvas, (x0, y0), (x1, y1), (220, 220, 220), 1, cv2.LINE_AA)

    if len(signal_history) == 0:
        return

    data = np.array(signal_history, dtype=np.float32)

    finite = np.isfinite(data)
    if not finite.any():
        return

    finite_data = data[finite]
    y_min = float(np.min(finite_data))
    y_max = float(np.max(finite_data))

    if abs(y_max - y_min) < 1e-6:
        y_max += 1.0
        y_min -= 1.0

    span = y_max - y_min
    y_min -= span * 0.1
    y_max += span * 0.1

    def map_y(value: float) -> int:
        norm = (value - y_min) / (y_max - y_min)
        return int(y1 - np.clip(norm, 0, 1) * (y1 - y0))

    zero_y = map_y(0.0)
    if y0 <= zero_y <= y1:
        cv2.line(canvas, (x0, zero_y), (x1, zero_y), (100, 100, 100), 1, cv2.LINE_AA)

    n = data.shape[0]
    for channel in range(8):
        series = data[:, channel]
        pts = []
        for i, value in enumerate(series):
            x = x0 if n == 1 else int(x0 + i * (x1 - x0) / (n - 1))
            y = map_y(float(value))
            pts.append((x, y))

        if len(pts) >= 2:
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, CHANNEL_COLORS[channel], 2, cv2.LINE_AA)
        elif len(pts) == 1:
            cv2.circle(canvas, pts[0], 2, CHANNEL_COLORS[channel], -1, cv2.LINE_AA)

        label_y = y0 + 16 + channel * 18
        label = f"I{channel}: {current_values[channel]:.2f}"
        cv2.putText(canvas, label, (x1 + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CHANNEL_COLORS[channel], 1, cv2.LINE_AA)

    cv2.putText(canvas, f"{y_max:.2f}", (5, y0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{y_min:.2f}", (5, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "8 mean EIT signals (per injection)", (x0, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)


def is_opencv_draw_usable() -> bool:
    try:
        test = np.zeros((8, 8, 3), dtype=np.uint8)
        cv2.line(test, (0, 0), (7, 7), (255, 255, 255), 1, cv2.LINE_AA)
        return True
    except cv2.error:
        return False


def run_replay_with_opencv(
    hand_df: pd.DataFrame,
    hand_timestamps: np.ndarray,
    eit_timestamps: np.ndarray,
    eit_values: np.ndarray,
    speed: float,
    history_size: int,
) -> None:
    landmark_cols = [f"lm{i}_{axis}" for i in range(21) for axis in ("x", "y")]

    window_name = "Multimodal Replay"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    top_h, bottom_h, width = 540, 300, 1000
    signal_history: deque[np.ndarray] = deque(maxlen=history_size)

    start_recording_ts = min(hand_timestamps[0], eit_timestamps[0])
    end_recording_ts = max(hand_timestamps[-1], eit_timestamps[-1])
    playback_start_wall = time.perf_counter()
    hand_index = 0
    eit_index = 0
    current_hand_index: int | None = None
    current_eit_values = np.full(8, np.nan, dtype=np.float32)
    paused = False
    pause_started = 0.0

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord(" "):
            if paused:
                paused = False
                paused_duration = time.perf_counter() - pause_started
                playback_start_wall += paused_duration
            else:
                paused = True
                pause_started = time.perf_counter()

        if not paused:
            elapsed_wall = time.perf_counter() - playback_start_wall
            target_recording_ts = start_recording_ts + elapsed_wall * speed

            while hand_index < len(hand_timestamps) and hand_timestamps[hand_index] <= target_recording_ts:
                current_hand_index = hand_index
                hand_index += 1

            while eit_index < len(eit_timestamps) and eit_timestamps[eit_index] <= target_recording_ts:
                current_eit_values = eit_values[eit_index]
                signal_history.append(current_eit_values)
                eit_index += 1

            if hand_index >= len(hand_timestamps) and eit_index >= len(eit_timestamps) and target_recording_ts >= end_recording_ts:
                paused = True

        hand_canvas = np.zeros((top_h, width, 3), dtype=np.uint8)
        signal_canvas = np.zeros((bottom_h, width, 3), dtype=np.uint8)

        if current_hand_index is not None:
            current_row = hand_df.iloc[current_hand_index]
            hand_sample = current_row[landmark_cols]
            draw_hand_panel(hand_canvas, hand_sample)
        else:
            cv2.putText(hand_canvas, "No hand sample yet", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 220), 2, cv2.LINE_AA)

        draw_signal_panel(signal_canvas, signal_history, current_eit_values)
        if len(signal_history) == 0:
            cv2.putText(signal_canvas, "No EIT sample yet", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 220), 2, cv2.LINE_AA)

        cv2.putText(hand_canvas, "Hand reconstruction", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(hand_canvas, f"Frame {(0 if current_hand_index is None else current_hand_index + 1)}/{len(hand_df)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(
            hand_canvas,
            f"Playback speed: {speed:.2f}x | Space: pause/resume | Q/Esc: quit",
            (20, top_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        frame = np.vstack([hand_canvas, signal_canvas])
        cv2.imshow(window_name, frame)

        if paused and hand_index >= len(hand_timestamps) and eit_index >= len(eit_timestamps):
            break

    cv2.destroyAllWindows()


def run_replay_with_matplotlib(
    hand_df: pd.DataFrame,
    hand_timestamps: np.ndarray,
    eit_timestamps: np.ndarray,
    eit_values: np.ndarray,
    speed: float,
    history_size: int,
) -> None:
    landmark_cols = [f"lm{i}_{axis}" for i in range(21) for axis in ("x", "y")]
    signal_history: deque[np.ndarray] = deque(maxlen=history_size)

    plt.style.use("dark_background")
    fig = plt.figure("Multimodal Replay", figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    ax_hand = fig.add_subplot(gs[0])
    ax_sig = fig.add_subplot(gs[1])

    ax_hand.set_title("Hand reconstruction")
    ax_hand.set_xlim(0.0, 1.0)
    ax_hand.set_ylim(1.0, 0.0)
    ax_hand.set_aspect("equal", adjustable="box")
    ax_hand.set_xticks([])
    ax_hand.set_yticks([])

    hand_lines = []
    for _ in HAND_CONNECTIONS:
        line, = ax_hand.plot([], [], lw=2)
        hand_lines.append(line)
    hand_scatter = ax_hand.scatter([], [], s=26, c="white")
    info_text = ax_hand.text(0.02, 0.95, "", transform=ax_hand.transAxes, va="top")

    ax_sig.set_title("8 mean EIT signals (per injection)")
    ax_sig.set_xlim(0, max(16, history_size))
    ax_sig.grid(alpha=0.25)
    channel_lines = []
    for idx, bgr in enumerate(CHANNEL_COLORS):
        rgb = tuple(v / 255.0 for v in (bgr[2], bgr[1], bgr[0]))
        line, = ax_sig.plot([], [], lw=1.8, color=rgb, label=f"I{idx}")
        channel_lines.append(line)
    ax_sig.legend(loc="upper right", ncol=4, fontsize=8)

    start_recording_ts = min(hand_timestamps[0], eit_timestamps[0])
    end_recording_ts = max(hand_timestamps[-1], eit_timestamps[-1])
    playback_start_wall = time.perf_counter()
    hand_index = 0
    eit_index = 0
    current_hand_index: int | None = None

    while plt.fignum_exists(fig.number):
        elapsed_wall = time.perf_counter() - playback_start_wall
        target_recording_ts = start_recording_ts + elapsed_wall * speed

        while hand_index < len(hand_timestamps) and hand_timestamps[hand_index] <= target_recording_ts:
            current_hand_index = hand_index
            hand_index += 1

        while eit_index < len(eit_timestamps) and eit_timestamps[eit_index] <= target_recording_ts:
            signal_history.append(eit_values[eit_index])
            eit_index += 1

        if current_hand_index is not None:
            row = hand_df.iloc[current_hand_index]
            hand_sample = row[landmark_cols]
            xs = [float(hand_sample[f"lm{i}_x"]) for i in range(21)]
            ys = [float(hand_sample[f"lm{i}_y"]) for i in range(21)]

            for line, (start, end) in zip(hand_lines, HAND_CONNECTIONS):
                line.set_data([xs[start], xs[end]], [ys[start], ys[end]])
            hand_scatter.set_offsets(np.column_stack([xs, ys]))
        else:
            hand_scatter.set_offsets(np.empty((0, 2)))
            for line in hand_lines:
                line.set_data([], [])

        hist_arr = np.array(signal_history, dtype=np.float32)
        if hist_arr.size > 0:
            x_vals = np.arange(hist_arr.shape[0])
            for channel in range(8):
                channel_lines[channel].set_data(x_vals, hist_arr[:, channel])
        else:
            x_vals = np.array([], dtype=np.float32)
            for channel in range(8):
                channel_lines[channel].set_data([], [])

        if hist_arr.size > 0 and np.isfinite(hist_arr).any():
            finite_vals = hist_arr[np.isfinite(hist_arr)]
            y_min = float(np.min(finite_vals))
            y_max = float(np.max(finite_vals))
            if abs(y_max - y_min) < 1e-6:
                y_max += 1.0
                y_min -= 1.0
            span = y_max - y_min
            ax_sig.set_ylim(y_min - 0.1 * span, y_max + 0.1 * span)

        ax_sig.set_xlim(0, max(history_size, len(x_vals)))
        frame_num = 0 if current_hand_index is None else current_hand_index + 1
        info_text.set_text(f"Frame {frame_num}/{len(hand_df)}  |  Playback {speed:.2f}x")
        plt.pause(0.001)

        if hand_index >= len(hand_timestamps) and eit_index >= len(eit_timestamps) and target_recording_ts >= end_recording_ts:
            break

    if plt.fignum_exists(fig.number):
        plt.show()


def run_replay(file_path: str, speed: float, history_size: int) -> None:
    hand_df, eit_df = load_multimodal_pickle(file_path)
    hand_df = preprocess_hand(hand_df)
    eit_timestamps, eit_values = preprocess_eit(eit_df)

    hand_timestamps = hand_df["timestamp"].to_numpy(dtype=float)
    if is_opencv_draw_usable():
        run_replay_with_opencv(hand_df, hand_timestamps, eit_timestamps, eit_values, speed, history_size)
    else:
        print("OpenCV drawing functions are incompatible with current numpy build. Switching to Matplotlib fallback...")
        run_replay_with_matplotlib(hand_df, hand_timestamps, eit_timestamps, eit_values, speed, history_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay multimodal hand + EIT data from a pickle file.")
    parser.add_argument("--file", type=str, default=None, help="Path to multimodal pickle file.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (e.g. 0.5, 1.0, 2.0).")
    parser.add_argument("--history", type=int, default=240, help="Number of points to keep in the EIT plot history.")
    args = parser.parse_args()

    if args.speed <= 0:
        raise ValueError("--speed must be > 0")
    if args.history < 16:
        raise ValueError("--history must be at least 16")

    file_path = args.file if args.file else find_latest_multimodal_pickle()
    print(f"Replaying: {file_path}")
    run_replay(file_path=file_path, speed=args.speed, history_size=args.history)


if __name__ == "__main__":
    main()
