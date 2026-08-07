"""Drive the xArm gripper live from the Olimex EMG-shield stream, with a
live raw+envelope plot so you can see whether your movement registered.

Reuses the trained model, scaler, extract_features(), predict() and
move_gripper() from Testing_robot_arm_V3.py, and the serial packet parsing
and envelope math (0xA5/0x5A sync, big-endian channel value, 30-sample
moving-average envelope) from olimex-emg_class_v2.py's acquisition script,
so live features are computed exactly the way the training data was.
"""

import struct
from collections import deque

import numpy as np
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Testing_robot_arm_V3 import (
    WINDOW_SIZE, STRIDE, BASELINE, extract_features, predict, move_gripper, arm,
)

# ── Settings ──────────────────────────────────────────────────────────────────

SERIAL_PORT = "COM6"   # check Device Manager -- must match the Olimex shield's port
BAUD_RATE = 57600
PACKET_SIZE = 17
CHANNEL_INDEX = 0

ENV_WINDOW = 30          # matches the acquisition script's envelope moving average
DEBOUNCE_COUNT = 3       # consecutive agreeing windows required before moving the gripper
PLOT_SAMPLES = 800       # ~3.2s of history at 250 Hz

# Below this envelope deviation from BASELINE, the window is treated as rest
# without even asking the model -- matches the acquisition scripts'
# THRESHOLD_INTENSITY. Without this gate the model has to call closing/opening
# vs. rest on noise alone, which is why it was flip-flopping constantly.
ACTIVITY_THRESHOLD = 33

# ── Live loop ─────────────────────────────────────────────────────────────────

def main():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    print(f"Connected to Olimex on {SERIAL_PORT}")

    raw_buffer = bytearray()
    # Long enough to compute an ENV_WINDOW-sample envelope for every one of the
    # last WINDOW_SIZE raw samples.
    raw_history = deque(maxlen=WINDOW_SIZE + ENV_WINDOW)
    env_history = deque(maxlen=WINDOW_SIZE)

    plot_raw = np.full(PLOT_SAMPLES, BASELINE, dtype=float)
    plot_env = np.full(PLOT_SAMPLES, BASELINE, dtype=float)

    state = {"new_samples": 0, "last_gesture": None, "pending_gesture": None, "pending_count": 0}

    fig, ax = plt.subplots(figsize=(12, 6))
    line_raw, = ax.plot(plot_raw, color="silver", alpha=0.6, label="Raw EMG")
    line_env, = ax.plot(plot_env, color="red", linewidth=2, label="Envelope")
    status_text = ax.text(0.02, 0.95, "STATUS: rest", transform=ax.transAxes,
                          color="gray", fontweight="bold")
    ax.set_ylim(0, 1024)
    ax.set_title("Live EMG -- close the window or Ctrl+C in the terminal to stop")
    ax.legend(loc="upper right")

    def update(frame):
        nonlocal plot_raw, plot_env

        if ser.in_waiting > 0:
            raw_buffer.extend(ser.read(ser.in_waiting))

        while len(raw_buffer) >= PACKET_SIZE:
            if raw_buffer[0] != 0xA5 or raw_buffer[1] != 0x5A:
                raw_buffer.pop(0)
                continue

            offset = 4 + CHANNEL_INDEX * 2
            val = float(struct.unpack(">H", raw_buffer[offset:offset + 2])[0])
            del raw_buffer[:PACKET_SIZE]

            raw_history.append(val)
            env_val = np.mean(np.abs(np.fromiter(raw_history, dtype=float)[-ENV_WINDOW:] - BASELINE)) + BASELINE
            env_history.append(env_val)

            plot_raw = np.roll(plot_raw, -1)
            plot_raw[-1] = val
            plot_env = np.roll(plot_env, -1)
            plot_env[-1] = env_val

            state["new_samples"] += 1
            if len(raw_history) < WINDOW_SIZE or state["new_samples"] < STRIDE:
                continue
            state["new_samples"] = 0

            window_raw = list(raw_history)[-WINDOW_SIZE:]
            window_env = list(env_history)[-WINDOW_SIZE:]
            activity = np.mean(window_env) - BASELINE

            if activity < ACTIVITY_THRESHOLD:
                gesture = "rest"
            else:
                feats = extract_features(window_raw, window_env)
                gesture = predict(feats)

            if gesture == state["pending_gesture"]:
                state["pending_count"] += 1
            else:
                state["pending_gesture"] = gesture
                state["pending_count"] = 1

            if state["pending_count"] >= DEBOUNCE_COUNT and gesture != state["last_gesture"]:
                move_gripper(gesture)
                state["last_gesture"] = gesture

            status_text.set_text(
                f"predicted: {gesture}  |  gripper: {state['last_gesture']}  |  "
                f"activity: {activity:.0f} (threshold {ACTIVITY_THRESHOLD})")

        line_raw.set_ydata(plot_raw)
        line_env.set_ydata(plot_env)
        return [line_raw, line_env, status_text]

    ani = animation.FuncAnimation(fig, update, interval=20, blit=False)
    print("Listening for live EMG... close the plot window or Ctrl+C to stop.\n")
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        arm.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
