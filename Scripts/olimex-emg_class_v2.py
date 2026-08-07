#%% show fast EMG signal+envelop
# import serial
# import struct
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from datetime import datetime
# import csv
# import os
# import time
# import tkinter as tk
# from tkinter import simpledialog
# #%%
# # --- Constants ---
# SERIAL_PORT = 'COM3'
# BAUD_RATE = 57600
# PACKET_SIZE = 17
# BUFFER_SIZE = 500  
# CHANNEL_INDEX = 0
# GUI_UPDATE_MS = 20 

# # Envelope Settings
# WINDOW_SIZE = 30  # Adjust this (20-50) to make the envelope smoother or more responsive

# SAVE_DIR = r"C:\Users\ajitj\OneDrive - Universitetet i Agder\Teaching\MasterProgram\Applied ML and Robotics\MAS512-G_AJ_2021\MAS-509-G\2025\Experiment-Lab\Olimex\emg_data"
# os.makedirs(SAVE_DIR, exist_ok=True)

# def parse_packet(packet):
#     sync0, sync1, version, count = struct.unpack('BBBB', packet[0:4])
#     if sync0 != 0xA5 or sync1 != 0x5A or version != 2: return None
#     data = struct.unpack('>6H', packet[4:16])
#     return {'data': data}

# def save_to_file(marker_label, raw_data, envelope_data):
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     filename = f"{marker_label}_{timestamp}.csv"
#     full_path = os.path.join(SAVE_DIR, filename)
#     with open(full_path, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["Marker", marker_label])
#         writer.writerow(["Sample", "Raw_Value", "Envelope_Value"])
#         for i in range(len(raw_data)):
#             writer.writerow([i, raw_data[i], envelope_data[i]])
#     print(f"[✔] Saved to: {full_path}")

# def main():
#     ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0) 
#     raw_serial_buffer = bytearray()
    
#     # Buffers for plotting
#     plot_raw = np.zeros(BUFFER_SIZE)
#     plot_env = np.zeros(BUFFER_SIZE)

#     state = {
#         'logging': False,
#         'pending_marker_time': None,
#         'logging_started_at': None,
#         'log_raw': [],
#         'log_env': [],
#         'marker_label': None,
#     }

#     fig, ax = plt.subplots(figsize=(12, 6))
#     line_raw, = ax.plot(plot_raw, color='lightgray', alpha=0.7, label='Raw EMG')
#     line_env, = ax.plot(plot_env, color='red', linewidth=2, label='Envelope (Intensity)')
    
#     ax.set_ylim(0, 1024)
#     ax.set_title("EMG Acquisition: Raw vs. Envelope (Grasp Detection)")
#     ax.legend(loc='upper right')

#     root = tk.Tk(); root.withdraw()

#     def on_key(event):
#         if event.key == 's':
#             marker = simpledialog.askstring("Marker Input", "Enter action (e.g. 'grasp'):", parent=root)
#             if marker:
#                 state['marker_label'] = marker
#                 state['pending_marker_time'] = time.time()
#                 print(f"Waiting 15s... Prepare for: {marker}")

#     def update(frame):
#         nonlocal plot_raw, plot_env, raw_serial_buffer
        
#         if ser.in_waiting > 0:
#             raw_serial_buffer += ser.read(ser.in_waiting)

#         while len(raw_serial_buffer) >= PACKET_SIZE:
#             if raw_serial_buffer[0] == 0xA5 and raw_serial_buffer[1] == 0x5A:
#                 packet = raw_serial_buffer[:PACKET_SIZE]
#                 parsed = parse_packet(packet)
                
#                 if parsed:
#                     val = parsed['data'][CHANNEL_INDEX]
                    
#                     # 1. Calculate Envelope (Mean Absolute Value)
#                     # We subtract 512 to center the signal at 0 for rectification
#                     rectified = abs(val - 512) 
                    
#                     # Update plot buffers
#                     plot_raw[:-1] = plot_raw[1:]
#                     plot_raw[-1] = val
                    
#                     # Compute Moving Average for the Envelope
#                     current_env = np.mean(np.abs(plot_raw[-WINDOW_SIZE:] - 512)) + 512
#                     plot_env[:-1] = plot_env[1:]
#                     plot_env[-1] = current_env
                    
#                     # 2. Handle Logging Logic
#                     curr_t = time.time()
#                     if state['pending_marker_time'] and (curr_t - state['pending_marker_time'] >= 15):
#                         state['logging'] = True
#                         state['logging_started_at'] = curr_t
#                         state['log_raw'], state['log_env'] = [], []
#                         state['pending_marker_time'] = None
#                         print(f"--- LOGGING STARTED: {state['marker_label']} ---")

#                     if state['logging']:
#                         state['log_raw'].append(val)
#                         state['log_env'].append(current_env)
#                         if curr_t - state['logging_started_at'] >= 15:
#                             state['logging'] = False
#                             save_to_file(state['marker_label'], state['log_raw'], state['log_env'])

#                 raw_serial_buffer = raw_serial_buffer[PACKET_SIZE:]
#             else:
#                 raw_serial_buffer.pop(0)

#         line_raw.set_ydata(plot_raw)
#         line_env.set_ydata(plot_env)
#         return [line_raw, line_env]

#     fig.canvas.mpl_connect('key_press_event', on_key)
#     ani = animation.FuncAnimation(fig, update, interval=GUI_UPDATE_MS, blit=True)
#     plt.show()
#     ser.close()

# if __name__ == "__main__":
#     main()





#%%

#To ensure a single, continuous acquisition for a complete action (like a full wrist bend and release), you need to implement a "Hysteresis" or "Hold-Over" logic.
#As an engineer, you know muscle signals have "gaps" where the intensity temporarily dips during a movement. 
# By increasing the POST_MOVE_DELAY, the script will "wait out" those small dips and only stop recording once the muscle has been completely relaxed for a significant amount of time.


# import serial
# import struct
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from datetime import datetime
# import csv
# import os
# import time
# import tkinter as tk
# from tkinter import simpledialog

# # --- Engineering Constants ---
# SERIAL_PORT = 'COM3'
# BAUD_RATE = 57600
# PACKET_SIZE = 17
# BUFFER_SIZE = 800         # Shows ~3 seconds of history
# CHANNEL_INDEX = 0
# BASELINE = 512            # Mid-point of 10-bit ADC

# # --- Trigger & Classification Logic ---
# THRESHOLD_INTENSITY = 45  # ADC units above baseline to trigger
# WINDOW_SIZE = 30          # Envelope smoothing window
# POST_MOVE_DELAY = 2.0     # Bridge gaps: Wait 2s of silence before saving
# MIN_ACTION_SAMPLES = 250  # Ignore "blips" shorter than ~1 second (at 250Hz)

# # Setup Directory
# SAVE_DIR = r"C:\Users\ajitj\OneDrive - Universitetet i Agder\Teaching\MasterProgram\Applied ML and Robotics\MAS512-G_AJ_2021\MAS-509-G\2025\Experiment-Lab\Olimex\emg_data"
# os.makedirs(SAVE_DIR, exist_ok=True)

# def save_to_file(label, raw_data, env_data, thresh):
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     filename = f"action_{label}_{timestamp}.csv"
#     full_path = os.path.join(SAVE_DIR, filename)
#     with open(full_path, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["Sample", "Raw_EMG", "Envelope", "Threshold_Used"])
#         for i in range(len(raw_data)):
#             writer.writerow([i, raw_data[i], env_data[i], thresh])
#     print(f"\n[✔] DATA SAVED: {filename} ({len(raw_data)} samples)")

# def main():
#     try:
#         ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0)
#     except Exception as e:
#         print(f"Error opening port: {e}")
#         return

#     raw_serial_buffer = bytearray()
#     plot_raw = np.full(BUFFER_SIZE, BASELINE)
#     plot_env = np.full(BUFFER_SIZE, BASELINE)

#     state = {
#         'logging': False,
#         'log_raw': [],
#         'log_env': [],
#         'last_active_t': 0,
#         'current_label': "wrist_bend" # Default label
#     }

#     # Setup Plot
#     fig, ax = plt.subplots(figsize=(12, 6))
#     line_raw, = ax.plot(plot_raw, color='silver', alpha=0.5, label='Raw EMG')
#     line_env, = ax.plot(plot_env, color='red', linewidth=2, label='Envelope')
#     thresh_line = ax.axhline(BASELINE + THRESHOLD_INTENSITY, color='green', linestyle='--', label='Trigger Threshold')
    
#     status_text = ax.text(0.02, 0.95, 'STATUS: IDLE', transform=ax.transAxes, color='gray', fontweight='bold')
    
#     ax.set_ylim(0, 1024)
#     ax.set_title(f"EMG Classification Acquisition (Threshold: {THRESHOLD_INTENSITY})")
#     ax.set_ylabel("ADC Value")
#     ax.legend(loc='upper right')

#     root = tk.Tk(); root.withdraw()

#     def update(frame):
#         nonlocal plot_raw, plot_env, raw_serial_buffer
        
#         if ser.in_waiting > 0:
#             raw_serial_buffer += ser.read(ser.in_waiting)

#         while len(raw_serial_buffer) >= PACKET_SIZE:
#             if raw_serial_buffer[0] == 0xA5 and raw_serial_buffer[1] == 0x5A:
#                 # Extract Channel 1 (A0)
#                 data_bytes = raw_serial_buffer[4 + CHANNEL_INDEX*2 : 6 + CHANNEL_INDEX*2]
#                 val = struct.unpack('>H', data_bytes)[0]
                
#                 # Envelope Calculation
#                 intensity = abs(val - BASELINE)
#                 plot_raw[:-1] = plot_raw[1:]; plot_raw[-1] = val
                
#                 current_env_int = np.mean(np.abs(plot_raw[-WINDOW_SIZE:] - BASELINE))
#                 current_env_val = current_env_int + BASELINE
#                 plot_env[:-1] = plot_env[1:]; plot_env[-1] = current_env_val
                
#                 # --- AUTO-TRIGGER WITH HYSTERESIS ---
#                 curr_t = time.time()
#                 is_above_thresh = current_env_int > THRESHOLD_INTENSITY

#                 if not state['logging'] and is_above_thresh:
#                     state['logging'] = True
#                     state['log_raw'], state['log_env'] = [], []
#                     state['last_active_t'] = curr_t
#                     status_text.set_text("STATUS: RECORDING...")
#                     status_text.set_color('red')

#                 if state['logging']:
#                     state['log_raw'].append(val)
#                     state['log_env'].append(current_env_val)
                    
#                     if is_above_thresh:
#                         state['last_active_t'] = curr_t # Reset the "silence" timer
                    
#                     # Stop only after consistent silence
#                     if curr_t - state['last_active_t'] > POST_MOVE_DELAY:
#                         state['logging'] = False
#                         status_text.set_text("STATUS: IDLE")
#                         status_text.set_color('gray')
                        
#                         if len(state['log_raw']) > MIN_ACTION_SAMPLES:
#                             save_to_file(state['current_label'], state['log_raw'], state['log_env'], BASELINE + THRESHOLD_INTENSITY)
#                         else:
#                             print("Discarded: Noise/Blip detected.")

#                 raw_serial_buffer = raw_serial_buffer[PACKET_SIZE:]
#             else:
#                 raw_serial_buffer.pop(0)

#         line_raw.set_ydata(plot_raw)
#         line_env.set_ydata(plot_env)
#         return [line_raw, line_env, status_text]

#     def on_key(event):
#         if event.key == 'l': # Press 'l' to change the label for the next action
#             new_label = simpledialog.askstring("Label", "Set label for next action (e.g., finger_snap):")
#             if new_label: state['current_label'] = new_label

#     fig.canvas.mpl_connect('key_press_event', on_key)
#     ani = animation.FuncAnimation(fig, update, interval=20, blit=True)
#     print(f"System Armed. Target Label: {state['current_label']}")
#     print("Press 'l' to change the action label.")
#     plt.show()
#     ser.close()

# if __name__ == "__main__":
#     main()

#%%
# press l - to make label, q to quit, the acquisition starts when threshol is met
# import serial
# import struct
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from datetime import datetime
# import csv
# import os
# import time
# import tkinter as tk
# from tkinter import simpledialog

# # --- Engineering Constants ---
# SERIAL_PORT = 'COM3'
# BAUD_RATE = 57600
# PACKET_SIZE = 17
# BUFFER_SIZE = 800         
# CHANNEL_INDEX = 0
# BASELINE = 512            

# # --- Source of Truth Sampling ---
# TARGET_FS = 250           # Fixed sampling rate from Arduino Timer
# SAMPLE_INTERVAL = 1.0 / TARGET_FS

# # --- Trigger & Classification Logic ---
# THRESHOLD_INTENSITY = 45   
# WINDOW_SIZE = 30           
# POST_MOVE_DELAY = 2.0      # Seconds of silence before saving
# MIN_ACTION_SAMPLES = 250   # Ignore events shorter than 1s

# # Setup Directory
# SAVE_DIR = r"C:\Users\ajitj\OneDrive - Universitetet i Agder\Teaching\MasterProgram\Applied ML and Robotics\MAS512-G_AJ_2021\MAS-509-G\2025\Experiment-Lab\Olimex\emg_data"
# os.makedirs(SAVE_DIR, exist_ok=True)

# def save_to_file(label, raw_data, env_data, thresh):
#     # Duration is calculated from sample count to ensure mathematical consistency
#     calc_duration = len(raw_data) * SAMPLE_INTERVAL
    
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     filename = f"action_{label}_{timestamp}.csv"
#     full_path = os.path.join(SAVE_DIR, filename)
    
#     with open(full_path, 'w', newline='') as f:
#         writer = csv.writer(f)
#         # --- Metadata (Standardized for ML) ---
#         writer.writerow(["Action_Label", label])
#         writer.writerow(["Sampling_Freq_Hz", TARGET_FS])
#         writer.writerow(["Duration_Sec", round(calc_duration, 4)])
#         writer.writerow(["Total_Samples", len(raw_data)])
#         writer.writerow([]) 
#         # --- Data ---
#         writer.writerow(["Sample_Index", "Raw_EMG", "Envelope", "Threshold"])
#         for i in range(len(raw_data)):
#             writer.writerow([i, raw_data[i], env_data[i], thresh])
            
#     print(f"\n[✔] DATA SAVED: {filename}")
#     print(f"    Samples: {len(raw_data)} | Duration: {round(calc_duration, 2)}s")

# def main():
#     try:
#         ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0)
#     except Exception as e:
#         print(f"Connection Error: {e}")
#         return

#     raw_serial_buffer = bytearray()
#     plot_raw = np.full(BUFFER_SIZE, BASELINE)
#     plot_env = np.full(BUFFER_SIZE, BASELINE)

#     state = {
#         'logging': False,
#         'log_raw': [],
#         'log_env': [],
#         'last_active_t': 0,
#         'current_label': "wrist_bend"
#     }

#     # Plot Setup
#     fig, ax = plt.subplots(figsize=(12, 6))
#     line_raw, = ax.plot(plot_raw, color='silver', alpha=0.5, label='Raw EMG')
#     line_env, = ax.plot(plot_env, color='red', linewidth=2, label='Envelope')
#     ax.axhline(BASELINE + THRESHOLD_INTENSITY, color='green', linestyle='--', label='Trigger')
    
#     status_text = ax.text(0.02, 0.95, 'STATUS: IDLE', transform=ax.transAxes, color='gray', fontweight='bold')
    
#     ax.set_ylim(0, 1024)
#     ax.set_title(f"EMG Acquisition | Target: {state['current_label']}")
#     ax.legend(loc='upper right')

#     root = tk.Tk(); root.withdraw()

#     def update(frame):
#         nonlocal plot_raw, plot_env, raw_serial_buffer
        
#         if ser.in_waiting > 0:
#             raw_serial_buffer += ser.read(ser.in_waiting)

#         while len(raw_serial_buffer) >= PACKET_SIZE:
#             if raw_serial_buffer[0] == 0xA5 and raw_serial_buffer[1] == 0x5A:
#                 # Extract Ch1
#                 val = struct.unpack('>H', raw_serial_buffer[4 + CHANNEL_INDEX*2 : 6 + CHANNEL_INDEX*2])[0]
                
#                 # Visual Processing
#                 plot_raw[:-1] = plot_raw[1:]; plot_raw[-1] = val
#                 intensity = abs(val - BASELINE)
#                 current_env_val = np.mean(np.abs(plot_raw[-WINDOW_SIZE:] - BASELINE)) + BASELINE
#                 plot_env[:-1] = plot_env[1:]; plot_env[-1] = current_env_val
                
#                 curr_t = time.time()
#                 is_above = (current_env_val - BASELINE) > THRESHOLD_INTENSITY

#                 # Trigger Logic
#                 if not state['logging'] and is_above:
#                     state['logging'] = True
#                     state['log_raw'], state['log_env'] = [], []
#                     state['last_active_t'] = curr_t
#                     status_text.set_text(f"RECORDING: {state['current_label']}")
#                     status_text.set_color('red')

#                 if state['logging']:
#                     state['log_raw'].append(val)
#                     state['log_env'].append(current_env_val)
#                     if is_above: state['last_active_t'] = curr_t
                    
#                     if curr_t - state['last_active_t'] > POST_MOVE_DELAY:
#                         state['logging'] = False
#                         status_text.set_text("STATUS: IDLE")
#                         status_text.set_color('gray')
                        
#                         if len(state['log_raw']) > MIN_ACTION_SAMPLES:
#                             save_to_file(state['current_label'], state['log_raw'], state['log_env'], BASELINE + THRESHOLD_INTENSITY)

#                 raw_serial_buffer = raw_serial_buffer[PACKET_SIZE:]
#             else:
#                 raw_serial_buffer.pop(0)

#         line_raw.set_ydata(plot_raw)
#         line_env.set_ydata(plot_env)
#         return [line_raw, line_env, status_text]

#     def on_key(event):
#         if event.key == 'l':
#             new = simpledialog.askstring("Label", "Label for next action:")
#             if new: 
#                 state['current_label'] = new
#                 ax.set_title(f"EMG Acquisition | Target: {state['current_label']}")

#     fig.canvas.mpl_connect('key_press_event', on_key)
#     ani = animation.FuncAnimation(fig, update, interval=20, blit=True)
#     plt.show()
#     ser.close()

# if __name__ == "__main__":
#     main()

#%%

# Terminal: Press 'l' to set the label.
# Terminal: Press 'g' (Go/Start) to arm the system.
# Wait: A 5-second countdown appears in the terminal (prepare your muscle).
# Acquire: The system enters "Threshold Mode." It waits for your motion, records the full action, saves the CSV, and returns to Idle.
# make the acq window on screen, press l,g, q in terminal

# import serial
# import struct
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from datetime import datetime
# import csv
# import os
# import time
# import tkinter as tk
# from tkinter import simpledialog

# # --- Engineering Constants ---
# SERIAL_PORT = 'COM3'
# BAUD_RATE = 57600
# PACKET_SIZE = 17
# BUFFER_SIZE = 800         
# CHANNEL_INDEX = 0
# BASELINE = 512            
# TARGET_FS = 250           
# SAMPLE_INTERVAL = 1.0 / TARGET_FS

# # --- Trigger Logic ---
# THRESHOLD_INTENSITY = 45    #45, roll-25
# WINDOW_SIZE = 70           # 30 roll -70
# POST_MOVE_DELAY = 2.0     # 2.0 roll -3.5
# MIN_ACTION_SAMPLES = 250   #  250 roll - 500 

# SAVE_DIR = r"C:\Users\ajitj\OneDrive - Universitetet i Agder\Teaching\MasterProgram\Applied ML and Robotics\MAS512-G_AJ_2021\MAS-509-G\2025\Experiment-Lab\Olimex\emg_data"
# os.makedirs(SAVE_DIR, exist_ok=True)

# def save_to_file(label, raw_data, env_data, thresh):
#     calc_duration = len(raw_data) * SAMPLE_INTERVAL
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     filename = f"action_{label}_{timestamp}.csv"
#     full_path = os.path.join(SAVE_DIR, filename)
#     with open(full_path, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["Action_Label", label, "Fs", TARGET_FS, "Duration", round(calc_duration, 4)])
#         writer.writerow([]) 
#         writer.writerow(["Sample_Index", "Raw_EMG", "Envelope", "Threshold"])
#         for i in range(len(raw_data)):
#             writer.writerow([i, raw_data[i], env_data[i], thresh])
#     print(f"\n[✔] SUCCESS: {filename} saved.")

# def main():
#     try:
#         ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0)
#     except Exception as e:
#         print(f"Error: {e}"); return

#     raw_serial_buffer = bytearray()
#     plot_raw = np.full(BUFFER_SIZE, BASELINE)
#     plot_env = np.full(BUFFER_SIZE, BASELINE)

#     state = {
#         'system_status': 'IDLE', # IDLE, ARMED, COUNTDOWN, MONITORING, LOGGING
#         'current_label': "None",
#         'logging': False,
#         'log_raw': [],
#         'log_env': [],
#         'last_active_t': 0,
#         'countdown_start': 0
#     }

#     fig, ax = plt.subplots(figsize=(12, 6))
#     line_raw, = ax.plot(plot_raw, color='silver', alpha=0.5, label='Raw EMG')
#     line_env, = ax.plot(plot_env, color='red', linewidth=2, label='Envelope')
#     ax.axhline(BASELINE + THRESHOLD_INTENSITY, color='green', linestyle='--', label='Trigger')
    
#     status_text = ax.text(0.02, 0.95, 'STATUS: IDLE', transform=ax.transAxes, color='gray', fontweight='bold')
#     ax.set_ylim(0, 1024)
#     ax.legend(loc='upper right')

#     root = tk.Tk(); root.withdraw()

#     def update(frame):
#         nonlocal plot_raw, plot_env, raw_serial_buffer
#         curr_t = time.time()

#         # Always read serial to keep buffer clear
#         if ser.in_waiting > 0:
#             raw_serial_buffer += ser.read(ser.in_waiting)

#         while len(raw_serial_buffer) >= PACKET_SIZE:
#             if raw_serial_buffer[0] == 0xA5 and raw_serial_buffer[1] == 0x5A:
#                 val = struct.unpack('>H', raw_serial_buffer[4:6])[0]
#                 plot_raw[:-1] = plot_raw[1:]; plot_raw[-1] = val
#                 env_val = np.mean(np.abs(plot_raw[-WINDOW_SIZE:] - BASELINE)) + BASELINE
#                 plot_env[:-1] = plot_env[1:]; plot_env[-1] = env_val
                
#                 # --- STATE MACHINE ---
#                 if state['system_status'] == 'COUNTDOWN':
#                     remaining = 5 - (curr_t - state['countdown_start'])
#                     if remaining > 0:
#                         status_text.set_text(f"PREPARE: {remaining:.1f}s")
#                     else:
#                         state['system_status'] = 'MONITORING'
#                         print("\n>>> SYSTEM LIVE: Perform your action now!")

#                 elif state['system_status'] == 'MONITORING' or state['system_status'] == 'LOGGING':
#                     is_active = (env_val - BASELINE) > THRESHOLD_INTENSITY
                    
#                     if not state['logging'] and is_active:
#                         state['logging'] = True
#                         state['system_status'] = 'LOGGING'
#                         state['log_raw'], state['log_env'] = [], []
#                         state['last_active_t'] = curr_t
#                         status_text.set_color('red')
                    
#                     if state['logging']:
#                         state['log_raw'].append(val)
#                         state['log_env'].append(env_val)
#                         status_text.set_text(f"RECORDING: {state['current_label']}")
#                         if is_active: state['last_active_t'] = curr_t
                        
#                         if curr_t - state['last_active_t'] > POST_MOVE_DELAY:
#                             state['logging'] = False
#                             state['system_status'] = 'IDLE'
#                             status_text.set_text("STATUS: IDLE")
#                             status_text.set_color('gray')
#                             if len(state['log_raw']) > MIN_ACTION_SAMPLES:
#                                 save_to_file(state['current_label'], state['log_raw'], state['log_env'], BASELINE + THRESHOLD_INTENSITY)
#                             print("\nAcquisition finished. System returned to IDLE.")
#                             print("Instructions: [l] Set Label | [g] Start Acquisition | [q] Quit")

#                 raw_serial_buffer = raw_serial_buffer[PACKET_SIZE:]
#             else:
#                 raw_serial_buffer.pop(0)

#         line_raw.set_ydata(plot_raw)
#         line_env.set_ydata(plot_env)
#         return [line_raw, line_env, status_text]

#     def on_key(event):
#         if event.key == 'l':
#             label = simpledialog.askstring("Label", "Set label for next action:")
#             if label: 
#                 state['current_label'] = label
#                 print(f"Label set to: {label}. Press 'g' to start the 5s timer.")
#         elif event.key == 'g':
#             if state['current_label'] == "None":
#                 print("Error: Set a label with 'l' first!")
#             else:
#                 print(f"Starting in 5 seconds... Get ready to perform: {state['current_label']}")
#                 state['countdown_start'] = time.time()
#                 state['system_status'] = 'COUNTDOWN'
#         elif event.key == 'q':
#             plt.close()

#     fig.canvas.mpl_connect('key_press_event', on_key)
#     ani = animation.FuncAnimation(fig, update, interval=20, blit=True)
    
#     print("\n--- EMG ACQUISITION SYSTEM ---")
#     print("Instructions:")
#     print(" [l] - Set action Label (MUST DO FIRST)")
#     print(" [g] - START (Starts 5s preparation timer)")
#     print(" [q] - QUIT")
#     print("------------------------------")
    
#     plt.show()
#     ser.close()

# if __name__ == "__main__":
#     main()

#%%

#To ensure every recording has the exact same number of samples for  Machine Learning pipeline,  switch from a threshold-trigger to a fixed-duration timer.
#In this modified logic, once the 5-second countdown ends, the system will record for exactly 15 seconds regardless of the signal intensity. 
# This guarantees that every CSV file has exactly 3,750 samples (at 250 Hz).
# press l to set the label, g to start, the acquisition starts in 5 sec, so as soon as you press g start the action, it acquires for 15 s, q to quit
#acq window on screen, l,g,q in terminal


import serial
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import csv
import os
import time
import tkinter as tk
from tkinter import simpledialog

# --- Constants ---
SERIAL_PORT = 'COM6' #check COM
BAUD_RATE = 57600
PACKET_SIZE = 17
BUFFER_SIZE = 800         
CHANNEL_INDEX = 0
BASELINE = 512            
TARGET_FS = 250           
RECORDING_DURATION = 5 #change to acquire time duration of EMG signal   
REQUIRED_SAMPLES = TARGET_FS * RECORDING_DURATION 

SAVE_DIR = r"C:\Users\celin\Documents\Mekatronikk\Prosjekt_UiA\Robot_Arm\X-arm-project\data\raw\open_fist_sunniva"
os.makedirs(SAVE_DIR, exist_ok=True)

def save_to_file(label, raw_data, env_data):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{label}_{timestamp}.csv"
    full_path = os.path.join(SAVE_DIR, filename)
    with open(full_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Action_Label", label])
        writer.writerow(["Fs_Hz", TARGET_FS])
        writer.writerow(["Total_Samples", len(raw_data)])
        writer.writerow([]) 
        writer.writerow(["Sample_Index", "Raw_EMG", "Envelope"])
        for i in range(len(raw_data)):
            writer.writerow([i, raw_data[i], env_data[i]])
    print(f"\n[✔] SUCCESS: {filename} saved.")

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    except Exception as e:
        print(f"Error: {e}"); return

    raw_serial_buffer = bytearray()
    plot_raw = np.full(BUFFER_SIZE, BASELINE, dtype=float)
    plot_env = np.full(BUFFER_SIZE, BASELINE, dtype=float)

    state = {
        'system_status': 'IDLE',
        'current_label': "None",
        'log_raw': [],
        'log_env': [],
        'countdown_start': 0
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    line_raw, = ax.plot(plot_raw, color='silver', alpha=0.6, label='Raw EMG')
    line_env, = ax.plot(plot_env, color='red', linewidth=2, label='Envelope')
    
    status_text = ax.text(0.02, 0.95, 'STATUS: IDLE', transform=ax.transAxes, color='gray', fontweight='bold')
    ax.set_ylim(0, 1024)
    ax.legend(loc='upper right')

    root = tk.Tk(); root.withdraw()

    def update(frame):
        nonlocal plot_raw, plot_env, raw_serial_buffer
        curr_t = time.time()

        if ser.in_waiting > 0:
            raw_serial_buffer += ser.read(ser.in_waiting)

        while len(raw_serial_buffer) >= PACKET_SIZE:
            if raw_serial_buffer[0] == 0xA5 and raw_serial_buffer[1] == 0x5A:
                # FIX: Unpack into an integer, not a tuple
                val_tuple = struct.unpack('>H', raw_serial_buffer[4:6])
                val = float(val_tuple[0])
                
                # Update plot arrays
                plot_raw = np.roll(plot_raw, -1)
                plot_raw[-1] = val
                
                env_val = np.mean(np.abs(plot_raw[-30:] - BASELINE)) + BASELINE
                plot_env = np.roll(plot_env, -1)
                plot_env[-1] = env_val
                
                # --- STATE MACHINE ---
                if state['system_status'] == 'COUNTDOWN':
                    remaining = 5 - (curr_t - state['countdown_start'])
                    if remaining > 0:
                        status_text.set_text(f"PREPARE: {remaining:.1f}s")
                    else:
                        state['system_status'] = 'RECORDING'
                        state['log_raw'], state['log_env'] = [], []
                        print(f"\n>>> STARTING 15s LOG: {state['current_label']}")

                elif state['system_status'] == 'RECORDING':
                    state['log_raw'].append(val)
                    state['log_env'].append(env_val)
                    
                    progress = (len(state['log_raw']) / REQUIRED_SAMPLES) * 100
                    status_text.set_text(f"RECORDING: {progress:.1f}%")
                    status_text.set_color('red')
                    
                    if len(state['log_raw']) >= REQUIRED_SAMPLES:
                        state['system_status'] = 'IDLE'
                        status_text.set_text("STATUS: IDLE")
                        status_text.set_color('gray')
                        save_to_file(state['current_label'], state['log_raw'], state['log_env'])
                        print("\nDone. [l] New Label | [g] Start Again | [q] Quit")

                raw_serial_buffer = raw_serial_buffer[PACKET_SIZE:]
            else:
                raw_serial_buffer.pop(0)

        line_raw.set_ydata(plot_raw)
        line_env.set_ydata(plot_env)
        return [line_raw, line_env, status_text]

    def on_key(event):
        # Force lower case check to ensure 'G' works too
        key = event.key.lower() if event.key else ""
        if key == 'l':
            label = simpledialog.askstring("Label", "Set label:")
            if label: 
                state['current_label'] = label
                print(f"Label set: {label}. Press 'g' to start.")
        elif key == 'g':
            if state['current_label'] == "None":
                print("Error: Set label first (press 'l')")
            elif state['system_status'] == 'IDLE':
                state['countdown_start'] = time.time()
                state['system_status'] = 'COUNTDOWN'
                print("Timer started...")
        elif key == 'q':
            plt.close()

    fig.canvas.mpl_connect('key_press_event', on_key)
    # blit=False is safer for troubleshooting text/dynamic updates
    ani = animation.FuncAnimation(fig, update, interval=20, blit=False)
    
    print("\n--- FIXED-LENGTH ACQUISITION ---")
    print("Instructions: Click plot window, then:")
    print(" [l] Set Label")
    print(" [g] Start (5s delay + 15s log)")
    print(" [q] Quit")
    
    plt.show()
    ser.close()

if __name__ == "__main__":
    main()