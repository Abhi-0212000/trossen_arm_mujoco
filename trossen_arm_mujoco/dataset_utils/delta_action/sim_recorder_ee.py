import threading
import time
import os
import pickle
import numpy as np
from flask import Flask, jsonify, request

# Robot imports
import trossen_arm

# Environment imports
from trossen_arm_mujoco.utils import make_sim_env
from trossen_arm_mujoco.ee_transforms import action_14d_robot_to_world_aa
from trossen_arm_mujoco.delta_ee_sim_env_serl_drq_RL import SERLGymWrapper1, CubeStackingEE
from trossen_arm_mujoco.constants import sample_random_ready_pose

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

# --- Configuration ---
LEADER_LEFT_IP = '192.168.1.4'
LEADER_RIGHT_IP = '192.168.1.2'
DATA_DIR = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/Tasks/Right_moving_to_cube/randomize_ee_keep_same_cube_pose/pickup_20_trajs_ee_sim_data"

app = Flask(__name__)

# --- State Management ---
class Status:
    IDLE = "IDLE"
    RESETTING = "RESETTING..."
    RESETTING_FOR_START = "RESETTING (PRE-RECORD)..."
    RECORDING = "RECORDING (TELEOP ACTIVE)"
    SAVING = "SAVING..."

class GlobalState:
    status = Status.IDLE
    episode_data = []
    
    # Flags for the main loop
    cmd_reset = False
    cmd_home = False
    cmd_start = False
    cmd_capture_ready = False
    cmd_random_home = False
    cmd_sleep = False
    
    # Randomization
    random_seed_counter = 0
    
    # Feedback string
    last_msg = "Ready."

state = GlobalState()

# --- Flask Routes ---
@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sim Data Collector</title>
        <style>
            body { font-family: sans-serif; text-align: center; padding: 40px; background: #f0f0f0; }
            .box { background: white; padding: 30px; border-radius: 10px; display: inline-block; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
            button { padding: 15px 25px; font-size: 16px; margin: 10px; cursor: pointer; border-radius: 5px; border: none; font-weight: bold; transition: 0.2s; }
            button:disabled { background: #ccc !important; cursor: not-allowed; color: #666; }
            #btn-start { background: #28a745; color: white; }
            #btn-stop { background: #dc3545; color: white; }
            #btn-home { background: #17a2b8; color: white; }
            #btn-reset { background: #ffc107; color: black; }
            #btn-del { background: #6c757d; color: white; }
            .status-box { font-size: 20px; margin: 20px 0; font-weight: bold; color: #333; }
            .msg { color: #666; font-style: italic; }
        </style>
        <script>
            function send(endpoint) {
                // Optimistic UI update
                document.querySelectorAll('button').forEach(b => b.disabled = true);
                fetch('/' + endpoint, {method: 'POST'})
                    .then(r => r.json())
                    .then(updateUI);
            }

            function updateUI(data) {
                const s = data.status;
                document.getElementById('status-text').innerText = s;
                document.getElementById('msg-text').innerText = data.msg;
                document.getElementById('file-count').innerText = data.count;

                // Button Logic
                const isIdle = (s === 'IDLE');
                const isRecording = (s === 'RECORDING (TELEOP ACTIVE)');
                const isWorking = s.includes('RESETTING') || s.includes('SAVING');

                document.getElementById('btn-start').disabled = !isIdle;
                document.getElementById('btn-stop').disabled = !isRecording;
                document.getElementById('btn-home').disabled = !isIdle;
                document.getElementById('btn-random').disabled = !isIdle;
                document.getElementById('btn-sleep').disabled = !isIdle;
                document.getElementById('btn-reset').disabled = !isIdle;
                document.getElementById('btn-del').disabled = !isIdle;
                document.getElementById('btn-capture').disabled = !isIdle;

                // Color coding status
                const stBox = document.getElementById('status-text');
                if(isRecording) stBox.style.color = "red";
                else if(isWorking) stBox.style.color = "orange";
                else stBox.style.color = "green";
            }

            // Polling
            setInterval(() => {
                fetch('/status').then(r => r.json()).then(updateUI);
            }, 500);
        </script>
    </head>
    <body>
        <div class="box">
            <h1>Sim Data Collector (50Hz)</h1>
            <div class="status-box">Status: <span id="status-text">Checking...</span></div>
            <div class="msg" id="msg-text">...</div>
            <p>Episodes Saved: <span id="file-count">0</span></p>
            <hr>
            <div>
                <button id="btn-start" onclick="send('start')">START RECORDING</button>
                <button id="btn-stop" onclick="send('stop')" disabled>STOP RECORDING</button>
            </div>
            <div>
                <button id="btn-home" onclick="send('home')">HOME</button>
                <button id="btn-random" onclick="send('random_home')" style="background: #ff6b6b; color: white;">🎲 RANDOM READY</button>
                <button id="btn-sleep" onclick="send('sleep')" style="background: #9b59b6; color: white;">😴 SLEEP ROBOTS</button>
                <button id="btn-reset" onclick="send('reset')">RESET ENV</button>
                <button id="btn-del" onclick="send('delete')">DELETE LAST</button>
            </div>
            <div>
                <button id="btn-capture" onclick="send('capture_ready')" style="background: #6f42c1; color: white;">📍 CAPTURE READY POSE</button>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/status')
def get_status():
    count = len([f for f in os.listdir(DATA_DIR) if f.endswith('.pkl')]) if os.path.exists(DATA_DIR) else 0
    return jsonify({'status': state.status, 'msg': state.last_msg, 'count': count})

@app.route('/start', methods=['POST'])
def start_cmd():
    if state.status == Status.IDLE:
        state.status = Status.RESETTING_FOR_START
        state.cmd_start = True
        state.last_msg = "Resetting before recording..."
    return get_status()

@app.route('/stop', methods=['POST'])
def stop_cmd():
    if state.status == Status.RECORDING:
        state.status = Status.SAVING
        save_episode()
        state.status = Status.IDLE
        state.last_msg = "Recording stopped & saved."
    return get_status()

@app.route('/home', methods=['POST'])
def home_cmd():
    if state.status == Status.IDLE:
        state.status = Status.RESETTING
        state.cmd_home = True
        state.last_msg = "Homing robots & resetting env..."
    return get_status()

@app.route('/reset', methods=['POST'])
def reset_cmd():
    if state.status == Status.IDLE:
        state.status = Status.RESETTING
        state.cmd_reset = True
        state.last_msg = "Resetting env only..."
    return get_status()

@app.route('/delete', methods=['POST'])
def delete_cmd():
    if state.status == Status.IDLE:
        delete_last_episode()
    return get_status()

@app.route('/capture_ready', methods=['POST'])
def capture_ready_cmd():
    if state.status == Status.IDLE:
        state.cmd_capture_ready = True
        state.last_msg = "Capturing ready pose..."
    return get_status()

@app.route('/random_home', methods=['POST'])
def random_home_cmd():
    if state.status == Status.IDLE:
        state.status = Status.RESETTING
        state.cmd_random_home = True
        state.last_msg = "Moving to random ready position..."
    return get_status()

@app.route('/sleep', methods=['POST'])
def sleep_cmd():
    if state.status == Status.IDLE:
        state.cmd_sleep = True
        state.last_msg = "Moving robots to safe sleep position..."
    return get_status()

# --- Helper Functions ---
def save_episode():
    if not state.episode_data:
        state.last_msg = "No data to save."
        return

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    existing = [f for f in os.listdir(DATA_DIR) if f.startswith('episode_') and f.endswith('.pkl')]
    indices = [int(f.split('_')[1].split('.')[0]) for f in existing if 'episode_' in f]
    next_idx = max(indices) + 1 if indices else 0
    
    filename = os.path.join(DATA_DIR, f"episode_{next_idx}.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(state.episode_data, f)
    
    print(f"Saved {filename} ({len(state.episode_data)} steps)")
    state.episode_data = []

def delete_last_episode():
    if not os.path.exists(DATA_DIR): return
    existing = [f for f in os.listdir(DATA_DIR) if f.startswith('episode_') and f.endswith('.pkl')]
    if not existing: 
        state.last_msg = "No episodes found."
        return
    
    indices = [(int(f.split('_')[1].split('.')[0]), f) for f in existing if 'episode_' in f]
    if indices:
        max_idx, filename = max(indices, key=lambda x: x[0])
        os.remove(os.path.join(DATA_DIR, filename))
        state.last_msg = f"Deleted {filename}"
        print(f"Deleted {filename}")

# --- Robot Logic ---
def move_robots_to_home(driver_left, driver_right, randomize_right=False, seed=None):
    """
    Move robots to home position.
    
    Args:
        driver_left: Left robot driver
        driver_right: Right robot driver
        randomize_right: If True, move right robot to random ready position instead of home
        seed: Random seed for reproducible randomization
    """
    if randomize_right:
        print("Moving robots: LEFT to HOME, RIGHT to RANDOM READY position...")
    else:
        print("Moving robots to HOME...")
    
    driver_left.set_all_modes(trossen_arm.Mode.position)
    driver_right.set_all_modes(trossen_arm.Mode.position)
    
    # Left robot: Always go to joint home
    target_left = np.concatenate([np.zeros(6), [0.044]])
    driver_left.set_all_positions(target_left)
    
    if randomize_right:
        # Right robot: Move to random EE position
        ready_pose = sample_random_ready_pose('right', seed=seed)
        # ready_pose is [x, y, z, quat_w, quat_x, quat_y, quat_z]
        # For set_cartesian_positions, we need ArrayDouble6 [x, y, z, aa_x, aa_y, aa_z]
        # and InterpolationSpace parameter
        target_cartesian = trossen_arm.ArrayDouble6([
            ready_pose[0], ready_pose[1], ready_pose[2],  # position
            0.0, 0.0, 0.0  # orientation (angle-axis, identity)
        ])
        print(f"  Right random ready position: [{ready_pose[0]:.4f}, {ready_pose[1]:.4f}, {ready_pose[2]:.4f}]")
        driver_right.set_cartesian_positions(
            target_cartesian, 
            trossen_arm.InterpolationSpace.cartesian,
            goal_time=2.0,
            blocking=True
        )
    else:
        # Right robot: Go to joint home
        target_right = np.concatenate([np.zeros(6), [0.044]])
        driver_right.set_all_positions(target_right)
    
    time.sleep(2.5)  # Give robots time to reach position
    
    # Switch to freedrive mode
    zero_effort = np.zeros(7)
    driver_left.set_all_modes(trossen_arm.Mode.external_effort)
    driver_left.set_all_external_efforts(zero_effort, 0.0, False)
    driver_right.set_all_modes(trossen_arm.Mode.external_effort)
    driver_right.set_all_external_efforts(zero_effort, 0.0, False)
    print("Robots in Freedrive.")

def main_loop():
    print(f"Connecting Leaders: {LEADER_LEFT_IP}, {LEADER_RIGHT_IP}")
    driver_left = trossen_arm.TrossenArmDriver()
    driver_left.configure(trossen_arm.Model.wxai_v0, trossen_arm.StandardEndEffector.wxai_v0_leader, LEADER_LEFT_IP, False)
    driver_right = trossen_arm.TrossenArmDriver()
    driver_right.configure(trossen_arm.Model.wxai_v0, trossen_arm.StandardEndEffector.wxai_v0_leader, LEADER_RIGHT_IP, False)

    # Init Env
    CONTROL_TIMESTEP = 0.02
    PHYSICS_TIMESTEP = 0.002
    cam_list = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    
    print("Creating environment...")
    dm_env = make_sim_env(CubeStackingEE, task_name="sim_transfer_cube", onscreen_render=True, cam_list=cam_list, control_timestep=CONTROL_TIMESTEP, physics_timestep=PHYSICS_TIMESTEP)
    max_episode_length = 800
    env = SERLGymWrapper1(env=dm_env, state_obs_dim=16, max_episode_length=max_episode_length, onscreen_render=True, cam_list=cam_list, action_dim=14, time_limit=20.0, save_video=False, video_path=None, control_mode="teleop")

    # ====================WRAP ENV with SERL WRAPPERS===================
    # Wrap with SERLObsWrapper to unpack images
    env = SERLObsWrapper(env, proprio_keys=None)
    # 2. Add temporal dimension (obs_horizon=1 adds batch dim)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    # ==================================================================

    move_robots_to_home(driver_left, driver_right)
    obs, _ = env.reset(seed=42)

    step_counter = 0
    episode_step_counter = 0
    last_print_time = time.perf_counter()
    
    print("\nSystem Ready. Waiting for Web Command...")

    while True:
        loop_start_time = time.perf_counter()

        # --- PROCESS COMMANDS ---
        if state.cmd_capture_ready:
            # Capture current right robot EE pose for defining ready position
            r_pos = np.array(driver_right.get_cartesian_positions())
            r_grip = driver_right.get_gripper_position()
            print("\n" + "="*70)
            print("📍 RIGHT ROBOT READY POSE CAPTURED:")
            print("="*70)
            print(f"Position (x, y, z):    [{r_pos[0]:.6f}, {r_pos[1]:.6f}, {r_pos[2]:.6f}]")
            print(f"Orientation (aa_x, aa_y, aa_z): [{r_pos[3]:.6f}, {r_pos[4]:.6f}, {r_pos[5]:.6f}]")
            print(f"Gripper: {r_grip:.6f}")
            print("\nAdd to constants.py as:")
            print(f"RIGHT_READY_CENTER = np.array([{r_pos[0]:.6f}, {r_pos[1]:.6f}, {r_pos[2]:.6f}])")
            print(f"\nSample box around it (±0.05 in each axis):")
            print(f"RIGHT_READY_POS_MIN = np.array([{r_pos[0]-0.05:.6f}, {r_pos[1]-0.05:.6f}, {r_pos[2]-0.05:.6f}])")
            print(f"RIGHT_READY_POS_MAX = np.array([{r_pos[0]+0.05:.6f}, {r_pos[1]+0.05:.6f}, {r_pos[2]+0.05:.6f}])")
            print("="*70 + "\n")
            state.cmd_capture_ready = False
            state.last_msg = f"Ready pose captured! Check terminal."
        
        if state.cmd_sleep:
            # Move both robots to safe sleep position (joint home, stay in position mode)
            print("Moving robots to SLEEP (safe position)...")
            driver_left.set_all_modes(trossen_arm.Mode.position)
            driver_right.set_all_modes(trossen_arm.Mode.position)
            
            target = np.concatenate([np.zeros(6), [0.044]])
            driver_left.set_all_positions(target)
            driver_right.set_all_positions(target)
            
            time.sleep(2.0)
            print("Robots at safe sleep position. (Still in position mode)")
            state.cmd_sleep = False
            state.last_msg = "Robots sleeping at home. Safe to stop."
        
        if state.cmd_home:
            # Move robots to home and reset env
            move_robots_to_home(driver_left, driver_right, randomize_right=False)
            obs, _ = env.reset()
            episode_step_counter = 0
            state.cmd_home = False
            state.status = Status.IDLE
            state.last_msg = "Home & Reset Done. Ready."
        
        if state.cmd_random_home:
            # Move left to home, right to random ready position, then reset env
            move_robots_to_home(driver_left, driver_right, randomize_right=True, seed=state.random_seed_counter)
            state.random_seed_counter += 1
            obs, _ = env.reset()
            episode_step_counter = 0
            state.cmd_random_home = False
            state.status = Status.IDLE
            state.last_msg = f"Random ready (seed={state.random_seed_counter-1}). Ready to record!"
        
        if state.cmd_reset:
            # Just reset env, don't move robots
            obs, _ = env.reset()
            episode_step_counter = 0
            state.cmd_reset = False
            state.status = Status.IDLE
            state.last_msg = "Env Reset Done. Ready."

        if state.cmd_start:
            # Start Recording Sequence: Reset -> Home -> Set Flag
            # move_robots_to_home(driver_left, driver_right)
            # obs, _ = env.reset()
            # episode_step_counter = 0
            state.episode_data = [] # Clear buffer
            state.cmd_start = False
            state.status = Status.RECORDING
            state.last_msg = "recording started! Go!"
            print("--- RECORDING STARTED ---")

        # --- CONTROL LOOP ---
        l_pos = np.array(driver_left.get_cartesian_positions())
        l_grip = driver_left.get_gripper_position()
        r_pos = np.array(driver_right.get_cartesian_positions())
        r_grip = driver_right.get_gripper_position()

        action = np.concatenate([l_pos[0:3], l_pos[3:6], [l_grip], r_pos[0:3], r_pos[3:6], [r_grip]])
        action_aa_in_sim_world_frame = action_14d_robot_to_world_aa(action)
        next_obs, reward, done, trunc, info = env.step(action_aa_in_sim_world_frame)

        if state.status == Status.RECORDING:
            transition = {
                'observations': obs, 'actions': action_aa_in_sim_world_frame, 'rewards': reward,
                'next_observations': next_obs, 'dones': done, 'infos': info
            }
            state.episode_data.append(transition)
            episode_step_counter += 1

            if done or trunc or episode_step_counter >= max_episode_length:
                print(f"Episode done! Auto-saving ({episode_step_counter} steps).")
                state.status = Status.SAVING
                save_episode()
                # obs, _ = env.reset()
                episode_step_counter = 0
                state.status = Status.IDLE
                state.last_msg = "Episode done. Auto-saved!"
                # obs, _ = env.reset()

            else:
                obs = next_obs
        else:
            obs = next_obs

        # --- TIMING ---
        elapsed = time.perf_counter() - loop_start_time
        if CONTROL_TIMESTEP > elapsed:
            time.sleep(CONTROL_TIMESTEP - elapsed)

        step_counter += 1
        if step_counter % 200 == 0:
            actual_hz = 1.0 / (time.perf_counter() - last_print_time) * 200
            last_print_time = time.perf_counter()
            print(f"Hz: {actual_hz:.1f}")

if __name__ == '__main__':
    t = threading.Thread(target=app.run, kwargs={'host':'0.0.0.0', 'port':5000, 'debug':False})
    t.daemon = True
    t.start()
    
    print(f"Sim Recorder Running at http://localhost:5000")
    main_loop()