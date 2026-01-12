1. Record the dataset using `./sim_recorder_ee.py`
   Creates `pkl` format datasets in folder.

2. Correct the recorder dataset and ensure it is in `serl pkl` format using `./cleanup_utils.py`'s `fix_pickle_files()` func

3. Initial clean up. GOAL: delete the NOT GOOD (incomplete or mistakes, ...) episodes. Just play the images and see
```bash
python /home/qte9489/personal_abhi/temp/hil-serl/trossen_arm_mujoco/trossen_arm_mujoco/dataset_utils/delta_action_ds/initial_cleanup.py /home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/serl_pkl_files --mode cleanup
```
func is `cleanup_folder`

4. Merge all pkls into one
```bash
python /home/qte9489/personal_abhi/temp/hil-serl/trossen_arm_mujoco/trossen_arm_mujoco/dataset_utils/delta_action_ds/initial_cleanup.py /home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/serl_pkl_files --mode merge --output /home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/merged_recordings.pkl
```

5. Filter static transitions and drop them using `/home/qte9489/personal_abhi/temp/hil-serl/trossen_arm_mujoco/trossen_arm_mujoco/dataset_utils/delta_action_ds/debug_serl_ds_preprocessing.py`
   Output: `merged_recordings_static_dropped.pkl`

6. Convert ALL actions (including grippers) to delta actions:
   ```python
   file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/merged_recordings_static_dropped.pkl"
   output_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/merged_recordings_static_dropped_delta_act.pkl"
   convert_actions_to_delta_act_including_gripper(file_path=file_path, output_file_path=output_file_path)
   ```
   Uses `abs_act_to_delta_act.py` func `convert_actions_to_delta_act_including_gripper()`
   This function converts position, rotation, AND gripper values to deltas (robot frame → world frame transformation included)

7. Check frame consistency:
   ```python
   file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/merged_recordings_static_dropped_delta_act.pkl"
   check_frame_1_consistency(file_path=file_path)
   ```
   Expected output:
   ```plaintext
   ----------------------------------------------------------------------------------------------------
   Analyzed 89 episodes.
   LEFT New Start Mean:  [-0.20590435 -0.01952423  0.18507643]
   LEFT Variation (Std): [2.87736361e-04 6.66118675e-05 2.30620263e-03]

   RIGHT New Start Mean:  [ 0.20603398 -0.0184105   0.19785715]
   RIGHT Variation (Std): [0.0005237  0.00112044 0.00478165]
   ----------------------------------------------------------------------------------------------------
   ✅ GOOD NEWS: The start positions are consistent (Variance < 2cm).
      You CAN use the constant Mean values in your Env Reset.
   ```

8. **If step 7 shows bad news** (high variance), prune first frame using `prune_first_frame_and_save()`

9. Get reference start states for environment reset:
   ```python
   inspect_start_states(file_path=file_path)
   ```
   Expected output:
   ```plaintext
   --------------------------------------------------------------------------------------------------------------------------------------------
   STATISTICS (To copy into your Env Constants):
   LEFT ARM MEAN START:  np.array([-0.20590, -0.01953, 0.18508])
      (Variation/Std):   [0.00029, 0.00006, 0.00230]

   RIGHT ARM MEAN START: np.array([0.20637, -0.01861, 0.19820])
      (Variation/Std):   [0.00022, 0.00012, 0.00422]
   --------------------------------------------------
   REFERENCE QUATERNIONS (From Ep 0):
   LEFT QUAT:  [ 0.99996865 -0.0046836   0.00620599 -0.00149689]
   RIGHT QUAT: [ 9.99941358e-01 -4.89149392e-04  1.07769215e-02 -9.48494596e-04]
   ```
   **NOTE**: Use these values for consistent env.reset() positions

10. Get action statistics to determine scaling factors:
    ```python
    get_per_episode_stats_with_steps(file_path=file_path)
    ```
    Analyze the output to choose appropriate action scales for position, rotation, and gripper
    Example: `ACTION_SCALE = np.array([0.025, 0.06, 0.005])  # [pos, rot, grip]`

11. Normalize ALL delta actions (position, rotation, gripper) using the chosen scales:
    ```python
    ACTION_SCALE = np.array([0.025, 0.06, 0.005])  # based on step 10 analysis
    file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/merged_recordings_static_dropped_delta_act.pkl"
    output_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/merged_recordings_static_dropped_delta_act_scaled.pkl"
    update_action_scale(file_path, output_file_path, ACTION_SCALE)
    ```
    The `update_action_scale()` function now handles all 3 components (position, rotation, gripper) in one go
    Result: All actions normalized to approximately [-1, 1] range

12. Add epsilon to avoid exact edge values:
    ```python
    input_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/merged_recordings_static_dropped_delta_act_scaled.pkl"
    output_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/merged_recordings_static_dropped_delta_act_scaled_epsilon.pkl"
    add_epsilon_to_near_edge_actions(input_file_path=input_file_path, output_file_path=output_file_path)
    ```
    Prevents +1.0 or -1.0 exact values that can cause NaN in loss functions during training

13. Visualize the data to verify and choose binarization threshold:
    ```python
    visualize_data(output_file_path, 'gripper', episode_indices=[0,1,24], data_source='action')
    ```
    Look at the graph, zoom in when gripper is closing, and choose a trigger threshold (e.g., 0.04)

14. Binarize gripper actions using state machine:
    ```python
    input_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/merged_recordings_static_dropped_delta_act_scaled_epsilon.pkl"
    output_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/merged_recordings_static_dropped_delta_act_scaled_epsilon_binarized_gripper.pkl"
    binarize_gripper_deltas(trigger_threshhold=0.04, input_file_path=input_file_path, output_file_path=output_file_path)
    ```
    Converts noisy gripper deltas to clean binary signals: 1.0 = opening, -1.0 = closing
    This prevents the "zero delta trap" where holding causes the gripper to relax and drop objects

15. Visualize final result to verify binarization:
    ```python
    visualize_data(output_file_path, 'gripper', episode_indices=[0,1,24], data_source='action')
    ```
    Should see clean square waves instead of noisy signals

16. Compute action ranges for RL exploration:
    ```python
    input_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/merged_recordings_static_dropped_delta_act_scaled_epsilon_binarized_gripper.pkl"
    compute_action_ranges(input_file_path)
    ```
    Outputs suggested action_space bounds:
    ```plaintext
    ======================================================================
    SUGGESTED ACTION SPACE FOR RANDOM SAMPLING:
    ======================================================================
    action_space = spaces.Box(
        low=np.array([...]),
        high=np.array([...]),
        shape=(14,),
        dtype=np.float32
    )
    ```
    Copy-paste this into your gym env wrapper to fill initial replay buffer with in-distribution samples

17. Verify final dataset by replaying in simulation:
    Use `test_data_replay_sim_env()` from `trossen_arm_mujoco/trossen_arm_mujoco/delta_ee_sim_env_serl_drq_BC.py`
    **CRITICAL**: Ensure `action_scale` in env wrapper matches your processing pipeline:
    ```python
    action_scale = [0.025, 0.06, 0.005]  # Must match ACTION_SCALE from step 11!
    ```

---

## Summary of Pipeline

**Key Changes from Old Pipeline:**
- **Merged gripper conversion**: `convert_actions_to_delta_act_including_gripper()` now handles ALL components (position, rotation, gripper) in one function
- **Single scaling step**: `update_action_scale()` normalizes all 3 components together using 3-element ACTION_SCALE
- **Simpler flow**: Removed the separate `make_delta_gripper_actions()` and `rescale_gripper_actions_and_save()` steps

**Final Dataset Format:**
- Position deltas: normalized to ~[-1, 1] by dividing by position scale (e.g., 0.025m)
- Rotation deltas: normalized to ~[-1, 1] by dividing by rotation scale (e.g., 0.06 rad)  
- Gripper actions: binarized to exactly {-1.0, 1.0} representing close/open commands
- All actions have epsilon applied to avoid exact edge values (except binarized grippers)

**Environment Configuration:**
- Set `action_scale = [pos_scale, rot_scale, grip_scale]` matching your processing
- Set `action_space` bounds from `compute_action_ranges()` output for better exploration


**NOTES:**

1. It is great to hear the grabbing is working! The "scooping" happens because the gripper starts closing *while* the arm is still approaching or lifting—this is actually a very robust way to pick things up (often called "dynamic grasping").

Here is the concise breakdown of the entire pipeline, why we did each step, and why the "Binary + Delta" logic is standard practice.

### 1. The Pipeline Explained (Step-by-Step)

Here is the lifecycle of your gripper data:

1. **Physical Reality (Raw):** The gripper moves from `0.044` (Open) to `0.0` (Closed).
2. **Delta Conversion:** We calculate `change = current - previous`.
* *Result:* Closing is `-0.005` (meters). Holding is `0.0`.

![delta gripper](./delta_gripper.png)


3. **Normalization (Scaling):** The Neural Network learns best when numbers are between -1 and 1. We divide by `0.005`.
* *Result:* Closing becomes `-1.0`. Holding becomes `0.0`.

![delta gripper scaled by 0.005](./delta_gripper_scaled_0_005.png)

4. **The Fix (Binarization):** We noticed the "Zero Delta Trap" (holding = 0.0 causes drop). We overwrote the zeros.
* *Result:* Closing is `-1.0`. Holding is **also** `-1.0`.
![delta gripper scaled by 0.005 binarized](./delta_gripper_scaled_0_005_binarized.png)

5. **Network Output:** The policy outputs `-1.0` ("I want to close").
6. **Env.Step (Execution):** We convert back to physics:
* `Command` = `-1.0` (Network)  `0.005` (Scale) = **`-0.005m`**.
* `Target` = `Current_Pos` + `-0.005m`.



### 2. Why keep scaling in `env.step`?

You asked: *If it predicts -1 or 1, why scale it? Why treat it as delta?*

**Answer: To control Velocity (Speed).**

If you did **not** scale:

* `Target` = `Current (0.04)` + `Network (-1.0)` = **`-0.96 meters`**.
* The robot would try to teleport the fingers 1 meter through the floor in a single timestamp. This causes physics explosions.

By scaling with `0.005`:

* `Target` = `Current (0.04)` + `(-0.005)` = **`0.035 meters`**.
* This tells the robot: **"Move 5mm closer this step."**
* Since the network *keeps* outputting -1.0 every step, the gripper closes smoothly: `0.04`  `0.035`  `0.030`... until it hits the object.

**Essentially:**

* **Network (-1):** "Full Throttle Reverse."
* **Scale (0.005):** "Max Speed of the car."

### 3. Why Binarization? (Is this scientific?)

This approach is widely used in robotic manipulation but often has different names like **"Gripper Action Discretization"** or **"Hysteresis Control."**

* **The Problem it solves:** Human demonstrations are noisy. When you hold an object, you stop moving your fingers (Delta = 0). Behavioral Cloning (BC) naively copies this "stop moving" command, causing the robot to relax its grip and drop the object.
* **The Solution:** By forcing the action to be **-1 (Squeeze)** continuously, we simulate a **constant torque/force**.
* **Scientific Context:** This is a simplified version of *action chunking* or *mode switching*. Papers like *RT-1* or *ACT* often handle grippers separately or use discrete bins (0=Open, 1=Close) precisely because continuous delta-regression for grippers is notoriously unstable.

---

### HIL-SERL Gripper Documentation

**Concept:**
The gripper is controlled via **Continuous Delta Position Control** with a **Binary Intent Signal**.

**1. Data Processing:**

* **Input:** Raw joint positions (0.0m to 0.044m).
* **Transformation:** Calculated deltas ().
* **Normalization:** Scaled by **0.005**. (Physical 5mm  Network 1.0).
* **State Machine Fix:**
* If input < -0.05: State  **-1.0** (CLOSE).
* If input > 0.05: State  **1.0** (OPEN).
* If input  0: **Maintain previous State.**


* **Result:** A clean square-wave signal where `-1.0` represents a constant command to squeeze.

**2. Network Output:**

* The policy outputs a value .
* Due to the fix, the output is typically saturated at **-1.0** or **1.0**.

**3. Execution Logic (`env.step`):**
We treat the network output as a **Velocity Command**.

* **Logic:** This continuously increments the target position downwards (closing) by 5mm per step.
* **Grasping:** Once the fingers physically contact the object, `Current_Pos` stops changing, but the `Target` continues to drive deeper. This discrepancy generates the **motor torque** required to hold the object firmly.