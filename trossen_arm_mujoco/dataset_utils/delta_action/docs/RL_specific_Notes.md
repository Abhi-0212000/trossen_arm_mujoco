### Things to make RL Faster
1. Bounding Box creation.
    - Run the original dataset(with actions in sim world coord frame) i.e `original_ds_with_act_in_sim_world_frame = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame.pkl"` using func `compute_action_ranges(original_ds_with_act_in_sim_world_frame)` in `/home/qte9489/personal_abhi/temp/hil-serl/trossen_arm_mujoco/trossen_arm_mujoco/dataset_utils/delta_action_ds/abs_act_to_delta_act.py`.

    you get the following data. Notice the max, min of Right Robot Position. i.e **Z range is (min=-0.0058, max= 0.3156)** and **X range is (min=-0.0648, max= 0.2080)** and **Y range is (min=-0.0549, max= 0.3082)**
    ```bash
    ======================================================================
    PER-DIMENSION BREAKDOWN:
    ======================================================================
    0. L_pos_x   : min=-0.2068, max=-0.2056, mean=-0.2059, std= 0.0003
    1. L_pos_y   : min=-0.0197, max=-0.0193, mean=-0.0195, std= 0.0001
    2. L_pos_z   : min= 0.1814, max= 0.1966, mean= 0.1849, std= 0.0019
    3. L_rot_x   : min=-0.0102, max=-0.0071, mean=-0.0095, std= 0.0006
    4. L_rot_y   : min=-0.0166, max= 0.0555, mean= 0.0331, std= 0.0071
    5. L_rot_z   : min=-0.0036, max=-0.0018, mean=-0.0028, std= 0.0003
    6. L_grip    : min= 0.0400, max= 0.0400, mean= 0.0400, std= 0.0000
    7. R_pos_x   : min=-0.0648, max= 0.2080, mean= 0.1029, std= 0.0670
    8. R_pos_y   : min=-0.0549, max= 0.3082, mean= 0.0536, std= 0.0867
    9. R_pos_z   : min=-0.0058, max= 0.3156, mean= 0.1329, std= 0.0954
    10. R_rot_x   : min=-0.2895, max= 0.2135, mean=-0.0328, std= 0.0576
    11. R_rot_y   : min=-1.0830, max= 0.0368, mean=-0.4301, std= 0.3138
    12. R_rot_z   : min=-0.8635, max= 0.1830, mean=-0.1694, std= 0.2276
    13. R_grip    : min=-0.0004, max= 0.0401, mean= 0.0240, std= 0.0183
    ```
    - Since Z min is in NEGATIVE --> below the table. 
    - Example Left and Right bounds. Visualize them using `python /home/qte9489/personal_abhi/temp/hil-serl/trossen_arm_mujoco/trossen_arm_mujoco/dataset_utils/visualize_bbox.py --mode visualize`
    ```plaintext
    LEFT_CARTESIAN_BOUNDS = (
        np.array([-0.25, -0.13, 0.05]),  # MINS
        np.array([ -0.17,   0.1, 0.25])  # MAXS
    )

    # --- RIGHT ROBOT (Worker) ---
    # Format: ([x_min, y_min, z_min], [x_max, y_max, z_max])
    # Updated from dataset analysis in visualize_bbox.py
    RIGHT_CARTESIAN_BOUNDS = (
        np.array([-0.10, -0.1, 0.005]),  # MINS  
        np.array([ 0.25,  0.32,  0.35])   # MAXS
    )
    ```
    - Clean the Dataset by clipping the Z properly `ex: atleast 1 or 2 mm above the table`. Replay the DS, verify everything and delete episodes that are not good or mark them as terminated by letting them play for full epoisode length.
    - Final DS `"/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_clipped_inBbox_fulldelta_act_scaled_act_binarized_epsilon_regenerated_deleted_some.pkl"`

    ```json
        {
            "action_scale": [0.025, 0.1, 0.02],
            "gripper_trigger_threshold for binarization": 0.005,
            "action_space": {
                "low": [-0.05265355855226517, -0.007589314598590136, -0.18495754897594452, -0.1017114594578743, -0.11622431874275208, -0.03330891206860542, 0.9998999834060669, -0.5016695261001587, -0.515146017074585, -0.8840358257293701, -0.5616023540496826, -0.9998999834060669, -0.6274166703224182, -0.9998999834060669],
                "high": [0.002395081566646695, 0.00666593573987484, 0.4290461540222168, 0.003975508734583855, 0.49415484070777893, 0.007743919268250465, 0.9998999834060669, 0.3446878492832184, 0.6354304552078247, 0.8687211275100708, 0.5261103510856628, 0.5571752786636353, 0.5912928581237793, 0.9998999834060669],
                "shape": [14],
                "dtype": "float32"
            },
            "Suggested Mean position of robots after reset": {
                "left mean": [-0.20590435, -0.01952423, 0.18507643],
                "right mean": [ 0.20603398, -0.0184105, 0.19785715]
            }
        }
    ```



### Interpreting Graphs
configuration (**Sparse Reward +1, Discount 0.99, Backup Entropy False**):

#### 1. Critic Predicted Qs (`predicted_qs`)

* **Goal:** Go towards **100**.
* **Why?** With `discount=0.99`, the maximum possible value is .
* **What it means:**
* **0:** The robot is failing completely.
* **30:** The robot is succeeding sometimes (or staying in the goal zone for 30% of the horizon). **This is good progress.**
* **100:** The robot is perfect. It reaches the goal immediately and holds it there forever.
* **>100:** Your critic is broken (hallucinating).



#### 2. Actor Loss (`actor_loss`)

* **Goal:** Go **as negative as possible** (specifically towards **-100**).
* **Why?** The actor wants to maximize the Q-value. Since `Loss = -Q`, as Q goes to 100, Loss goes to -100.
* **What it means:**
* **0:** The robot is clueless.
* **-30:** The robot is learning (corresponds to your Q of 30).
* **Positive:** Something is wrong (unless you have very high entropy, but with `backup_entropy=False`, positive loss usually means your Q-values are negative, which shouldn't happen with sparse +1 rewards).


| Metric | Bad Value ❌ | Good / In-Progress ⚠️ | Perfect / Converged ✅ |
| --- | --- | --- | --- |
| **Predicted Q** | Near 0 (Stuck) | **30 to 80** (Learning) | **~100** (Mastery) |
| **Actor Loss** | Near 0 | **-30 to -80** | **~ -100** |




Based on the **RLPD** and **SERL** papers (and the standard DrQ-v2 / SAC logic they are built on), here is the definitive answer regarding how to handle `dones` and `masks` in your demo buffer.

### The Short Answer

**For your "Stack and Hold" (Hovering) task, you should keep `done=False` (Mask=1) at the end of the demo episodes.**

You should **ONLY** set `done=True` (Mask=0) if the episode ended because of a **failure** (e.g., robot broke, safety violation) or if the task is strictly "one-shot" and cannot physically continue (like dropping an object into a bin where it disappears).

---

### The Detailed Explanation from the Papers

#### 1. Why `done=False` (Mask=1) is correct for you

Both RLPD and SERL rely on **Q-Learning** (specifically DrQ-v2 / SAC). In Q-learning, the "Done" flag fundamentally changes how the Value of a state is calculated:

* **If `done=True` (Mask=0):**



The algorithm assumes the world **ends** here. The future value is zero.
* *Problem:* If you mark your successful "Hovering" state as `done=True`, the agent learns that **Success = End of Rewards**. It will think that holding the cube is worth *less* than approaching it (because approaching allows for future rewards, while holding kills the episode).


* **If `done=False` (Mask=1):**



The algorithm assumes the task **continues**. It bootstraps the value from the next state.
* *Benefit:* This tells the agent: "You are in a high-reward state, and if you stay here, you will KEEP getting high rewards forever." This creates the massive value accumulation () that forces the policy to stabilize and hover.



#### 2. Evidence from RLPD

RLPD explicitly builds on **DrQ-v2** and focuses on "infinite horizon" style tasks where possible.

* **Page 17 (D4RL AntMaze Example):** The paper notes that for some sparse tasks, "Upon reaching the goal, the episode terminates." In these specific sparse goal cases, `done=True` is used.
* **However, for Manipulation (Adroit/Kitchen):** These are typically treated as continuous control tasks. The standard practice in the underlying implementations (like DrQ-v2) is to treat "Success" as a transition to a sink state (or reset) but mathematically handle it as a time-limit truncation (`done=False`) so the value function doesn't collapse.

#### 3. Evidence from SERL

SERL is designed for **Real-World** manipulation.

* In the real world, "episodes" are artificial concepts. The robot doesn't vanish when it succeeds; it physically stays there holding the object.
* **Demo Buffer Construction:** When you record human demos, you usually stop recording when the task is done. This is an **artificial time limit**, not a physical termination.
* Therefore, the correct mathematical handling is **Bootstrapping** (`done=False`).

### How to configure your Demo Buffer

When you save your pickle file, ensure the transitions at the end of your successful demonstrations look like this:

| Field | Value | Reason |
| --- | --- | --- |
| **Reward** | `High` (e.g. 3.6) | Robot is in success state. |
| **Done** | `False` | The robot *could* continue holding it if we let it. |
| **Mask** | `1.0` | Allow bootstrapping (Future Value is valid). |

**Exception (When to use `done=True`):**
If your demo contains **failed attempts** (e.g., the robot knocked the cube off the table), the last step of that failed trajectory should be `done=True` (Mask=0) to tell the network "This path leads to death/failure."

### Correct Logic for your `regenerate_ds_state` script

In your regeneration script, you are currently doing this correctly by extending the episode with `done=False` loops. When you finally hit the hard limit `max_episode_length`, standard Gym logic says `truncated=True`, but for the replay buffer training, we usually store this as `mask=1` (bootstrapping allowed) because it was a time limit, not a failure.

**Recommendation:** In your final dataset pickle, for successful episodes, set `dones=False` (or `0.0`) and `masks=1.0` for **ALL** steps, including the last one.