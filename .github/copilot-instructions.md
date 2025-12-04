# AI Coding Agent Instructions for IsaacLab-Laura

## Project Overview

**IsaacLab-Laura** is a robotics manipulation research framework built on [IsaacLab](https://github.com/isaac-sim/IsaacLab), focused on **Chemistry Inspired Manipulation Learning Lab Skills (ChILLS)** — a sim-to-real imitation learning benchmark for lab object manipulation (lift, place, pour tasks).

- **Base:** NVIDIA Isaac Sim + IsaacLab 2.0 (GPU-accelerated physics simulation)
- **Frontend:** Python 3.11, manager-based environments, HDF5 datasets, RoboMimic training
- **Entry point:** `isaaclab.sh` launcher script; ChILLS in `source/chills/`
- **Current branch:** `skillgen_demo`

---

## Critical Architecture Patterns

### 1. **Configuration-Driven Design via `@configclass`**
- All tasks, environments, and agents use `isaaclab.utils.configclass` decorator
- No `__init__` in config classes; prefer dataclass field defaults
- Example: `FrankaIkEnvCfg` extends `ManagerBasedRLEnvCfg`, sets `sim.dt=0.01`, `decimation=2`
- **Location:** `source/chills/chills/envs/franka_env.py`, `source/isaaclab_tasks/`
- **Key convention:** Use `__post_init__()` to override parent configs cleanly

### 2. **Manager-Based Environment Workflow** (not DirectRL)
- ChILLS uses **manager-based** environments with modular MDP components:
  - **Observations:** `observations_cfg()` → stacked obs terms (e.g., ee_pos, object_pos)
  - **Actions:** `FrankaActionsCfg` → maps deltas to joint commands
  - **Rewards:** `rewards_cfg()` → scalar reward per step
  - **Terminations:** `terminations_cfg()` → episode end conditions (success, failure, time)
- Each task inherits from `ManagerBasedRLEnvCfg` and composes scene + MDP configs
- **Location:** `source/chills/chills/envs/`, `source/chills/chills/tasks/`

### 3. **Data Pipeline: 1_recorded → 2_annotated → 3_collected → Gen**
- **1_recorded:** Raw human demos (HDF5) from teleop (`scr/1_record.py`)
- **2_annotated:** Trimmed, labeled with key events (`scr/2_annotate.py`)
- **3_collected:** Consolidated canonical dataset (`scr/3_consolidate.py`)
- **Generated:** Mimic-augmented demos (×10–100×) in `datasets/chills_lfs/` (`scr/4_generate.py`)
- **Training:** Uses any collected/generated HDF5; outputs to `models/runs/<Task>/<algo>/<seed>/`
- **File naming convention:** `<x>_dataset24x.hdf5`, `<x>_collected.hdf5`, `<x>0to<y>kGen.hdf5`

### 4. **Mimic Augmentation (Data Generation)**
- Uses `isaaclab_mimic` extension (separate from core ChILLS tasks)
- Converts **BC (behavioral cloning) dataset** → **large synthetic episodes** via in-sim randomization
- Special env variants: `LiftLabwareMimic-v0`, `PlaceLabwareMimic-v0` (mimic-only)
- Key config: `env_cfg.datagen_config` with `generation_num_trials`, `generation_keep_failed`, `generation_relative`
- **Location:** `source/isaaclab_mimic/isaaclab_mimic/datagen/`, `source/chills/chills/envs/`

### 5. **Quaternion Convention: wxyz (Always)**
- Isaac Lab uses **wxyz** (w=scalar, xyz=vector)
- Never mix with xyzw or Euler angles without explicit conversion
- Use `isaaclab.utils.math` for transforms; do not roll custom conversions
- Example: `rot=[0.707, 0, 0, 0.707]` is 90° around z-axis

---

## Build & Execution Workflows

### Environment Setup
```bash
# Create conda/uv environment (Isaac Sim 4.5 or 5.0)
./isaaclab.sh -c env_isaaclab  # conda (auto-selects Python 3.10 for sim 4.5)
# OR
./isaaclab.sh -u env_isaaclab  # uv

# Activate, then install ChILLS
conda activate env_isaaclab
cd source/chills
pip install -e .
```

### Core Scripts (Numbered Pipeline)
All scripts in `source/chills/chills/scr/` follow a consistent CLI pattern (via `mods/app_parser.py`):

| Script | Purpose | Example |
|--------|---------|---------|
| `0_teleop.py` | Device setup & sanity checks | `python scr/0_teleop.py --task LiftLabware-v0 --teleop_device gamepad` |
| `1_record.py` | Record human demos | `python scr/1_record.py --task LiftLabware-v0 --dataset_file datasets/LiftLabware/1_recorded/10x.hdf5 --num_demos 10` |
| `2_annotate.py` | Label key events, filter | `python scr/2_annotate.py --input_file ... --output_file ... --auto --num_success_steps 30` |
| `3_consolidate.py` | Merge multiple HDF5s | `python scr/3_consolidate.py --inputs "datasets/*/2_annotated/*.hdf5" --output_file ... ` |
| `4_generate.py` | Mimic augmentation | `python scr/4_generate.py --task LiftLabwareMimic-v0 --num_envs 128 --generation_num_trials 4000 --input_file ... --output_file ...` |
| `5_replay.py` | Visualize dataset episodes | `python scr/5_replay.py --task LiftLabware-v0 --input_file ... --sample 5` |
| `6_add_key_obs.py` | Inject observation keys | `python scr/6_add_key_obs.py --input_file ... --output_file ... --keys ...` |
| `7_train.py` | Train RoboMimic models | `python scr/7_train.py --task PlaceLabware-v0 --dataset ... --algo bc_rnn --seed 0 --device cuda` |
| `8_play.py` | Run trained checkpoint | `python scr/8_play.py --task PlaceLabware-v0 --checkpoint models/runs/.../best_val.pth --episodes 10` |
| `9_gather_best_val.py` | Harvest best checkpoints | `python scr/9_gather_best_val.py --runs_root runs/ --dest_root best_runs/` |

### High-Level Orchestrators
- **`run.py`** (authoritative): `python run.py pipeline -E PlaceLabware -A bc_rnn bc_trans --runner auto`
- **`go.sh`** (bash): `ENV_NAME=PlaceLabware ./go.sh 4 6` (steps 4=generate, 6=train)
- **`auto.ps1`** (PowerShell): `./auto.ps1 -EnvName PlaceLabware -Algos bc_rnn -Script 4,6`

**Always use `isaaclab.sh -p` (via `--runner auto`) to execute Python scripts** so Isaac Sim environment is properly loaded.

---

## Project-Specific Conventions

### 1. **Naming & File Organization**
- **Tasks:** `<Task>Labware`, e.g., `LiftLabware-v0`, `PlaceLabware-v0` (gym registry)
- **Task classes:** `source/chills/chills/tasks/lift_labware.py`
- **Task configs:** `source/chills/chills/tasks/<task>_env_cfg.py`
- **Datasets root:** `datasets/<Task>/[1_recorded|2_annotated|3_collected]/`
- **Generated datasets:** `datasets/chills_lfs/<Task>/<count>to<count>Gen.hdf5`
- **Model checkpoints:** `models/runs/<Task>/<algo>/<seed>/<dataset_name>/best_val_model.pth`

### 2. **Asset & Labware Factory**
- USD labware stored in `source/chills/chills/assets/` (e.g., `beaker/beaker.usd`)
- Factory functions: `source/chills/chills/envs/labware_cfg.py` → `beaker()`, `conical_flask()`, `hotplate()`, `scale()`
- All assets use semantic tags: `[("class", "beaker")]`
- Frame transforms: Use `FrameTransformerCfg` with `OffsetCfg` for EE-relative observations

### 3. **Environment Control: Absolute Targets ↔ Delta Actions**
- **ChillsMimicEnv** wraps base env to convert:
  - Absolute EE target pose → delta joint commands (IK-resolved)
  - Subtask logic: phases, success checks, object reference tracking
- EE frame always in **world coordinates**; observations translated to EE-relative
- **Location:** `source/isaaclab_mimic/isaaclab_mimic/envs/ChillsMimicEnv.py`

### 4. **RoboMimic Training Algorithms**
Supported configs in `source/chills/chills/tasks/agents/robomimic/`:
- **BC-RNN:** `bc_rnn.json` (recurrent policy, good for long horizons)
- **BC-Transformer:** `bc_trans.json` (transformer-based, preferred for longer sequences)
- **HBC:** `hbc.json` (hierarchical behavior cloning)
- Upcoming: Diffusion models (RoboMimic ≥0.5.0)

### 5. **Logging & Reproducibility**
- TensorBoard logs: `datasets/<Task>/logs/` or `models/runs/<Task>/<algo>/<seed>/`
- Script execution logs: `logs/script_logger.md` (managed by `run.py`)
- **Always fix seed** for reproducible runs: `--seed <int>`
- Build version tracking: `isaaclab.sh` detects Isaac Sim 4.5 vs 5.0; logs printed at startup

### 6. **Headless vs Viewer Rendering**
- `--headless` for large-scale data generation (faster, no GPU overhead for display)
- Viewer mode for debugging / sanity checks; default eye: `(1.0, 1.0, 1.0)`, lookat: `(0.0, 0.0, -0.1)`
- Translucent rendering enabled by default in base Franka config for glass/fluid assets

---

## Key Files & Where to Look

| Purpose | Location |
|---------|----------|
| Base Franka scene & env config | `source/chills/chills/envs/franka_env.py` |
| Task definitions (lift, place, pour) | `source/chills/chills/tasks/lift_labware.py`, `place_labware.py` |
| MDP module examples | `source/chills/chills/envs/mdp/` (observations, rewards, terminations) |
| Labware factory & presets | `source/chills/chills/envs/labware_cfg.py` |
| Common CLI argument parser | `source/chills/chills/scr/mods/app_parser.py` |
| Mimic env wrapper | `source/isaaclab_mimic/isaaclab_mimic/envs/ChillsMimicEnv.py` |
| Data generation orchestrator | `source/isaaclab_mimic/isaaclab_mimic/datagen/generation.py` |
| Task registration & gym entry points | `source/isaaclab_tasks/isaaclab_tasks/` |
| Isaac Lab core docs & reference arch | `docs/source/refs/reference_architecture/index.rst` |

---

## Common Debugging Patterns

1. **"No success termination term found"** → Ensure task config has `terminations_cfg()` with `success` term
2. **Quaternion rotation looks wrong** → Verify wxyz ordering; check frame transform direction
3. **Viewer hangs or TLAS warnings** → Use `--headless` or reduce scene complexity; verify GPU driver
4. **USD asset missing** → Confirm file exists at `source/chills/chills/assets/<name>/<name>.usd`
5. **Dataset file mismatch** → Filename must match pipeline expectation (`_recorded`, `_annotated`, `_collected`, `Gen`)
6. **Training converges poorly** → Check observation scaling, action ranges, reward shaping; verify dataset quality with `5_replay.py`

---

## Testing & Validation

- **Unit tests:** `source/chills/chills/test.py` or pytest in `tools/`
- **Data sanity checks:** Always replay a random sample before training: `python scr/5_replay.py --sample 5`
- **Pre-commit hooks:** `./isaaclab.sh -f` runs formatters/linters (isort, black, flake8)
- **Pytest:** `./isaaclab.sh -t` runs all tests with timeout guards

---

## External Dependencies & Integration Points

- **IsaacLab core:** `isaaclab.*` (assets, envs, sensors, sim utils)
- **Isaac Sim binary:** Loaded via `_isaac_sim/` symlink or pip (`isaacsim-rl`)
- **RoboMimic:** Dataset loading, policy training, inference in `isaaclab_mimic`
- **HDF5 I/O:** `h5py` for demo storage & augmentation
- **Hydra:** Config management in IsaacLab tasks (not used in ChILLS scripts directly)
- **Gym registry:** Standard `gymnasium` for environment registration

---

## Writing New Code

When adding features (e.g., new task, MDP module):

1. **Define config class** with `@configclass` decorator (no `__init__`)
2. **Inherit from appropriate base** (`ManagerBasedRLEnvCfg`, `SubTaskConfig`, etc.)
3. **Use official Isaac Lab utilities** (`isaaclab.utils.math`, `isaaclab.sim.*`) — don't roll custom transforms
4. **Maintain naming conventions** (`<Task>Labware-v0`, `_env_cfg.py`, `_collected.hdf5`)
5. **Add docstrings** to modules & classes (author, purpose, high-level design)
6. **Log execution** via existing patterns (`scr/mods/logger.py` or native Python logging)
7. **Test locally** with `--headless` before submitting; commit with passing unit tests

---

## References

- **ChILLS README:** `source/chills/README.md` (quick-start, pipeline overview)
- **ChILLS scripts README:** `source/chills/chills/scr/README.md` (detailed per-script guide)
- **IsaacLab docs:** `docs/source/refs/reference_architecture/index.rst` (architecture walkthrough)
- **PyProject config:** `pyproject.toml` (import sections, tool configs, version constraints)
- **Launcher script:** `isaaclab.sh` (env detection, Isaac Sim path resolution, Python exe extraction)
