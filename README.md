[README_SwarmShield_FL(1).md](https://github.com/user-attachments/files/28632718/README_SwarmShield_FL.1.md)
# SwarmShield-FL — Trust-Based Decentralized Federated Learning

**Heart Rate Prediction · Poisoning Detection · Gossip Reputation**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dimitris-Gerakas-CERTH/SwarmShield-FL/blob/main/Trust_FL_Experiment_Colab.ipynb)

> This codebase is the experimental backbone of the paper *"Shielding Swarm Intelligence with Gossip Reputation for Robust Heart Rate Prediction against Data Poisoning"* (ICE 2026, Paper 361).  
> **Runs on Google Colab only** — click the badge above to launch directly. Local and Codespaces execution are not currently supported.

---

## Overview

SwarmShield-FL is a **fully decentralized, peer-to-peer federated learning system** for physiological time-series prediction. There is no central aggregator. Each node holds its own model, trains on one patient's data, and evaluates neighbors individually before deciding whether to accept their weights.

The core research question: *can a network of nodes detect and isolate data-poisoning attackers using only local MAE-based peer evaluation and gossip reputation — without any central authority?*

The answer the system demonstrates: yes, and the detection is one-sided by design (see [Trust Mechanism](#trust-mechanism) below).

---

## Repository Structure

```
SwarmShield-FL/
├── Trust_FL_Experiment_Colab.ipynb    # Interactive Colab notebook (start here)
├── cleaned_data/                      # Patient CSVs — 22 files
│   ├── patient_01_clean.csv
│   ├── patient_02_clean.csv
│   └── ...
└── README.md
```

The notebook embeds `trust_fl_experiment.py` as a base64 blob and writes it to disk at runtime. This makes the notebook fully self-contained — no separate script file needs to be present in the repo.

---

## Dataset

- **Source:** 22 patients, wearable accelerometer + heart rate recordings
- **Format:** CSV per patient, columns `timestep / axis1 / axis2 / axis3 / hr`
- **Location:** `cleaned_data/` folder in the repo
- **Privacy note:** Data contains real physiological signals. Handle accordingly.

Each node in an experiment is assigned one patient. Nodes do not share raw data — only model weights.

**Quick mode (default):** `reduced_fraction=0.05` uses 5% of each patient's time series, which makes a full experiment run in ~1–2 minutes on CPU. For more stable results, increase to `0.2`–`1.0`.

---

## Architecture

### Node Lifecycle (one FL round)

```
1. LOCAL TRAINING    Each node trains its MLP on its own patient data (Adam, MSE loss)
        ↓
2. WEIGHT BROADCAST  All nodes share their current model weights (simulated in-memory)
        ↓
3. PEER EVALUATION   Each node evaluates each neighbor 1-on-1:
                       - Temporarily merge own weights + neighbor weights (FedAvg)
                       - Evaluate merged model on local test set
                       - If MAE degrades > threshold → RED vote (penalize trust)
                       - Otherwise           → GREEN vote (reward trust)
        ↓
4. GOSSIP (optional) Each node shares its red/green verdicts with trusted neighbors.
                     Reports are weighted by the reporter's trust score.
        ↓
5. AGGREGATION       Each node aggregates only with neighbors that pass ALL three checks:
                       (a) cumulative trust score > 0
                       (b) passed this round's direct evaluation (on our whitelist)
                       (c) bidirectional: neighbor also whitelisted us (they trust us too)
```

### Model

`HRPredictorMLP` — a simple 2-layer MLP:

```
Input (window_size × 4 features) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1)
```

Input size = `window_size * 4` (default: 10 steps × 4 channels = 40). The sliding window includes `axis1, axis2, axis3, hr` from the past 10 timesteps; the target is the next `hr` value.

---

## Trust Mechanism

### Scoring

Each node maintains a `TrustScore` per neighbor that accumulates across rounds:

| Event | Score change |
|-------|-------------|
| Green vote (merge improves MAE) | `+ green_reward` (default +0.1) |
| Red vote (merge degrades MAE) | `− red_penalty` (default −0.3) |
| Gossip: suspicious report | `− red_penalty × gossip_weight × reporter_trust_factor` |
| Gossip: clean report | `+ green_reward × gossip_weight × reporter_trust_factor` |

A node is **blocked** when its cumulative score drops to ≤ 0.

**Best parameters found:** `threshold=0.08`, `green=0.15`, `red=0.25`, `warmup=2`, `gossip_weight=0.5`

### Why Detection is One-Sided (important for understanding results)

A poisoned node shares corrupted weights. When a **clean node** merges with those weights, its local MAE worsens → red vote → poisoned node gets penalized → eventually blocked.

The reverse does **not** happen: when a **poisoned node** evaluates a clean neighbor, the clean model actually *improves* the poisoned node's MAE (it partially corrects the poisoned distribution) → the merge test passes → the poisoned node gives a green vote to clean nodes.

**Consequence:** poisoned nodes do not flag clean nodes as suspicious. Detection flows entirely from clean nodes toward poisoned ones. The bidirectional sharing mechanism works as a **one-sided quarantine**: once a clean node detects a poisoned neighbor, it stops exporting its own weights to that neighbor, preventing the poisoned node from free-riding on clean gradients.

This asymmetry is a deliberate and validated design feature, not a limitation.

---

## Gossip Reputation

Enabled with `gossip=True`. After each peer evaluation round:

1. Each node builds a report: `{neighbor_id: is_suspicious}` based on its current-round whitelist
2. Reports are forwarded only to **trusted** neighbors (on the whitelist)
3. Receiving nodes update trust scores for the reported node, scaled by `gossip_weight × reporter_trust_factor`
4. The `reporter_trust_factor` normalizes the reporter's cumulative score to `[0, 1]` — high-trust reporters carry more weight

Gossip accelerates detection in sparse topologies where not all nodes have direct contact with all others.

---

## Attack Strategies

All attacks modify patient data before training. The `apply_poison()` function supports:

| Attack | What it does |
|--------|-------------|
| `label_noise` | Adds Gaussian noise to HR values (controlled by `noise_std`) |
| `label_flip` | Reflects HR around the mean: `hr → 2*mean - hr` |
| `label_constant` | Replaces HR with the mean value (flattens signal) |
| `feature_noise` | Adds noise to accelerometer axes |
| `temporal_shift` | Shifts HR by 10% of the time series length |
| `combined_subtle` | Mild noise on both HR and accelerometer |
| `combined_aggressive` | Strong noise on both HR and accelerometer |

**Most effective for detection experiments:** `label_noise` with `noise_std=0.8–0.9` or `label_flip`.

---

## Running Experiments

> ⚠️ **Google Colab only.** This notebook is designed and tested exclusively for Google Colab. Local execution and GitHub Codespaces are not currently supported. All dependencies are pre-installed in the Colab environment and the notebook relies on Colab's runtime for GPU access.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dimitris-Gerakas-CERTH/SwarmShield-FL/blob/main/Trust_FL_Experiment_Colab.ipynb)

### Colab — step by step

Open `Trust_FL_Experiment_Colab.ipynb` via the badge above, then run cells in order:

1. **Setup** — installs dependencies
2. **Get Data & Script** — clones repo, writes script
3. **Verify** — checks imports and GPU
4. **Clean Experiment** — baseline, no poisoning
5. **Poisoned WITHOUT Gossip** — baseline detection rate
6. **Poisoned WITH Gossip** — compare gossip benefit
7. **Attack Comparison** — runs all attack types, tabulates results
8. **Topology Comparison** *(optional)*
9. **Threshold Sensitivity** *(optional)*

Set runtime to GPU (`Runtime → Change runtime type → T4`) for ~3–5× speedup.

### Script (CLI)

```bash
# Clean experiment
python trust_fl_experiment.py --nodes 7 --rounds 5

# With 2 poisoned nodes (patients 1 and 2)
python trust_fl_experiment.py --nodes 7 --rounds 5 --poisoned 2 --attack label_noise

# Specific poisoned patients
python trust_fl_experiment.py --nodes 7 --rounds 5 --poisoned_nodes 3 7 --attack label_flip

# With gossip enabled
python trust_fl_experiment.py --nodes 7 --rounds 5 --poisoned 2 --gossip --gossip_weight 0.5

# Full parameter set
python trust_fl_experiment.py \
    --nodes 8 --rounds 7 --epochs 2 \
    --topology full --threshold 0.08 \
    --poisoned_nodes 4 5 --attack label_noise --noise_std 0.9 \
    --green_reward 0.15 --red_penalty 0.25 \
    --gossip --gossip_weight 0.5 \
    --random_window --seed 42
```

### Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--nodes` | 7 | Number of patients (max 22) |
| `--rounds` | 5 | FL rounds |
| `--epochs` | 1 | Local epochs per round |
| `--topology` | `full` | `full / star / ring / line` |
| `--threshold` | 0.10 | MAE degradation tolerance for red vote |
| `--reduced_fraction` | 0.05 | Data fraction (0.05 = quick mode) |
| `--poisoned` | 0 | Shorthand: poison first N nodes |
| `--poisoned_nodes` | — | Explicit patient IDs, e.g. `3 7` |
| `--attack` | `label_noise` | See attack table above |
| `--noise_std` | 0.5 | Noise intensity (0.8–0.9 for strong attacks) |
| `--green_reward` | 0.1 | Trust score increase per green vote |
| `--red_penalty` | 0.3 | Trust score decrease per red vote |
| `--gossip` | off | Enable gossip reputation |
| `--gossip_weight` | 0.5 | Gossip influence (0 = off, 1 = same as direct eval) |
| `--random_window` | off | Sample random time window per patient |
| `--seed` | 42 | Reproducibility |

---

## Reading the Output

A typical run prints:

```
ROUND 3/5
  Local training...
    P01: train_loss=0.0234
    ...
  Peer evaluation (threshold: 8% MAE degradation)...
    P01 → P03 ☠: MAE 0.1823→0.2941 (+61.3%) 🔴 RED
    P01 → P02: MAE 0.1823→0.1791 (-1.7%) ✅ PASS
  Trust-based aggregation (bidirectional)...
    P01: 5/6 accepted [P02:✅, P03 ☠:🔴flagged, ...]
```

The final summary shows:
- **Trust scores** per node, per neighbor (cumulative score, green/red vote counts)
- **Detection results**: which poisoned nodes were blocked by which clean nodes
- **Round-by-round MAE table**: clean avg, poisoned avg, overall avg
- **Final improvement %** for clean nodes

---

## Extension Points for Collaboration

The following are natural places to instrument or extend the codebase:

**For gradient-level work (FL side):**
- `FLNode.train_local()` — hook here to log per-batch gradients before weight sharing
- `FLNode.get_weights()` / `FLNode.set_weights()` — intercept weight exchange to simulate a curious aggregator or gradient inversion

**For split learning comparison:**
- The MLP can be split at any layer. A cut between layer 1 and layer 2 would expose the 64-dim intermediate activations as "smashed data" — a natural analog to split learning's cut-layer output
- `HRPredictorMLP` forward pass would need to be split into `encode()` and `decode()` methods

**For privacy analysis:**
- The training loop is synchronous and in-memory, making it easy to snapshot gradients at any round
- Patient data is 1D time-series (not images) — reconstruction quality metrics (DTW, MSE on raw signal, HR label leakage) are open research questions

**Warm-up rounds (not yet in notebook):**
- Adding a `warmup_rounds` parameter (free aggregation before trust scoring begins) improves clean-node MAE by ~51% vs ~42% without degrading detection. The parameter structure supports this; it just needs wiring into `run_experiment()`.

---

## Known Limitations

- **Simulation only:** all communication is in-memory; no actual network stack
- **Synchronous rounds:** all nodes train and evaluate in lockstep; no asynchrony or dropout
- **Small dataset:** 22 patients, 5% data fraction in quick mode — results are directional, not production benchmarks
- **No differential privacy:** gradients/weights are shared without noise; the system addresses *Byzantine robustness*, not *privacy*. This gap is intentional and is a natural research extension.

---

## Dependencies

```
torch>=1.12
pandas
numpy
```

All pre-installed on Google Colab — no manual installation needed. The Setup cell in the notebook runs `pip install` as a safety check but these packages are available by default in any Colab runtime.

---

## Citation

If you build on this work, please cite:

```
[ICE 2026 Paper 361 — citation details to be added post-publication]
CoGNETs Horizon Europe project, Grant Agreement No. 101135930
```

---

*Last updated: June 2026 — Dimitris Gerakas, ITI-CERTH*
