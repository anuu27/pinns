from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.pipeline import (
    build_model,
    build_solver,
    build_simulator,
    build_datasets,
)

from src.utils.config import ConfigLoader, ensure_directories
from src.utils.seed import set_seed
from src.utils.device import DeviceManager
from src.utils.checkpoint import CheckpointManager

from src.data.dataset import FEATURE_SCALES


# ======================================================
# Load model
# ======================================================

config = ConfigLoader.load(Path("configs/config.yaml"))
ensure_directories(config)
set_seed(config.seed)

device_manager = DeviceManager(
    preferred_device=config.trainer.device,
    prefer_mixed_precision=False,
)

model = build_model(config)

checkpoint_path = (
    Path(config.paths.checkpoints_dir)
    / "best_model.pt"
)

checkpoint = CheckpointManager(
    config.paths.checkpoints_dir
).load(
    checkpoint_path,
    map_location=device_manager.device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device_manager.device)
model.eval()


# ======================================================
# Build datasets
# ======================================================

train_dataset, _, _, _ = build_datasets(config)

solver = build_solver(config)

simulator = build_simulator(config)

policy = simulator.generate_random_policies(1)[0]

print("\nPolicy used for comparison:\n")
print(policy)


# ======================================================
# Sweep interest rates
# ======================================================

rates = np.linspace(0.01, 0.08, 30)

predicted_reserves = []
true_reserves = []

# ======================================================
# Compare PINN vs Thiele Solver
# ======================================================

for r in rates:

    # -------------------------
    # Build feature vector
    # -------------------------
    features = torch.tensor(
        [[
            0.0,
            float(policy.age),
            r,
            policy.premium,
            policy.sum_assured,
            policy.mortality_profile.intensity_at(0.0),
        ]],
        dtype=torch.float32,
    )

    # -------------------------
    # Normalize features
    # -------------------------
    features[:, 0] /= FEATURE_SCALES["time"]
    features[:, 1] /= FEATURE_SCALES["age"]
    features[:, 2] /= FEATURE_SCALES["interest_rate"]
    features[:, 3] /= FEATURE_SCALES["premium"]
    features[:, 4] /= FEATURE_SCALES["sum_assured"]
    features[:, 5] /= FEATURE_SCALES["mortality"]

    features = features.to(device_manager.device)

    # -------------------------
    # PINN prediction
    # -------------------------
    with torch.no_grad():

        z = model(features).item()

        reserve = (
            z * train_dataset.target_std
            + train_dataset.target_mean
        )

        reserve *= policy.sum_assured

    predicted_reserves.append(reserve)

    # -------------------------
    # Classical Thiele Solver
    # -------------------------
    policy.interest_rate = float(r)

    trajectory = solver.solve(
        policy=policy,
        num_steps=config.data.time_steps,
    )

    true_reserve = float(trajectory.reserves[0])

    true_reserves.append(true_reserve)

    print(
        f"Rate={r:.4f} | "
        f"PINN={reserve:.2f} | "
        f"Thiele={true_reserve:.2f}"
    )


# ======================================================
# Convert to numpy arrays
# ======================================================

predicted_reserves = np.asarray(predicted_reserves)
true_reserves = np.asarray(true_reserves)

print("\n")
print("Predicted shape :", predicted_reserves.shape)
print("True shape      :", true_reserves.shape)

assert len(predicted_reserves) == len(true_reserves)

assert not np.isnan(predicted_reserves).any()
assert not np.isnan(true_reserves).any()


# ======================================================
# Error Metrics
# ======================================================

mae = np.mean(np.abs(predicted_reserves - true_reserves))

rmse = np.sqrt(
    np.mean(
        (predicted_reserves - true_reserves) ** 2
    )
)

print("\n")
print("=" * 60)
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print("=" * 60)


# ======================================================
# Plot
# ======================================================

report_dir = Path(config.paths.reports_dir)

report_dir.mkdir(parents=True, exist_ok=True)

plot_path = report_dir / "interest_rate_validation.png"

plt.figure(figsize=(8,5))

plt.plot(
    rates * 100,
    predicted_reserves,
    linewidth=3,
    label="PINN"
)

plt.plot(
    rates * 100,
    true_reserves,
    "--",
    linewidth=3,
    label="Thiele Solver"
)

plt.xlabel("Interest Rate (%)")
plt.ylabel("Reserve (£)")
plt.title("PINN vs Thiele Solver")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(plot_path, dpi=300)

print(f"\nPlot saved to {plot_path}")
