from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
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


# ------------------------------------------------------------
# Load configuration
# ------------------------------------------------------------

config = ConfigLoader.load(Path("configs/config.yaml"))
ensure_directories(config)
set_seed(config.seed)

device_manager = DeviceManager(
    preferred_device=config.trainer.device,
    prefer_mixed_precision=False,
)

# ------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------

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

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device_manager.device)
model.eval()

# ------------------------------------------------------------
# Get dataset statistics
# ------------------------------------------------------------

train_dataset, _, _, _ = build_datasets(config)

target_mean = train_dataset.target_mean
target_std = train_dataset.target_std

# ------------------------------------------------------------
# Simulator and solver
# ------------------------------------------------------------

simulator = build_simulator(config)
solver = build_solver(config)

# ------------------------------------------------------------
# Fixed policy characteristics
# ------------------------------------------------------------

AGE = 40
TERM = 20
SUM_ASSURED = 250000

rates = np.linspace(0.01, 0.08, 30)

predicted_reserves = []
true_reserves = []

print(f"Age          : {AGE}")
print(f"Term         : {TERM}")
print(f"Sum Assured  : {SUM_ASSURED}")

# ------------------------------------------------------------
# Compare PINN and Thiele across interest rates
# ------------------------------------------------------------

for r in rates:

    # Build a NEW actuarially-consistent policy
    current_policy = simulator._build_policy(
        policy_id="debug_policy",
        age=AGE,
        term=TERM,
        interest_rate=float(r),
        sum_assured=SUM_ASSURED,
    )

    # --------------------------------------------------------
    # Build feature vector
    # --------------------------------------------------------

    features = torch.tensor(
        [[
            0.0,
            float(current_policy.age),
            current_policy.interest_rate,
            current_policy.premium,
            current_policy.sum_assured,
            current_policy.mortality_profile.intensity_at(0.0),
        ]],
        dtype=torch.float32,
    )

    # Normalize exactly like training
    features[:, 0] /= FEATURE_SCALES["time"]
    features[:, 1] /= FEATURE_SCALES["age"]
    features[:, 2] /= FEATURE_SCALES["interest_rate"]
    features[:, 3] /= FEATURE_SCALES["premium"]
    features[:, 4] /= FEATURE_SCALES["sum_assured"]
    features[:, 5] /= FEATURE_SCALES["mortality"]

    features = features.to(device_manager.device)

    # --------------------------------------------------------
    # PINN prediction
    # --------------------------------------------------------

    with torch.no_grad():

        z = model(features).item()

    reserve = (
        z * target_std
        + target_mean
    ) * current_policy.sum_assured

    predicted_reserves.append(reserve)

    # --------------------------------------------------------
    # Classical Thiele reserve
    # --------------------------------------------------------

    trajectory = solver.solve(
        current_policy,
        config.data.time_steps,
    )

    true_reserves.append(
        trajectory.reserves[0]
    )

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(9,6))

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
    label="Classical Thiele"
)

plt.xlabel("Interest Rate (%)")
plt.ylabel("Reserve (£)")
plt.title("Reserve vs Interest Rate")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Error statistics
# ------------------------------------------------------------

predicted_reserves = np.array(predicted_reserves)
true_reserves = np.array(true_reserves)

print()
print("=" * 60)
print("Comparison Statistics")
print("=" * 60)

rmse = np.sqrt(np.mean((predicted_reserves - true_reserves) ** 2))
mae = np.mean(np.abs(predicted_reserves - true_reserves))

print(f"RMSE : {rmse:,.2f}")
print(f"MAE  : {mae:,.2f}")

print()
print("Maximum Absolute Error:")
print(f"{np.max(np.abs(predicted_reserves-true_reserves)):,.2f}")

print()
print("Average Relative Error:")
print(
    f"{100*np.mean(np.abs(predicted_reserves-true_reserves)/true_reserves):.2f}%"
)