import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch.utils.data import TensorDataset, DataLoader
from .. import DoublePoissonNN
from ..backbones import MLP
from ...enums import OptimizerType


# ---- 1️⃣ Load data ----
torch.manual_seed(42)

n_points = 400

x_all = torch.rand(n_points) * 10 - 3  # range [-3, 7]


mask_outside = (x_all < 0) | (x_all > 2)
x_train_outside = x_all[mask_outside]

mask_inside = (x_all >= 0) & (x_all <= 2)
x_train_inside = x_all[mask_inside][::5]


x_train = torch.cat([x_train_outside, x_train_inside]).unsqueeze(1)

lambda_x_all = torch.exp(0.7 * x_all - 0.05 * x_all**2 + 1.0)
lambda_x_train = torch.exp(0.7 * x_train - 0.05 * x_train**2 + 1.0)

y_all = torch.poisson(lambda_x_all)
y_train = torch.poisson(lambda_x_train)


fig, axes = plt.subplots(1,2, figsize=(14, 4))
axes[0].scatter(x_train.numpy(), y_train.numpy(), alpha=0.6)
axes[0].set_xlabel('x')
axes[0].set_ylabel('count y')
axes[0].set_title('Training Data')

axes[1].scatter(x_all.numpy(), y_all.numpy(), alpha=0.6)
axes[1].set_xlabel('x')
axes[1].set_ylabel('count')
axes[1].set_title('All Data')
fig.show()

# Optional but highly recommended: normalize X
x_mean, x_std = x_train.mean(), x_train.std()
x_train = (x_train - x_mean) / x_std
x_all = ((x_all - x_mean) / x_std).unsqueeze(1)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=8, shuffle=True)
test_loader = DataLoader(TensorDataset(x_all, y_all), batch_size=8)

# ---- 2️⃣ Define model ----
model = DoublePoissonNN(
    backbone_type=MLP,  # use your provided MLP
    backbone_kwargs={"input_dim": 1, "output_dim": 64},
    optim_type=OptimizerType.ADAM,
    optim_kwargs={"lr": 1e-4},  # small learning rate = more stable
)

# ---- 3️⃣ Training loop ----
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)  # smaller lr
epochs = 500

for epoch in range(epochs):
    total_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_hat = model(x_batch)
        y_hat = torch.clamp(y_hat, min=-10, max=10)  # prevents exp blowup
        loss = model.loss_fn(y_hat, y_batch)
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss={total_loss/len(train_loader):.4f}")
# ---- 4️⃣ Evaluate ----
model.eval()
with torch.no_grad():
    preds = model._predict_impl(x_all)
    mu, phi = torch.split(preds, [1, 1], dim=-1)
print("Sample predictions (μ):", mu[:5].flatten())

# Make predictions on the test set
model.eval()
with torch.no_grad():
    preds = model._predict_impl(x_all)
    # print(preds)
    mu, phi = torch.split(preds, [1, 1], dim=-1)
    mu = mu.flatten().numpy()
    phi = phi.flatten().numpy()
    sigma = np.sqrt(mu*phi) # standard deviation

x_vals = x_all.flatten().numpy()
y_vals = y_all.flatten().numpy()

# Sort by x for plotting a nice curve
sorted_idx = np.argsort(x_vals)
x_vals = x_vals[sorted_idx]
mu = mu[sorted_idx]
sigma = sigma[sorted_idx]
y_vals = y_vals[sorted_idx]

# Plot predicted mean and uncertainty
plt.figure(figsize=(8, 5))
plt.plot(x_vals, mu, color='blue', label='Predicted mean (μ)')
plt.fill_between(
    x_vals,
    mu - sigma,
    mu + sigma,
    color='blue',
    alpha=0.2,
    label='Predicted ±1σ'
)
# Overlay true test data
plt.scatter(x_vals, y_vals, color='red', s=20, alpha=0.6, label='Test data')
plt.xlabel("x")
plt.ylabel("y")
plt.title("DoublePoissonNN predictions with uncertainty")
plt.legend()
plt.show()