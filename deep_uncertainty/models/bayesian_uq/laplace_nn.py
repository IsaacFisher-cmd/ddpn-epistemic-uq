from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.func
from torch.utils.data import DataLoader
from posteriors.laplace import diag_fisher

from .. import DoublePoissonNN

class DoublePoissonLaplaceDiagFisher(DoublePoissonNN):
    """
    Double Poisson neural network using Diagonal Fisher Laplace posterior approximation.
    """

    def __init__(
        self,
        num_mc_samples: int = 50,
        lr: float = 1e-3,
        init_prec_diag: float = 1.0,  # REDUCED from 10.0 - less restrictive prior
        grad_clip_norm: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # Disable automatic optimization
        self.automatic_optimization = False

        self.num_mc_samples = num_mc_samples
        self.lr = lr
        self.init_prec_diag = init_prec_diag
        self.grad_clip_norm = grad_clip_norm

        # Posterior state and transform
        self.posterior_state = None
        self.posterior_transform = None
        self._is_posterior_fitted = False

    def functional(self, params: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        """Stateless functional call to the model."""
        return torch.func.functional_call(self, params, (x,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass using current model parameters."""
        return super().forward(x)

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Prediction implementation."""
        return super()._predict_impl(x)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        return_samples: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Generate predictions with uncertainty estimates using posterior samples."""
        if not self._is_posterior_fitted:
            raise RuntimeError("Posterior not fitted. Call init_posterior() first.")

        all_mu_samples = []
        all_phi_samples = []  # Also track phi for diagnostics

        self.eval()
        with torch.no_grad():
            for _ in range(self.num_mc_samples):
                sampled_params = self._sample_parameters()

                if sampled_params is None:
                    output = self(x)
                else:
                    output = self.functional(sampled_params, x)

                output = torch.clamp(output, min=-10, max=10)
                mu, phi = torch.split(output, [1, 1], dim=-1)
                all_mu_samples.append(mu)
                all_phi_samples.append(phi)

        # Stack samples
        samples = torch.stack(all_mu_samples, dim=0)

        # Compute statistics
        mu_mean = samples.mean(dim=0)
        mu_std = samples.std(dim=0)

        if return_samples:
            return mu_mean, mu_std, samples
        else:
            return mu_mean, mu_std, None

    def log_posterior(
        self,
        params: Dict[str, torch.Tensor],
        batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log posterior for a batch of data."""
        x, y = batch

        # Compute output with given params
        output = self.functional(params, x)
        output = torch.clamp(output, min=-10, max=10)

        # Split output into mu and phi
        mu, phi = torch.split(output, [1, 1], dim=-1)
        mu = torch.exp(mu.flatten())  # IMPORTANT: exp transform for mu > 0
        phi = torch.abs(phi.flatten()) + 1e-6  # Ensure positive
        y_flat = y.flatten()

        # Double Poisson log likelihood per sample
        eps = 1e-8

        term1 = y_flat * torch.log(mu + eps)
        term2 = -mu
        term3 = -0.5 * torch.log(2 * torch.pi * (y_flat + eps) * phi)
        term4 = -(y_flat - mu).pow(2) / (2 * mu * phi + eps)

        log_prob = term1 + term2 + term3 + term4

        return log_prob, torch.tensor([])

    def training_step(self, batch: Any, batch_idx: int):
        """Training step using manual optimization."""
        optimizer = self.optimizers()
        optimizer.zero_grad()

        x, y = batch
        y_hat = self(x)
        y_hat = torch.clamp(y_hat, min=-10, max=10)

        loss = self.loss_fn(y_hat, y)

        if torch.isnan(loss) or torch.isinf(loss):
            self.log('train_loss', 0.0, prog_bar=True)
            return

        self.manual_backward(loss)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_norm)

        optimizer.step()

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def _sample_parameters(self) -> Optional[Dict[str, torch.Tensor]]:
        """Draw one set of weights from the Laplace posterior distribution."""
        if not self._is_posterior_fitted or self.posterior_state is None:
            return None

        try:
            sampled_params = diag_fisher.sample(self.posterior_state)
            return sampled_params
        except Exception as e:
            print(f"Warning: Parameter sampling failed: {e}")
            return None

    def init_posterior(self, train_loader: DataLoader):
        """Initialize and fit the Diagonal Fisher Laplace posterior approximation."""
        print("\n:wrench: Initializing Diagonal Fisher Laplace posterior...")

        # Get current model parameters
        params = dict(self.named_parameters())

        # Build Laplace transform
        self.posterior_transform = diag_fisher.build(
            log_posterior=self.log_posterior,
            per_sample=True,
            init_prec_diag=self.init_prec_diag,
        )

        # Initialize state
        self.posterior_state = self.posterior_transform.init(params)

        # Fit posterior using training data
        self.eval()
        batch_count = 0
        successful_updates = 0

        for x_batch, y_batch in train_loader:
            batch = (x_batch, y_batch)

            try:
                self.posterior_state, aux = self.posterior_transform.update(
                    self.posterior_state, batch
                )
                batch_count += 1
                successful_updates += 1

                if batch_count % 5 == 0:
                    print(f"  Processed {batch_count}/{len(train_loader)} batches")

            except Exception as e:
                print(f"  :warning: Skipping batch {batch_count}: {e}")
                batch_count += 1
                continue

        if successful_updates == 0:
            print(":x: WARNING: No successful updates to posterior!")
            self._is_posterior_fitted = False
            return

        self._is_posterior_fitted = True
        print(f"\n:white_check_mark: Laplace posterior fitted with {successful_updates}/{batch_count} batches")

        # Print diagnostics
        self._print_posterior_diagnostics()

    def _print_posterior_diagnostics(self):
        """Print diagnostic information about the fitted posterior."""
        if self.posterior_state is None:
            return

        print("\n:bar_chart: Precision matrix statistics:")
        total_params = 0

        for name, prec in self.posterior_state.prec_diag.items():
            n_params = prec.numel()
            total_params += n_params
            print(f"  {name} ({n_params} params): "
                  f"min={prec.min().item():.2e}, "
                  f"max={prec.max().item():.2e}, "
                  f"mean={prec.mean().item():.2e}")

        print(f"\nTotal parameters: {total_params}")

        mean_prec = sum(
            prec.mean().item()
            for prec in self.posterior_state.prec_diag.values()
        ) / len(self.posterior_state.prec_diag)
        print(f"Average precision: {mean_prec:.2e}")

    def configure_optimizers(self):
        """Configure optimizer for manual optimization."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np

    print("="*60)
    print("Testing DoublePoissonLaplaceDiagFisher")
    print("="*60)

    # ---- :one: Create synthetic data ----
    torch.manual_seed(42)
    n_points = 400

    x_all = torch.rand(n_points) * 10 - 3  # range [-3, 7]

    # Create sparse training data (gap between 0 and 2)
    mask_outside = (x_all < 0) | (x_all > 2)
    x_train_outside = x_all[mask_outside]

    mask_inside = (x_all >= 0) & (x_all <= 2)
    x_train_inside = x_all[mask_inside][::5]

    x_train = torch.cat([x_train_outside, x_train_inside]).unsqueeze(1)

    # Generate Poisson counts
    true_log_lambda_fn = lambda x: (x * 0.3) + (1.5 * torch.cos(x * 1.5)) + 1.0

    lambda_x_all = torch.exp(true_log_lambda_fn(x_all))
    lambda_x_train = torch.exp(true_log_lambda_fn(x_train))

    y_all = torch.poisson(lambda_x_all)
    y_train = torch.poisson(lambda_x_train)

    print(f"\n:bar_chart: Dataset created:")
    print(f"  Training points: {len(x_train)}")
    print(f"  Test points: {len(x_all)}")
    print(f"  Y range: [{y_train.min():.1f}, {y_train.max():.1f}]")
    print(f"  Gap region: [0, 2] (sparse training data)")

    # Normalize
    x_mean, x_std = x_train.mean(), x_train.std()
    x_train_norm = (x_train - x_mean) / x_std
    x_all_norm = ((x_all - x_mean) / x_std).unsqueeze(1)

    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(x_train_norm, y_train),
        batch_size=32,
        shuffle=True
    )

    # ---- :two: Create and train model ----
    print("\n:wrench: Creating model...")

    # Simple MLP backbone for testing
    class SimpleMLP(torch.nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim

            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, output_dim)
            )

        def forward(self, x):
            return self.net(x)

    # Import OptimizerType
    try:
        from deep_uncertainty.enums import OptimizerType
    except ImportError:
        class OptimizerType:
            ADAM = "adam"

    model = DoublePoissonLaplaceDiagFisher(
        backbone_type=SimpleMLP,
        backbone_kwargs={"input_dim": 1, "output_dim": 128},
        optim_type=OptimizerType.ADAM,
        optim_kwargs={"lr": 1e-3},
        num_mc_samples=50,
        lr=1e-3,
        init_prec_diag=1.0,
        grad_clip_norm=1.0
    )

    # ---- :three: Training loop ----
    print("\n:weight_lifter: Training model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 500

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_hat = model(x_batch)
            y_hat = torch.clamp(y_hat, min=-10, max=10)
            loss = model.loss_fn(y_hat, y_batch)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if n_batches > 0 and epoch % 100 == 0:
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}")

    print(":white_check_mark: Training complete!")

    # ---- :four: Fit Laplace posterior ----
    print("\n" + "="*60)
    model.init_posterior(train_loader)
    print("="*60)

    if not model._is_posterior_fitted:
        print(":x: Posterior fitting failed! Exiting...")
        exit(1)

    # ---- :five: Get deterministic prediction (MAP estimate) ----
    print("\n:crystal_ball: Generating predictions...")
    model.eval()

    with torch.no_grad():
        det_output = model(x_all_norm)
        det_mu, det_phi = torch.split(det_output, [1, 1], dim=-1)
        det_mu = det_mu.flatten().numpy()

    print(f"  Deterministic (MAP) prediction range: [{det_mu.min():.2f}, {det_mu.max():.2f}]")

    # ---- :six: Generate predictions with Laplace uncertainty ----
    mu_mean, mu_std, samples = model.predict_with_uncertainty(
        x_all_norm,
        return_samples=True
    )

    mu_mean = mu_mean.flatten().numpy()
    mu_std = mu_std.flatten().numpy()
    samples_np = samples.squeeze(-1).numpy()

    print(f"  Posterior mean range: [{mu_mean.min():.2f}, {mu_mean.max():.2f}]")
    print(f"  Posterior std range: [{mu_std.min():.4f}, {mu_std.max():.4f}]")

    # ---- :seven: Create single visualization ----
    print("\n:chart_with_upwards_trend: Creating visualization...")

    x_plot = x_all.numpy()
    y_plot = y_all.numpy()
    x_train_plot = x_train.flatten().numpy()
    y_train_plot = y_train.numpy()

    # Sort for plotting
    sort_idx = np.argsort(x_plot)
    x_plot_sorted = x_plot[sort_idx]
    det_mu_sorted = det_mu[sort_idx]
    mu_mean_sorted = mu_mean[sort_idx]
    mu_std_sorted = mu_std[sort_idx]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot test data (light)
    ax.scatter(x_plot, y_plot, alpha=0.2, s=15, color='lightblue',
               label='Test data', zorder=1)

    # Plot training data (prominent)
    ax.scatter(x_train_plot, y_train_plot, alpha=0.7, s=40,
               color='orange', edgecolors='darkorange', linewidth=0.5,
               label='Training data', zorder=3)

    # Plot deterministic prediction (MAP estimate)
    # ax.plot(x_plot_sorted, det_mu_sorted, 'r-', linewidth=2.5, label='MAP prediction (μ)', zorder=4)

    # Plot Laplace posterior mean
    ax.plot(x_plot_sorted, mu_mean_sorted * det_mu_sorted, 'b-', linewidth=2,
            label='Posterior mean', zorder=5, alpha=0.8)

    # Plot uncertainty bands (±2σ for 95% CI)
    ax.fill_between(
        x_plot_sorted,
        mu_mean_sorted * det_mu_sorted - mu_std_sorted,
        mu_mean_sorted * det_mu_sorted + mu_std_sorted,
        alpha=0.25,
        color='blue',
        label='±2σ (95% CI)',
        zorder=2
    )

    # Highlight sparse region
    ax.axvspan(0, 2, alpha=0.15, color='red', label='Sparse region', zorder=0)

    # Formatting
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('count', fontsize=12)
    ax.set_title('Double Poisson with Laplace Posterior (Diagonal Fisher)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('laplace_double_poisson_result.png', dpi=150, bbox_inches='tight')
    print(":white_check_mark: Plot saved as 'laplace_double_poisson_result.png'")
    plt.show()

    # ---- :eight: Print summary statistics ----
    print("\n" + "="*60)
    print(":bar_chart: SUMMARY STATISTICS")
    print("="*60)

    # Overall statistics
    print(f"\n:chart_with_upwards_trend: Overall:")
    print(f"  MAP prediction: μ ∈ [{det_mu.min():.2f}, {det_mu.max():.2f}]")
    print(f"  Posterior mean: μ ∈ [{mu_mean.min():.2f}, {mu_mean.max():.2f}]")
    print(f"  Posterior std:  σ ∈ [{mu_std.min():.4f}, {mu_std.max():.4f}]")

    # Focus on the gap region
    gap_mask = (x_plot >= 0) & (x_plot <= 2)
    outside_mask = ~gap_mask

    print(f"\n:mag: Uncertainty in SPARSE region [0, 2]:")
    print(f"  Mean std: {mu_std[gap_mask].mean():.3f}")
    print(f"  Max std:  {mu_std[gap_mask].max():.3f}")
    print(f"  Mean prediction: {mu_mean[gap_mask].mean():.2f}")

    print(f"\n:mag: Uncertainty in DENSE region (outside gap):")
    print(f"  Mean std: {mu_std[outside_mask].mean():.3f}")
    print(f"  Max std:  {mu_std[outside_mask].max():.3f}")
    print(f"  Mean prediction: {mu_mean[outside_mask].mean():.2f}")

    ratio = mu_std[gap_mask].mean() / (mu_std[outside_mask].mean() + 1e-8)
    print(f"\n:bar_chart: Uncertainty ratio (sparse/dense): {ratio:.2f}x")
    if ratio > 1.5:
        print("   :white_check_mark: GOOD: Model shows higher uncertainty in sparse region!")
    elif ratio > 1.0:
        print("   :warning:  OK: Slight increase in uncertainty in sparse region")
    else:
        print("   :x: POOR: Model not capturing epistemic uncertainty properly")

    print("\n:white_check_mark: Test complete!")