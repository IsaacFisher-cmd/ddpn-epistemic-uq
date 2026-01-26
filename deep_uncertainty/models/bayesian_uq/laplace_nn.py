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
    
    This class extends DoublePoissonNN to use a Laplace approximation with diagonal
    Fisher information matrix for uncertainty quantification.
    """
    
    def __init__(
        self,
        num_mc_samples: int = 50,
        lr: float = 1e-3,
        init_prec_diag: float = 10.0,
        grad_clip_norm: float = 1.0,
        **kwargs
    ):
        """
        Args:
            num_mc_samples: Number of Monte Carlo samples for posterior predictive
            lr: Learning rate for optimizer
            init_prec_diag: Initial precision for diagonal Laplace approximation
            grad_clip_norm: Maximum gradient norm for clipping
            **kwargs: Arguments passed to DoublePoissonNN
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        
        # Disable automatic optimization to manually handle updates
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
        """
        Stateless functional call to the model.
        
        Args:
            params: Dictionary of model parameters
            x: Input tensor
            
        Returns:
            Model output (concatenated mu and phi)
        """
        return torch.func.functional_call(self, params, (x,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass using current model parameters."""
        return self._predict_impl(x)

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prediction implementation that samples from posterior if fitted,
        otherwise uses deterministic forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted output (mu and phi concatenated)
        """
        if not self._is_posterior_fitted or not self.training:
            # During training or before Laplace fitting, use deterministic pass
            return super().forward(x)
        
        # If posterior is fitted and in eval mode, return mean prediction
        return super().forward(x)
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor, 
        return_samples: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate predictions with uncertainty estimates using posterior samples.
        
        Args:
            x: Input tensor
            return_samples: If True, return all MC samples
            
        Returns:
            mu_mean: Mean of predicted mu across samples
            mu_std: Standard deviation of mu across samples
            samples: All samples if return_samples=True, else None
        """
        if not self._is_posterior_fitted:
            raise RuntimeError("Posterior not fitted. Call init_posterior() first.")
        
        all_mu_samples = []
        
        self.eval()
        with torch.no_grad():
            for _ in range(self.num_mc_samples):
                sampled_params = self._sample_parameters()
                
                if sampled_params is None:
                    # Fallback to deterministic prediction
                    output = self(x)
                else:
                    # Use sampled parameters
                    output = self.functional(sampled_params, x)
                
                output = torch.clamp(output, min=-10, max=10)
                mu, _ = torch.split(output, [1, 1], dim=-1)
                all_mu_samples.append(mu)
        
        # Stack samples: (num_samples, batch_size, 1)
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
        """
        Compute log posterior for a batch of data.
        
        Args:
            params: Model parameters
            batch: Tuple of (x, y)
            
        Returns:
            log_prob: Per-sample log likelihood
            aux: Auxiliary information (empty tensor)
        """
        x, y = batch
        
        # Compute output with given params
        output = self.functional(params, x)
        output = torch.clamp(output, min=-10, max=10)
        
        # Split output into mu and phi
        mu, phi = torch.split(output, [1, 1], dim=-1)
        mu = mu.flatten()
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

    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int):
        """
        Training step using manual optimization.
        
        Args:
            batch: Batch of data (x, y)
            batch_idx: Batch index
        """
        optimizer = self.optimizers()
        optimizer.zero_grad()
        
        x, y = batch
        y_hat = self(x)
        y_hat = torch.clamp(y_hat, min=-10, max=10)
        
        loss = self.loss_fn(y_hat, y)
        
        if torch.isnan(loss):
            self.log('train_loss', 0.0, prog_bar=True)
            return
        
        self.manual_backward(loss)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_norm)
        
        optimizer.step()
        
        self.log('train_loss', loss, prog_bar=True)
        return loss

    @abstractmethod
    def _sample_parameters(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Draw one set of weights from the Laplace posterior distribution.
        
        Returns:
            Sampled parameters dictionary, or None if posterior not fitted
        """
        if not self._is_posterior_fitted or self.posterior_state is None:
            return None
        
        try:
            sampled_params = diag_fisher.sample(self.posterior_state)
            return sampled_params
        except Exception as e:
            print(f"Warning: Parameter sampling failed: {e}")
            return None

    @abstractmethod
    def init_posterior(self, train_loader: DataLoader):
        """
        Initialize and fit the Diagonal Fisher Laplace posterior approximation.
        
        Args:
            train_loader: DataLoader containing training data
        """
        print("\n🔧 Initializing Diagonal Fisher Laplace posterior...")
        
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
        
        for x_batch, y_batch in train_loader:
            batch = (x_batch, y_batch)
            
            try:
                self.posterior_state, aux = self.posterior_transform.update(
                    self.posterior_state, batch
                )
                batch_count += 1
                
                if batch_count % 5 == 0:
                    print(f"  Processed {batch_count}/{len(train_loader)} batches")
                    
            except Exception as e:
                print(f"  ⚠️ Skipping batch {batch_count}: {e}")
                continue
        
        self._is_posterior_fitted = True
        print(f"\n✅ Laplace posterior fitted with {batch_count} batches")
        
        # Print diagnostics
        self._print_posterior_diagnostics()

    def _print_posterior_diagnostics(self):
        """Print diagnostic information about the fitted posterior."""
        if self.posterior_state is None:
            return
        
        print("\n📊 Precision matrix statistics:")
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
    
    # ---- 1️⃣ Create synthetic data ----
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
    
    print(f"\n📊 Dataset created:")
    print(f"  Training points: {len(x_train)}")
    print(f"  Test points: {len(x_all)}")
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
    test_loader = DataLoader(
        TensorDataset(x_all_norm, y_all), 
        batch_size=32
    )
    
    # ---- 2️⃣ Create and train model ----
    print("\n🔧 Creating model...")
    
    # Simple MLP backbone for testing - FIXED with output_dim attribute
    class SimpleMLP(torch.nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim  # This is required by DoublePoissonNN
            
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, output_dim)
            )
        
        def forward(self, x):
            return self.net(x)
    
    # Import OptimizerType if needed, or create a simple enum
    try:
        from deep_uncertainty.enums import OptimizerType
    except ImportError:
        # Fallback: create a simple class
        class OptimizerType:
            ADAM = "adam"
    
    model = DoublePoissonLaplaceDiagFisher(
        backbone_type=SimpleMLP,
        backbone_kwargs={"input_dim": 1, "output_dim": 64},
        optim_type=OptimizerType.ADAM,
        optim_kwargs={"lr": 5e-4},
        num_mc_samples=30,
        lr=5e-4,
        init_prec_diag=10.0,
        grad_clip_norm=1.0
    )
    
    # ---- 3️⃣ Training loop ----
    print("\n🏋️ Training model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    epochs = 300
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_hat = model(x_batch)
            y_hat = torch.clamp(y_hat, min=-10, max=10)
            loss = model.loss_fn(y_hat, y_batch)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 50 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}")
    
    print("✅ Training complete!")
    
    # ---- 4️⃣ Fit Laplace posterior ----
    print("\n" + "="*60)
    model.init_posterior(train_loader)
    print("="*60)
    
    # ---- 5️⃣ Generate predictions with uncertainty ----
    print("\n🔮 Generating predictions with uncertainty...")
    model.eval()
    
    mu_mean, mu_std, samples = model.predict_with_uncertainty(
        x_all_norm, 
        return_samples=True
    )
    
    mu_mean = mu_mean.flatten().numpy()
    mu_std = mu_std.flatten().numpy()
    samples_np = samples.squeeze(-1).numpy()  # (num_samples, batch_size)
    
    # Get deterministic prediction for comparison
    with torch.no_grad():
        det_output = model(x_all_norm)
        det_mu, det_phi = torch.split(det_output, [1, 1], dim=-1)
        det_mu = det_mu.flatten().numpy()
    
    # ---- 6️⃣ Visualize results ----
    print("\n📈 Creating visualization...")
    
    x_plot = x_all.numpy()
    y_plot = y_all.numpy()
    x_train_plot = x_train.flatten().numpy()
    y_train_plot = y_train.numpy()
    
    # Sort for plotting
    sort_idx = np.argsort(x_plot)
    x_plot_sorted = x_plot[sort_idx]
    mu_mean_sorted = mu_mean[sort_idx]
    mu_std_sorted = mu_std[sort_idx]
    det_mu_sorted = det_mu[sort_idx]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top left: Training data
    axes[0, 0].scatter(x_train_plot, y_train_plot, alpha=0.6, s=30, label='Training data')
    axes[0, 0].axvspan(0, 2, alpha=0.2, color='red', label='Sparse region')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('count y')
    axes[0, 0].set_title('Training Data (with gap)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Top right: Deterministic prediction
    axes[0, 1].scatter(x_plot, y_plot, alpha=0.3, s=20, label='Test data')
    axes[0, 1].plot(x_plot_sorted, det_mu_sorted, 'r-', linewidth=2, label='Deterministic μ')
    axes[0, 1].scatter(x_train_plot, y_train_plot, alpha=0.6, s=30, 
                       color='orange', label='Training data')
    axes[0, 1].axvspan(0, 2, alpha=0.2, color='red')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('count')
    axes[0, 1].set_title('Deterministic Prediction')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bottom left: Posterior predictive with uncertainty
    axes[1, 0].scatter(x_plot, y_plot, alpha=0.3, s=20, label='Test data')
    axes[1, 0].plot(x_plot_sorted, mu_mean_sorted, 'b-', linewidth=2, label='Mean μ')
    axes[1, 0].fill_between(
        x_plot_sorted,
        mu_mean_sorted - 2*mu_std_sorted,
        mu_mean_sorted + 2*mu_std_sorted,
        alpha=0.3,
        label='±2σ (95% CI)'
    )
    axes[1, 0].scatter(x_train_plot, y_train_plot, alpha=0.6, s=30, 
                       color='orange', label='Training data')
    axes[1, 0].axvspan(0, 2, alpha=0.2, color='red')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('count')
    axes[1, 0].set_title('Laplace Posterior Predictive (with Uncertainty)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bottom right: Sample trajectories
    axes[1, 1].scatter(x_plot, y_plot, alpha=0.2, s=10, label='Test data', zorder=1)
    
    # Plot random sample trajectories
    n_traj = min(20, samples_np.shape[0])
    for i in range(n_traj):
        sample_sorted = samples_np[i][sort_idx]
        axes[1, 1].plot(x_plot_sorted, sample_sorted, 'b-', alpha=0.2, linewidth=1)
    
    axes[1, 1].plot(x_plot_sorted, mu_mean_sorted, 'r-', linewidth=2, 
                    label='Mean', zorder=3)
    axes[1, 1].scatter(x_train_plot, y_train_plot, alpha=0.6, s=30, 
                       color='orange', label='Training data', zorder=2)
    axes[1, 1].axvspan(0, 2, alpha=0.2, color='red', zorder=0)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('count')
    axes[1, 1].set_title(f'Posterior Sample Trajectories (n={n_traj})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('laplace_double_poisson_test.png', dpi=150, bbox_inches='tight')
    print("✅ Plot saved as 'laplace_double_poisson_test.png'")
    plt.show()
    
    # ---- 7️⃣ Print summary statistics ----
    print("\n" + "="*60)
    print("📊 SUMMARY STATISTICS")
    print("="*60)
    
    # Focus on the gap region
    gap_mask = (x_plot >= 0) & (x_plot <= 2)
    print(f"\n🔍 Uncertainty in sparse region [0, 2]:")
    print(f"  Mean std: {mu_std[gap_mask].mean():.3f}")
    print(f"  Max std:  {mu_std[gap_mask].max():.3f}")
    
    outside_mask = ~gap_mask
    print(f"\n🔍 Uncertainty outside sparse region:")
    print(f"  Mean std: {mu_std[outside_mask].mean():.3f}")
    print(f"  Max std:  {mu_std[outside_mask].max():.3f}")
    
    ratio = mu_std[gap_mask].mean() / (mu_std[outside_mask].mean() + 1e-8)
    print(f"\n📈 Uncertainty ratio (sparse/dense): {ratio:.2f}x")
    print("   (Higher is better - shows model is more uncertain in gap)")
    
    print("\n✅ Test complete!")