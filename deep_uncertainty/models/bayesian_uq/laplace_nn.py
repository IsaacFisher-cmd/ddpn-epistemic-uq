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
        init_prec_diag: float = 1.0,
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