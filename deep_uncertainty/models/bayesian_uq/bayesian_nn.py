from abc import abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.func
from deep_uncertainty.models.double_poisson_nn import DoublePoissonNN

class DoublePoissonBayesianNN(DoublePoissonNN):
    """
    Abstract base class for Bayesian Double Poissonn neural networks.
    """
    def __init__(
        self,
        num_mc_samples: int = 10,
        lr: float = 1e-3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        
        # 1. Disable automatic optimization to let 'posteriors' handle updates
        self.automatic_optimization = False
        
        self.num_mc_samples = num_mc_samples
        self.lr = lr
        
        # This will hold the posterior distribution state (e.g., mean/variance)
        self.posterior_state = None
        self.posterior_transform = None

    def functional(self, params: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        """A stateless functional call to the model, required by 'posteriors'."""
        return torch.func.functional_call(self, params, (x,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._predict_impl(x)

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bayesian prediction using Monte Carlo averaging over 'num_mc_samples'.
        """
        self.eval()
        outputs = []

        with torch.no_grad():
            for _ in range(self.num_mc_samples):
                params = self._sample_parameters()
                
                if params is not None:
                    # Functional call using sampled weights
                    y_hat = self.functional(params, x)
                else:
                    # Fallback for methods like MC Dropout
                    y_hat = self.backbone(x)
                
                # Double Poisson outputs are log(mu) and log(phi)
                outputs.append(torch.exp(y_hat))

        # Average the parameters (mu, phi) across samples
        # Shape: (Samples, Batch, 2) -> (Batch, 2)
        stacked_outputs = torch.stack(outputs)
        mean_prediction = stacked_outputs.mean(dim=0)
        
        return mean_prediction

    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int):
        """
        Pass in the current state (params, aux) to the Posteriors.
        Should update self.posterior_transform
        """
        pass

    @abstractmethod
    def _sample_parameters(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Draw one set of weights from the posterior distribution.
        """
        pass

    @abstractmethod
    def init_posterior(self, train_loader):
        """
        Subclasses must define the 'posteriors' transform (VI, Laplace, etc.)
        and initialize self.posterior_state here.
        """
        pass

    def configure_optimizers(self):
        """
        In manual optimization, we still return an optimizer for Lightning 
        internals, even if the 'posteriors' transform wraps it.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    