import torch


class ControlledSde:
    """
    Controlled Stochastic Differential Equation
    """
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def evolve(self, dt: float, state: torch.tensor, action: torch.tensor, mean_field: torch.distributions) -> torch.tensor:
        """
        Evolves the SDE forward with step size `dt` using Euler-Maruyama method
        """
        dt = torch.tensor(dt)
        dW = torch.normal(0.0, torch.sqrt(dt))
        return state + self.mu(state, action, mean_field) * dt + self.sigma(state, action, mean_field) * dW
