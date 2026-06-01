"""
Diffusion utility functions: noise scheduling, sampling, etc.
"""

import torch
import torch.nn as nn
import numpy as np


class DDPMScheduler:
    """
    DDPM-style noise scheduler.
    Handles alpha (cumulative product of betas) and variance schedules.
    """

    def __init__(self, n_steps=1000, beta_start=0.0001, beta_end=0.02):
        """
        Args:
            n_steps: Total number of diffusion steps
            beta_start: Starting beta value
            beta_end: Ending beta value
        """
        self.n_steps = n_steps

        # Linear schedule for betas
        self.betas = torch.linspace(beta_start, beta_end, n_steps)

        # Cumulative products
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])

        # Useful quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)

    def to(self, device):
        """Move scheduler to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(
            device
        )
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        return self

    def add_noise(self, x_0, t, noise=None):
        """
        Add noise to clean sample at timestep t.
        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * eps

        Args:
            x_0: (batch_size, C, H, W) clean samples
            t: (batch_size,) timesteps in [0, n_steps)
            noise: (batch_size, C, H, W) noise or None to sample
        Returns:
            x_t: (batch_size, C, H, W) noisy samples
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Index into schedule
        t = t.long().clamp(0, self.n_steps - 1)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(
            -1, 1, 1, 1
        )

        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t

    @torch.no_grad()
    def sample(
        self,
        model,
        shape,
        spectrum_tokens=None,
        device="cuda",
        guidance_scale=0.0,
        guidance_fn=None,
        guidance_start_step=None,
        guidance_end_step=None,
        guidance_num_applications=None,
    ):
        """
        Reverse diffusion sampling (DDPM).

        Args:
            model: Denoiser model
            shape: (batch_size, C, H, W)
            spectrum_tokens: (batch_size, n_tokens, spectrum_token_dim) or None
            device: torch device
            guidance_scale: Scale for guidance (0 = no guidance)
            guidance_fn: Function(x_t, t) -> guidance signal
            guidance_start_step: Step to start guidance
            guidance_end_step: Optional step to stop guidance (inclusive)
            guidance_num_applications: Optional number of guidance applications
        Returns:
            x_0: (batch_size, C, H, W) generated samples
        """
        batch_size = shape[0]
        x_t = torch.randn(shape, device=device)

        if guidance_start_step is None:
            guidance_start_step = int(self.n_steps * 0.35)
        if guidance_end_step is None:
            guidance_end_step = 0

        guidance_steps = None
        if guidance_scale > 0 and guidance_fn is not None and guidance_num_applications:
            if guidance_num_applications <= 0:
                guidance_steps = set()
            else:
                guidance_steps = set(
                    int(step)
                    for step in torch.linspace(
                        float(guidance_start_step),
                        float(guidance_end_step),
                        steps=int(guidance_num_applications),
                    ).round().long().tolist()
                )

        for step in range(self.n_steps - 1, -1, -1):
            t = torch.full((batch_size,), step / self.n_steps, device=device)

            # Predict noise
            noise_pred = model(x_t, t, spectrum_tokens)

            # Apply guidance if enabled
            if guidance_scale > 0 and guidance_fn is not None:
                if guidance_steps is not None:
                    should_apply_guidance = step in guidance_steps
                else:
                    should_apply_guidance = step <= guidance_start_step and step >= guidance_end_step

                if should_apply_guidance:
                    with torch.enable_grad():
                        guidance = guidance_fn(x_t, t)
                    noise_pred = noise_pred + guidance_scale * guidance

            # DDPM update step
            alpha_t = self.alphas[step]
            alpha_cumprod_t = self.alphas_cumprod[step]
            alpha_cumprod_prev_t = self.alphas_cumprod_prev[step]

            # Variance
            variance = (
                (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t) * (1 - alpha_t)
            )
            variance = variance.clamp(min=1e-20)

            # Mean
            coeff1 = torch.sqrt(1 / alpha_t)
            coeff2 = (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)

            x_t_mean = coeff1 * (x_t - coeff2 * noise_pred)

            # Add noise (except on last step)
            if step > 0:
                noise = torch.randn_like(x_t)
                x_t = x_t_mean + torch.sqrt(variance) * noise
            else:
                x_t = x_t_mean

        return x_t


class GuidanceHelper:
    """Helper for surrogate residual guidance during diffusion sampling."""

    def __init__(self, forward_predictor, target_spectrum, lambda_guidance=1.0):
        """
        Args:
            forward_predictor: Forward model (geometry -> spectrum)
            target_spectrum: (batch_size, n_spectrum_points) target spectrum
            lambda_guidance: Guidance strength
        """
        self.forward_predictor = forward_predictor
        self.target_spectrum = target_spectrum
        self.lambda_guidance = lambda_guidance

    def compute_guidance(self, x_t, t, decoder=None):
        """
        Compute residual guidance: residual between predicted and target spectrum.

        Args:
            x_t: (batch, C, H, W) current latent at step t
            t: (batch,) time step
            decoder: Decoder to reconstruct geometry from latent
        Returns:
            guidance: Guidance signal (shape compatible with model output)
        """
        latent = x_t.detach().clone().requires_grad_(True)

        if decoder is not None:
            x_recon = decoder(latent)
        else:
            x_recon = latent

        spectrum_pred = self.forward_predictor(x_recon)
        target = self.target_spectrum.to(spectrum_pred.device)
        if target.dim() == 1:
            target = target.unsqueeze(0)
        if target.shape[0] != spectrum_pred.shape[0]:
            target = target.expand(spectrum_pred.shape[0], -1)

        residual_loss = torch.mean((spectrum_pred - target) ** 2)
        guidance = torch.autograd.grad(
            residual_loss,
            latent,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]

        guidance = -self.lambda_guidance * guidance
        guidance_norm = guidance.flatten(1).norm(dim=1, keepdim=True).clamp(min=1e-6)
        guidance = guidance / guidance_norm.view(-1, 1, 1, 1)

        return guidance
