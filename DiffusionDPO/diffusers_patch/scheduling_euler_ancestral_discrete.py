# https://github.com/huggingface/diffusers/blob/cc7d88f247a70018366390359f84cb27a9546b64/src/diffusers/schedulers/scheduling_euler_ancestral_discrete.py#L165

# Copyright 2024 Katherine Crowson and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->EulerAncestralDiscrete
class EulerAncestralDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
        logprob (`torch.Tensor`):
            Log probability of the ancestral sampling step transition.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None
    logprob: Optional[torch.Tensor] = None

def step_with_logprob(
    self: EulerAncestralDiscreteScheduler,
    model_output: torch.Tensor,
    timestep: Union[float, torch.Tensor],
    sample: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    return_dict: bool = True,
) -> Union[EulerAncestralDiscreteSchedulerOutput, Tuple]:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.Tensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.Tensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        return_dict (`bool`):
            Whether or not to return a
            [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or tuple.

    Returns:
        [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or `tuple`:
            If return_dict is `True`,
            [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] is returned,
            otherwise a tuple is returned where the first element is the sample tensor, second is the predicted
            original sample, and third is the log probability of the ancestral sampling step.

    """

    if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
        raise ValueError(
            (
                "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                " one of the `scheduler.timesteps` as a timestep."
            ),
        )

    if not self.is_scale_input_called:
        logger.warning(
            "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
            "See `StableDiffusionPipeline` for a usage example."
        )

    if self.step_index is None:
        self._init_step_index(timestep)

    sigma = self.sigmas[self.step_index]

    # Upcast to avoid precision issues when computing prev_sample
    sample = sample.to(torch.float32)

    # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
    if self.config.prediction_type == "epsilon":
        pred_original_sample = sample - sigma * model_output
    elif self.config.prediction_type == "v_prediction":
        # * c_out + input * c_skip
        pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
    elif self.config.prediction_type == "sample":
        raise NotImplementedError("prediction_type not implemented yet: sample")
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
        )

    sigma_from = self.sigmas[self.step_index]
    sigma_to = self.sigmas[self.step_index + 1]
    sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

    # 2. Convert to an ODE derivative
    derivative = (sample - pred_original_sample) / sigma

    dt = sigma_down - sigma

    prev_sample = sample + derivative * dt

    device = model_output.device
    noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=device, generator=generator)

    # Calculate logprob for the ancestral sampling step
    # The noise follows a multivariate Gaussian: N(0, sigma_up^2 * I)
    # Log probability density: -0.5 * ||noise||^2 / sigma_up^2 - 0.5 * D * log(2π * sigma_up^2)
    if sigma_up > 0:
        # Calculate the squared L2 norm of noise
        noise_norm_sq = torch.sum(noise ** 2, dim=list(range(1, noise.ndim)), keepdim=True)
        # Number of dimensions (total elements per sample)
        D = torch.tensor(noise.numel() // noise.shape[0], dtype=noise.dtype, device=device)
        # Log probability density
        logprob = -0.5 * noise_norm_sq / (sigma_up ** 2) - 0.5 * D * torch.log(2 * math.pi * sigma_up ** 2)
        # Sum over spatial dimensions to get logprob per sample
        logprob = torch.sum(logprob, dim=list(range(1, logprob.ndim)))
    else:
        # Deterministic case (no noise added)
        # probability=1, log(1) = 0
        logprob = torch.zeros(noise.shape[0], dtype=noise.dtype, device=device)

    prev_sample = prev_sample + noise * sigma_up

    # Cast sample back to model compatible dtype
    prev_sample = prev_sample.to(model_output.dtype)

    # upon completion increase step index by one
    self._step_index += 1

    if not return_dict:
        return (
            prev_sample,
            pred_original_sample,
            logprob,
        )

    return EulerAncestralDiscreteSchedulerOutput(
        prev_sample=prev_sample, pred_original_sample=pred_original_sample, logprob=logprob
    )
