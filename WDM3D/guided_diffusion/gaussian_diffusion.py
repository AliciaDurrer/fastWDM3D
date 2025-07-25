"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
from PIL import Image
from torch.autograd import Variable
import enum
import torch.nn.functional as F
from torchvision.utils import save_image
import torch
import math
import numpy as np
import torch as th
from .train_util import visualize
from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from scipy import ndimage
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from guided_diffusion import dist_util
import nibabel as nib
from . import logger

from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

dwt = DWT_3D("haar")
idwt = IDWT_3D("haar")

def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * \
        (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def get_sigma_schedule(beta_min, beta_max, n_timestep):
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    sigmas = betas**0.5
    a_s = torch.sqrt(1 - betas)
    return betas

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, beta_min, beta_max):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1 #1000 / num_diffusion_timesteps
        beta_start = scale * beta_min
        beta_end = scale * beta_max
        logger.log('beta start and end', beta_start, beta_end)
        return get_sigma_schedule(
            beta_start, beta_end, num_diffusion_timesteps)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        mode="default",
        loss_level="image",
    ):
        self.model_mean_type = model_mean_type
        print("self.model_meantype", self.model_mean_type)
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.mode = mode
        self.loss_level = loss_level

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        print(betas)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])
        print("self.num_timesteps", self.num_timesteps)

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)  # t
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])  # t-1
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)  # t+1
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        logger.log('self.betas', self.betas)
        logger.log('self.alphas', alphas)
        logger.log('self.sqrt_alphas_cumprod', self.sqrt_alphas_cumprod)
        logger.log('self.sqrt_one_minus_alphas_cumprod', self.sqrt_one_minus_alphas_cumprod)
        logger.log('self.log_one_minus_alphas_cumprod', self.log_one_minus_alphas_cumprod)
        logger.log('self.sqrt_recip_alphas_cumprod', self.sqrt_recip_alphas_cumprod)
        logger.log('self.sqrt_recipm1_alphas_cumprod', self.sqrt_recipm1_alphas_cumprod)
        logger.log('self.posterior_variance', self.posterior_variance)
        logger.log('self.posterior_log_variance_clipped', self.posterior_log_variance_clipped)
        logger.log('self.posterior_mean_coef1', self.posterior_mean_coef1)
        logger.log('self.posterior_mean_coef2', self.posterior_mean_coef2)

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        #print(self.sqrt_alphas_cumprod.shape, t, x_start.shape, self.sqrt_alphas_cumprod.shape, noise.shape)
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """

        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self,
        model,
        x,
        t,
        input_masked,
        mask,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]

        assert t.shape == (B,)

        x_cat = torch.cat((x, input_masked, mask), dim=1)
        model_output = model(x_cat, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                B, _, H, W, D = x.size()
                x *= np.sqrt(8.0)
                x_idwt = idwt(
                    x[:, 0, :, :, :].view(B, 1, H, W, D),
                    x[:, 1, :, :, :].view(B, 1, H, W, D),
                    x[:, 2, :, :, :].view(B, 1, H, W, D),
                    x[:, 3, :, :, :].view(B, 1, H, W, D),
                    x[:, 4, :, :, :].view(B, 1, H, W, D),
                    x[:, 5, :, :, :].view(B, 1, H, W, D),
                    x[:, 6, :, :, :].view(B, 1, H, W, D),
                    x[:, 7, :, :, :].view(B, 1, H, W, D),
                )

                x_idwt_clamp = x_idwt.clamp(-1, 1)

                LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(x_idwt_clamp)
                x = th.cat([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
                x /= np.sqrt(8.0)

                return x
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )

            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        if self.mode == "segmentation":
            x_t = x_t[:, -pred_xstart.shape[1] :, ...]
        assert pred_xstart.shape == x_t.shape
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return eps

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            print("tresc", t)
            print("self.num_timesteps222", self.num_timesteps)
            print("scaledtimsetsep", t.float() * (1000.0 / self.num_timesteps))
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, update=None, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """

        if update is not None:
            print("CONDITION MEAN UPDATE NOT NONE")

            new_mean = (
                p_mean_var["mean"].detach().float()
                + p_mean_var["variance"].detach() * update.float()
            )
            a = update

        else:
            a, gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
            new_mean = (
                p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
            )

        return a, new_mean

    def condition_score2(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.
        See condition_mean() for details on cond_fn.
        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        t = t.long()
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        a, cfn = cond_fn(x, self._scale_timesteps(t).long(), **model_kwargs)
        eps = eps - (1 - alpha_bar).sqrt() * cfn

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out, cfn

    def sample_known(self, img, batch_size=1):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop_known(
            model, (batch_size, channels, image_size, image_size), img
        )

    def p_sample_loop(
        self,
        model,
        shape,
        t,
        noise=None,
        input_masked=None,
        mask=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            time=t,
            noise=noise,
            input_masked=input_masked,
            mask=mask,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample(
        self,
        model,
        x,
        t,
        input_masked=None,
        mask=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            input_masked=input_masked,
            mask=mask,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop_known(
        self,
        model,
        shape,
        img,
        org=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        noise_level=500,
        progress=False,
        classifier=None,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        b = shape[0]

        t = th.randint(499, 500, (b,), device=device).long().to(device)

        org = img[0].to(device)
        img = img[0].to(device)
        indices = list(range(t))[::-1]
        noise = th.randn_like(img[:, :4, ...]).to(device)
        x_noisy = self.q_sample(x_start=img[:, :4, ...], t=t, noise=noise).to(device)
        x_noisy = torch.cat((x_noisy, img[:, 4:, ...]), dim=1)

        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            time=noise_level,
            noise=x_noisy,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            org=org,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            classifier=classifier,
        ):
            final = sample

        return final["sample"], x_noisy, img

    def p_sample_loop_interpolation(
        self,
        model,
        shape,
        img1,
        img2,
        lambdaint,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        b = shape[0]
        t = th.randint(299, 300, (b,), device=device).long().to(device)
        img1 = torch.tensor(img1).to(device)
        img2 = torch.tensor(img2).to(device)
        noise = th.randn_like(img1).to(device)
        x_noisy1 = self.q_sample(x_start=img1, t=t, noise=noise).to(device)
        x_noisy2 = self.q_sample(x_start=img2, t=t, noise=noise).to(device)
        interpol = lambdaint * x_noisy1 + (1 - lambdaint) * x_noisy2
        print("interpol", interpol.shape)
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            time=t,
            noise=interpol,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"], interpol, img1, img2

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        time=1000,
        noise=None,
        input_masked=None,
        mask=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """

        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        indices = list(range(time))[::-1]
        # print('indices', indices)
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    input_masked=input_masked,
                    mask=mask,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,  # index of current step
        t_cpu=None,
        t_prev=None,  # index of step that we are going to compute,  only used for heun
        t_prev_cpu=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
        sampling_steps=0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        relerr = lambda x, y: (x - y).abs().sum() / y.abs().sum()
        if cond_fn is not None:
            out, saliency = self.condition_score2(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        eps_orig = self._predict_eps_from_xstart(
            x_t=x, t=t, pred_xstart=out["pred_xstart"]
        )
        if self.mode == "default":
            shape = x.shape
        elif self.mode == "segmentation":
            shape = eps_orig.shape
        else:
            raise NotImplementedError(f'mode "{self.mode}" not implemented')

        if not sampling_steps:
            alpha_bar_orig = _extract_into_tensor(self.alphas_cumprod, t, shape)
            alpha_bar_prev_orig = _extract_into_tensor(
                self.alphas_cumprod_prev, t, shape
            )
        else:
            xp = np.arange(0, 1000, 1, dtype=np.float)
            alpha_cumprod_fun = interp1d(
                xp,
                self.alphas_cumprod,
                bounds_error=False,
                fill_value=(self.alphas_cumprod[0], self.alphas_cumprod[-1]),
            )
            alpha_bar_orig = alpha_cumprod_fun(t_cpu).item()
            alpha_bar_prev_orig = alpha_cumprod_fun(t_prev_cpu).item()
        sigma = (
            eta
            * ((1 - alpha_bar_prev_orig) / (1 - alpha_bar_orig)) ** 0.5
            * (1 - alpha_bar_orig / alpha_bar_prev_orig) ** 0.5
        )
        noise = th.randn(size=shape, device=x.device)
        mean_pred = (
            out["pred_xstart"] * alpha_bar_prev_orig**0.5
            + (1 - alpha_bar_prev_orig - sigma**2) ** 0.5 * eps_orig
        )
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop_interpolation(
        self,
        model,
        shape,
        img1,
        img2,
        lambdaint,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        b = shape[0]
        t = th.randint(199, 200, (b,), device=device).long().to(device)
        img1 = torch.tensor(img1).to(device)
        img2 = torch.tensor(img2).to(device)
        noise = th.randn_like(img1).to(device)
        x_noisy1 = self.q_sample(x_start=img1, t=t, noise=noise).to(device)
        x_noisy2 = self.q_sample(x_start=img2, t=t, noise=noise).to(device)
        interpol = lambdaint * x_noisy1 + (1 - lambdaint) * x_noisy2
        print("interpol", interpol.shape)
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            time=t,
            noise=interpol,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"], interpol, img1, img2

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        sampling_steps=0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        b = shape[0]
        # t = th.randint(0,1, (b,), device=device).long().to(device)
        t = 1000
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            time=t,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            sampling_steps=sampling_steps,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_known(
        self,
        model,
        shape,
        img,
        mode="default",
        org=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        noise_level=1000,  # must be same as in training
        progress=False,
        conditioning=False,
        conditioner=None,
        classifier=None,
        eta=0.0,
        sampling_steps=0,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        b = shape[0]
        t = th.randint(0, 1, (b,), device=device).long().to(device)
        img = img.to(device)

        indices = list(range(t))[::-1]
        if mode == "segmentation":
            noise = None
            x_noisy = None
        elif mode == "default":
            noise = None
            x_noisy = None
        else:
            raise NotImplementedError(f'mode "{mode}" not implemented')

        final = None
        # pass images to be segmented as condition
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            segmentation_img=img,  # image to be segmented
            time=noise_level,
            noise=x_noisy,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            sampling_steps=sampling_steps,
        ):
            final = sample

        return final["sample"], x_noisy, img

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        segmentation_img=None,  # define to perform segmentation
        time=1000,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        sampling_steps=0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            if segmentation_img is None:  # normal sampling
                img = th.randn(*shape, device=device)
            else:  # segmentation mode
                label_shape = (
                    segmentation_img.shape[0],
                    model.out_channels,
                    *segmentation_img.shape[2:],
                )
                img = th.randn(
                    label_shape,
                    dtype=segmentation_img.dtype,
                    device=segmentation_img.device,
                )

        indices = list(range(time))[::-1]  # klappt nur für batch_size == 1

        print("indices", indices)

        if sampling_steps:
            tmp = np.linspace(999, 0, sampling_steps)
            tmp = np.append(tmp, -tmp[-2])
            indices = tmp[:-1].round().astype(np.int)
            indices_prev = tmp[1:].round().astype(np.int)
        else:
            indices_prev = [i - 1 for i in indices]

        if True:  # progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i, i_prev in zip(indices, indices_prev):  # 1000 -> 0
            if segmentation_img is not None:
                prev_img = img
                img = th.cat((segmentation_img, img), dim=1)
            t = th.tensor([i] * shape[0], device=device)
            t_prev = th.tensor([i_prev] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    t_cpu=i,
                    t_prev=t_prev,
                    t_prev_cpu=i_prev,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                    sampling_steps=sampling_steps,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(
        self,
        model,
        x_start,
        t,
        classifier=None,
        model_kwargs=None,
        noise=None,
        labels=None,
        mode="default",
        loss_level="image",
    ):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs - original image resolution.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :param labels: must be specified for mode='segmentation'
        :param mode:  can be default (image generation), segmentation
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}

        y_masked = x_start[:, 0, :, :, :]
        y_mask = x_start[:, 1, :, :, :]
        y_gt = x_start[:, 2, :, :, :]

        y_masked = y_masked.unsqueeze(1)
        y_mask = y_mask.unsqueeze(1)
        y_gt = y_gt.unsqueeze(1)

        # Wavelet transform the input image
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(y_masked)
        x_masked_dwt = th.cat([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        x_masked_dwt /= np.sqrt(8.0)

        # Wavelet transform the mask
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(y_mask)
        x_mask_dwt = th.cat([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        x_mask_dwt /= np.sqrt(8.0)

        # Wavelet transform the ground truth image
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(y_gt)
        x_gt_dwt = th.cat([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        x_gt_dwt /= np.sqrt(8.0)

        if mode == "default":
            noise = th.randn_like(y_gt)  # Sample noise - original image resolution.
            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(noise)
            noise_dwt = th.cat(
                [LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1
            )  # Wavelet transformed noise
            noise_dwt /= np.sqrt(8.0)
            x_t = self.q_sample(x_gt_dwt, t, noise=noise_dwt)  # Sample x_t

        else:
            raise ValueError(f'invalid mode {mode=}, needs to be "default"')

        model_input = torch.cat((x_t, x_masked_dwt, x_mask_dwt), dim=1)

        model_output = model(
            model_input, self._scale_timesteps(t), **model_kwargs
        )  # Model outputs denoised wavelet subbands

        model_output *= np.sqrt(8.0)

        # Inverse wavelet transform the model output
        B, _, H, W, D = model_output.size()
        model_output_idwt = idwt(
            model_output[:, 0, :, :, :].view(B, 1, H, W, D),
            model_output[:, 1, :, :, :].view(B, 1, H, W, D),
            model_output[:, 2, :, :, :].view(B, 1, H, W, D),
            model_output[:, 3, :, :, :].view(B, 1, H, W, D),
            model_output[:, 4, :, :, :].view(B, 1, H, W, D),
            model_output[:, 5, :, :, :].view(B, 1, H, W, D),
            model_output[:, 6, :, :, :].view(B, 1, H, W, D),
            model_output[:, 7, :, :, :].view(B, 1, H, W, D),
        )

        ########################## new loss

        mask_original = y_mask.type(torch.int)

        model_output_idwt_masked = model_output_idwt.clone()
        model_output_idwt_masked_mask_only = model_output_idwt_masked[mask_original == 1]

        y_gt_masked = y_gt.clone()
        y_gt_masked_mask_only = y_gt_masked[mask_original == 1]

        rec_loss_IWT_masked = F.l1_loss(model_output_idwt_masked_mask_only, y_gt_masked_mask_only)

        rec_loss_IWT_full = F.l1_loss(model_output_idwt, y_gt)

        rec_loss_IWT = rec_loss_IWT_full + rec_loss_IWT_masked

        return rec_loss_IWT, model_output, model_output_idwt, y_gt, y_masked, x_t, x_masked_dwt, x_mask_dwt, t[0]

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        # print('numstept',self.num_timesteps )
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)

            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bptimestepsd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
