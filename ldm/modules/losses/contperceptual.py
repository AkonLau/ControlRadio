import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.functional import peak_signal_noise_ratio
def l1(x, y):
    return torch.abs(x-y)

def l2(x, y):
    return torch.pow((x-y), 2)

class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge",
                 pixel_loss="l2"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        assert pixel_loss in ["l1", "l2"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        if pixel_loss == "l1":
            self.pixel_loss = l1
        else:
            self.pixel_loss = l2

        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        # rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        rec_loss = self.pixel_loss(inputs.contiguous(), reconstructions.contiguous())

        if self.perceptual_weight > 0:
            p_loss = self.perceptual_weight * self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            nll_loss = rec_loss + p_loss
        else:
            p_loss = torch.tensor([0.0])
            nll_loss = rec_loss

        nll_loss = nll_loss / torch.exp(self.logvar) + self.logvar
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        else:
            weighted_nll_loss = nll_loss

        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            weighted_kl_loss = self.kl_weight * kl_loss
            weighted_g_loss = d_weight * disc_factor * g_loss
            loss = weighted_nll_loss + weighted_kl_loss + weighted_g_loss

            # 2026Äê7ÔÂ15ÈÕmodify logs by liukj
            # log mse_loss
            criterion = nn.MSELoss()
            # for evaluate the generation ability of the denoiser
            vae_mse_loss_0 = criterion(reconstructions.clamp_(-1., 1.).contiguous(),
                                       inputs.clamp_(-1., 1.).contiguous())

            normal_pred= reconstructions.clamp_(-1., 1.).contiguous() * 0.5 + 0.5  # scale to [0, 1]
            normal_target = inputs.clamp_(-1., 1.).contiguous() * 0.5 + 0.5  # scale to [0, 1]
            # print(normal_pred.size(), normal_target.max(), normal_target.min())
            # print(normal_target.size(), normal_target.max(), normal_target.min())
            vae_mse_loss = criterion(normal_pred, normal_target)
            vae_rmse_loss = torch.sqrt(vae_mse_loss)
            vae_nmse_loss = self.compute_nmse(normal_pred, normal_target)
            vae_ssim = structural_similarity_index_measure(normal_pred, normal_target)
            vae_psnr = peak_signal_noise_ratio(normal_pred, normal_target)

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/g_loss".format(split): weighted_g_loss.detach().mean(),
                   "{}/kl_loss".format(split): weighted_kl_loss.detach().mean(),
                   "{}/nll_loss".format(split): weighted_nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/logvar".format(split): self.logvar.detach(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   # "{}/xxx".format(xxx): torch.tensor(xxx),
                   "{}/vae_mse_loss_0".format(split): vae_mse_loss_0.detach().mean(),
                   "{}/vae_mse_loss".format(split): vae_mse_loss.detach().mean(),
                   "{}/vae_rmse_loss".format(split): vae_rmse_loss.detach().mean(),
                   "{}/vae_nmse_loss".format(split): vae_nmse_loss.detach().mean(),
                   "{}/vae_ssim".format(split): vae_ssim.detach().mean(),
                   "{}/vae_psnr".format(split): vae_psnr.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

    def compute_nmse(self, predicted, ground_truth):
        assert predicted.shape == ground_truth.shape, "Predicted and ground truth must have the same shape."
        numerator = torch.sum((predicted - ground_truth) ** 2, dim=[1, 2, 3])
        denominator = torch.sum(ground_truth ** 2, dim=[1, 2, 3])
        denominator = torch.clamp(denominator, min=1e-6)

        nmse = numerator / denominator
        return nmse.mean()

