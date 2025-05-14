import torch
import torch.nn as nn
import numpy as np
from .independent_VAE import IVAE
from .consistency_models import ConsistencyAE
from utils.misc import mask_image
from .mi_estimators import CLUBSample


class dualvae(nn.Module):
    def __init__(self, config, device='cpu'):
        super(dualvae, self).__init__()
        self.config = config
        self.views = self.config.views
        self.device = device
        self.c_dim = config.consistency.c_dim
        self.v_dim = config.vspecific.v_dim

        # Specific model
        self.spe_enc = IVAE(args=self.config, device=device)

        # Consistency model
        self.cons_enc = ConsistencyAE(
            basic_hidden_dim=config.consistency.basic_hidden_dim,
            c_dim=config.consistency.c_dim,
            v_dim=config.vspecific.v_dim,
            continous=config.consistency.continous,
            in_channel=config.consistency.in_channel,
            num_res_blocks=config.consistency.num_res_blocks,
            ch_mult=config.consistency.ch_mult,
            block_size=config.consistency.block_size,
            temperature=config.consistency.temperature,
            latent_ch=config.consistency.latent_ch,
            kld_weight=config.consistency.kld_weight,
            views=config.views,
            categorical_dim=config.dataset.class_num,
            dropout=config.train.dropout
        )
        self.mi_est = nn.ModuleList([CLUBSample(self.v_dim,
                                                self.c_dim,
                                                config.disent.hidden_size) for _ in range(self.views)])

    def get_loss(self, Xs):
        # extract specific-views
        assert len(Xs) == self.views
        return_details = {}

        cons_Kld_loss, cons_return_details = self.cons_enc.get_loss(Xs=Xs,
                                                      mask_ratio=self.config.train.masked_ratio,
                                                      mask_patch_size=self.config.train.mask_patch_size,
                                                      _mask_view=self.config.train.mask_view,
                                                      mask_view_ratio=self.config.train.mask_view_ratio
                                                      )
        return_details.update(cons_return_details)
        C = self.cons_enc.consistency_features(Xs)

        spe_loss, spe_loss_details = self.spe_enc.get_loss(Xs, C)
        return_details.update(spe_loss_details)
        spe_repr = self.spe_enc.vspecific_features(Xs)
        # MI loss
        tot_disent_loss = 0.

        for i in range(self.views):
            mi_est = self.mi_est[i]
            cur_disent_loss = mi_est.learning_loss(C, spe_repr[i])

            tot_disent_loss += cur_disent_loss

        return_details['disent_loss'] = tot_disent_loss.item()
        return_details['total_loss'] = (cons_Kld_loss + spe_loss + tot_disent_loss).item()

        return cons_Kld_loss + spe_loss + tot_disent_loss, return_details

    def forward(self, Xs):
        con_repr = self.cons_enc(Xs)
        return con_repr

    def all_features(self, Xs):
        C = self.consistency_features(Xs)
        spe_repr = self.vspecific_features(Xs)  # list
        # V = spe_repr[self.config.vspecific.best_view]
        concat_list = [torch.cat([C, v], dim=-1) for v in spe_repr]  # list
        all_concate = torch.cat([C, torch.cat(spe_repr, dim=1)], dim=1)
        return C, spe_repr, concat_list, all_concate

    @torch.no_grad()
    def generate(self, Xs):
        C = self.consistency_features(Xs)
        return self.spe_enc.generate(Xs=Xs,C=C)

    @torch.no_grad()
    def specific_latent_dist(self, x, v):  # distribution of view v
        mu, logvar = self.spe_enc.__getattr__(f"venc_{v + 1}").encode(x)
        return mu, logvar

    @torch.no_grad()
    def consistency_latent_dist(self, Xs):
        mu, logvar = self.cons_enc.encode(Xs)
        return mu, logvar

    @torch.no_grad()
    def consistency_features(self, Xs):
        consist_feature = self.cons_enc.consistency_features(Xs)
        return consist_feature

    @torch.no_grad()
    def vspecific_features(self, Xs, best_view=False):
        vspecific_features = []
        for i in range(self.views):
            venc = self.spe_enc.__getattr__(f"venc_{i + 1}")
            feature = venc.latent(Xs[i])
            vspecific_features.append(feature)
        if best_view:
            return vspecific_features[self.config.best_view]
        else:
            return vspecific_features

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu










