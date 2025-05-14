from .specificity_models import ViewSpecificAE
import torch.nn as nn
import torch
class IVAE(nn.Module):
    def __init__(self, args, device = 'cpu'):
        super(IVAE, self).__init__()
        self.args = args
        self.device = device
        self.views = args.views
        # m unimodal vae
        for i in range(self.args.views):
            self.__setattr__(f"venc_{i + 1}", ViewSpecificAE(c_enable=True,
                                                             v_dim=self.args.vspecific.v_dim,
                                                             c_dim=self.args.consistency.c_dim,
                                                             latent_ch=self.args.vspecific.latent_ch,
                                                             num_res_blocks=self.args.vspecific.num_res_blocks,
                                                             block_size=self.args.vspecific.block_size,
                                                             channels=self.args.vspecific.in_channel,
                                                             basic_hidden_dim=self.args.vspecific.basic_hidden_dim,
                                                             ch_mult=self.args.vspecific.ch_mult,
                                                             init_method=self.args.backbone.init_method,
                                                             kld_weight=self.args.vspecific.kld_weight,
                                                             device=self.device,
                                                             dropout=self.args.train.dropout))

    def forward(self, Xs):
        outs = []
        for i in range(self.views): #each view
            venc = self.__getattr__(f"venc_{i+1}")
            out, _, _ = venc(Xs[i]) #decoder(z), mu, log(v)
            outs.append(out)

        return outs

    def get_loss(self, Xs, C=None):
        return_details = {}
        loss = 0.
        for i in range(self.views):
            venc = self.__getattr__(f"venc_{i+1}")
            recon_loss, kld_loss = venc.get_loss(Xs[i], C)

            return_details[f"v{i+1}_recon-loss"] = recon_loss.item()
            return_details[f"v{i+1}_kld-loss"] = kld_loss.item()

            loss += kld_loss + recon_loss
            return_details[f"v{i+1}_total-loss"] = loss.item()

        return_details['ivae_total_loss'] = loss.item()
        return loss, return_details

    def vspecific_features(self, Xs, best_view=False):
        vspecific_features = []
        for i in range(self.views):
            venc = self.__getattr__(f"venc_{i + 1}")
            feature = venc.latent(Xs[i])
            vspecific_features.append(feature)
        if best_view:
            return vspecific_features[self.config.best_view]
        else:
            return vspecific_features

    @torch.no_grad()
    def generate(self, Xs, C):
        outputs = []
        for i, x in enumerate(Xs):
            output, _, _ = self.__getattr__(f"venc_{i + 1}")(x, C)
            outputs.append(output)

        return outputs







    












