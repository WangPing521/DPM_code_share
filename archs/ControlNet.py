from torch import nn
from tqdm import tqdm

from archs.UNet_with_attention_Temb_openAI import TimestepEmbedSequential
from archs.utils import zero_module, timestep_embedding
from Prepare_container.model_container import DPM_container, self_load_state_dict
import torch

class ControlNet(nn.Module):
    def __init__(
            self,
            config_box,
            locked_model,
            state_dict,
            hint_channels=1,
    ):
        super(ControlNet, self).__init__()
        self.config_box = config_box
        self.locked_model = locked_model

        DPM_Cmodel, _ = DPM_container(config_box)
        self.control_model = self_load_state_dict(DPM_Cmodel, state_dict.get('module_state'), indicator='diffusion_model')
        self.control_model.to(config_box['Trainer']['device'])

        self.hint_channels = hint_channels
        self.input_zero_out = self.make_zero_conv(1)

        # insert zero_conv into the inputblock and middleblock for self.control_model
        self.zero_conv = nn.ModuleList([self.make_zero_conv(self.config_box['Diffusion']['model_channels'])])

        idx_indicator = tqdm(range(len(self.control_model.model.input_blocks)))
        for (idx, sub_block) in zip(idx_indicator, self.control_model.model.input_blocks):
            if idx != 0:
                ch = sub_block[0].out_channels
                self.zero_conv.append(self.make_zero_conv(ch))

        self.middle_block_out = self.make_zero_conv(self.control_model.model.middle_block[0].out_channels)

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(nn.Conv2d(channels, channels, 1, padding=0)))

    def forward(self, x, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.config_box['Diffusion']['model_channels'], repeat_only=False)
        emb = self.control_model.model.time_embed(t_emb)
        emb = emb.detach_()
        hs = []
        h = x + self.input_zero_out(context, emb, x) # there is no error for the order, please remember
        for control_module, zero_conv, locked_module in zip(self.control_model.model.input_blocks, self.zero_conv, self.locked_model.model.input_blocks):
            h = control_module(h, emb, context)
            x = locked_module(x, emb, context)
            h = zero_conv(h, emb, context)
            de_in = h + x
            hs.append(de_in)


        h = self.control_model.model.middle_block(h, emb, context)
        h = self.middle_block_out(h, emb, context)
        x = self.locked_model.model.middle_block(x, emb, context)
        out = x + h

        # the rest fixed decoder
        for locked_decoder_block in self.locked_model.model.output_blocks:
            out = torch.cat([out, hs.pop()], dim=1)
            out = locked_decoder_block(out, emb, context)

        out = self.locked_model.model.out(out)

        return out
