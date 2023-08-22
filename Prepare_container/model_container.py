from Architectures.UNet_with_attention_Temb_ping import UnetAttnResidulTimeemb
import torch
from Architectures.UNet_with_attention_Temb_openAI import UnetAttnTembAI
from DPM_models.DPM_original import DDPM
from general_utils.fixSeed_fn_tool import fix_all_seed
from torch import optim
from general_utils.privateOptimizer import RAdam


def DPM_container(config_box):
    fix_all_seed(config_box['seed'])
    model = UnetAttnTembAI(
        image_size=config_box['Diffusion']['img_size'],
        group=config_box['Diffusion']['group'],
        in_channels=config_box['Diffusion']['input_dim'],
        model_channels=config_box['Diffusion']['model_channels'],
        out_channels=config_box['Diffusion']['input_dim'],
        num_heads=config_box['Diffusion']['num_heads'],
    )
    diffusion_model = DDPM(
        model,
        timesteps=config_box['Diffusion']['timesteps'],
        img_size=config_box['Diffusion']['img_size'],
        loss_type=config_box['Diffusion']['loss_type'],
        beta_schedule=config_box['Diffusion']['beta_scheduler'],
        log_every_t=config_box['Diffusion']['log_every_t'],
        v_posterior=config_box['Diffusion']['v_posterior'],
        channels=config_box['Diffusion']['input_dim'],
        save_path=config_box['Trainer']['save_dir'],
    )
    optimizer = optim.Adam(diffusion_model.parameters(), lr=config_box['Optim']['lr'])

    return diffusion_model, optimizer


def self_load_state_dict(model, state_dict: 'OrderedDict[str, Any]', indicator='encoder'):
    m_keys = model.state_dict().keys()
    pre_keys = state_dict.keys()

    for name in m_keys:
        if f'{indicator}.{name}' in pre_keys:
            param = state_dict[f'{indicator}.{name}']
            model.state_dict().get(name).copy_(param)
    return model

def controNet_optimizer(control_model, config_box):
    lr = config_box['Optim']['lr']
    params = list(control_model.input_zero_out.parameters())
    params += list(control_model.control_model.model.input_blocks.parameters())
    params += list(control_model.zero_conv.parameters())
    params += list(control_model.control_model.model.middle_block.parameters())
    params += list(control_model.middle_block_out.parameters())
    if config_box['Optim']['name'] in ['Adam']:
        optimizer = torch.optim.Adam(params, lr=lr)
    else:
        optimizer = RAdam(params, lr=lr)

    return optimizer
