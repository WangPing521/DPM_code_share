from DPM.DPM_condition import DDPM_condition
from Prepare_container.dataloader_container import data_load
from Prepare_container.model_container import DPM_container, controNet_optimizer, self_load_state_dict
from archs.ControlNet import ControlNet
from general_utils.config_manager import ConfigManager
import torch
import torch.multiprocessing as mp

from trainers.trainer_ControlNetDPM import Control_DPM_Trainer

if __name__ == "__main__":
    mp.set_start_method('spawn')
    cmanager = ConfigManager("../configs/ControlNet_diffusion.yaml", strict=True)
    config = cmanager.config

    DPM_model, _ = DPM_container(config)
    weight = f"{config['Inference']['DPM_checkpoint']}/last.pth"
    state_dict = torch.load(weight)
    locked_model = self_load_state_dict(DPM_model, state_dict.get('module_state'), indicator='diffusion_model')
    locked_model.to(config['Trainer']['device'])
    for param in locked_model.parameters():
        param.detach_()

    Control_model = ControlNet(config, locked_model, state_dict, hint_channels=config['Diffusion']['input_dim'])
    optimizer = controNet_optimizer(Control_model, config)

    Control_diffusion = DDPM_condition(
        Control_model,
        timesteps=config['Diffusion']['timesteps'],
        img_size=config['Diffusion']['img_size'],
        loss_type=config['Diffusion']['loss_type'],
        beta_schedule=config['Diffusion']['beta_scheduler'],
        log_every_t=config['Diffusion']['log_every_t'],
        v_posterior=config['Diffusion']['v_posterior'],
        channels=config['Diffusion']['input_dim'],
        save_path=config['Trainer']['save_dir']
    )

    Loaders_container = data_load(config_box=config)

    train_S_loader = Loaders_container.get(config["Domain"]["source"])[0]
    val_S_loader = Loaders_container.get(config["Domain"]["source"])[1]
    test_S_loader = Loaders_container.get(config["Domain"]["source"])[2]

    train_T_loader = Loaders_container.get(config["Domain"]["target"])[0]
    val_T_loader = Loaders_container.get(config["Domain"]["target"])[1]
    test_T_loader = Loaders_container.get(config["Domain"]["target"])[2]

    trainer = Control_DPM_Trainer(
        diffusion_model=Control_diffusion,
        optimizer=optimizer,
        trainS_loader=train_S_loader,
        config=config,
        **config['Trainer']
    )

    trainer.start_training()
