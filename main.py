from DPM.DPM_original import DDPM
from Prepare_container.dataloader_container import data_load
from general_utils.config_manager import ConfigManager
from trainers.trainer_DPM import DPM_Trainer
from torch import optim

from archs.UNet_with_attention_Temb_openAI import UnetAttnTembAI

if __name__ == "__main__":
    cmanager = ConfigManager("configs/diffusion.yaml", strict=True)
    config = cmanager.config

    model = UnetAttnTembAI(
        image_size=config['Diffusion']['img_size'],
        group=config['Diffusion']['group'],
        in_channels=config['Diffusion']['input_dim'],
        model_channels=config['Diffusion']['model_channels'],
        out_channels=config['Diffusion']['input_dim'],
        num_heads=config['Diffusion']['num_heads'],
    )
    diffusion_model = DDPM(
        model,
        timesteps=config['Diffusion']['timesteps'],
        img_size=config['Diffusion']['img_size'],
        loss_type=config['Diffusion']['loss_type'],
        beta_schedule=config['Diffusion']['beta_scheduler'],
        log_every_t=config['Diffusion']['log_every_t'],
        v_posterior=config['Diffusion']['v_posterior'],
        channels=config['Diffusion']['input_dim'],
        save_path=config['Trainer']['save_dir'],
    )
    optimizer = optim.Adam(diffusion_model.parameters(), lr=config['Optim']['lr'])

    Loaders_container = data_load(config_box=config)

    train_S_loader = Loaders_container.get(config["Domain"]["source"])[0]
    val_S_loader = Loaders_container.get(config["Domain"]["source"])[1]
    test_S_loader = Loaders_container.get(config["Domain"]["source"])[2]

    train_T_loader = Loaders_container.get(config["Domain"]["target"])[0]
    val_T_loader = Loaders_container.get(config["Domain"]["target"])[1]
    test_T_loader = Loaders_container.get(config["Domain"]["target"])[2]

    trainer = DPM_Trainer(
        diffusion_model=diffusion_model,
        optimizer=optimizer,
        trainS_loader=train_S_loader,
        config=config,
        **config['Trainer']
    )

    trainer.start_training()
