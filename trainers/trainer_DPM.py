from pathlib import Path

from Storage_and_Meter.metric_container import SummaryWriter, Storage
from general_utils.config_fn_tool import set_environment, write_yaml
from general_utils.image_save_fn_tool import save_diffusion_imgs
from general_utils.path_tool import RUN_PATH
from general_utils.privateOptimizer import tqdm
from meters_register.diffusion import diffusion_meters
from trainers._base import _TrainerBase

class DPM_Trainer(_TrainerBase):

    def __init__(
            self,
            diffusion_model,
            optimizer,
            trainS_loader,
            max_epoch: int = 100,
            iterations: int = 400,
            save_dir: str = "base",
            config: dict = None,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.optimizer = optimizer
        self.TrainS_loader = trainS_loader
        self.max_epochs = max_epoch
        self.iterations = iterations
        self.save_dir: Path = Path(RUN_PATH) / str(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.start_step = 0
        if config:
            self.config = config.copy()
            self.config.pop("Config", None)
            write_yaml(self.config, save_dir=self.save_dir, save_name="config.yaml")
            set_environment(config.get("Environment"))

        self.device = self.config['Trainer']['device']
        self._storage = Storage(self.save_dir)
        self.writer = SummaryWriter(str(self.save_dir))
        self.meters = diffusion_meters()

    def run_step(self, s_data, cur_batch):
        imgS, gt, filename = (
            s_data[0][0].to(self.device),
            s_data[0][1].to(self.device),
            s_data[1],
        )
        loss = self.diffusion_model(imgS)
        return loss

    def train(self, trainS_loader, cur_epoch):
        self.diffusion_model.train()
        train_steps_indicator = tqdm(range(self.iterations))
        train_steps_indicator.set_description(f"training step {cur_epoch:04d}")

        for cur_batch, (batch_id, trainS_data) in enumerate(zip(train_steps_indicator, trainS_loader)):
            self.optimizer.zero_grad()

            loss = self.run_step(s_data=trainS_data, cur_batch=cur_batch)

            loss.backward()
            self.optimizer.step()

            self.meters['loss'].add(loss.item())

            report_dict = self.meters.statistics()
            train_steps_indicator.set_postfix_statics(report_dict, cache_time=20)
        train_steps_indicator.close()

        report_dict = self.meters.statistics()
        assert report_dict is not None
        return dict(report_dict)

    def val(self, batch_size, cur_epoch):
        self.diffusion_model.eval()

        img, intermediate = self.diffusion_model.p_sample_loop(shape=(batch_size, 1, self.config['Diffusion']['img_size'], self.config['Diffusion']['img_size']), return_intermediates=True)
        save_diffusion_imgs(img.squeeze(1), root=self.config['Trainer']['save_dir'], cur_epoch=cur_epoch)
        save_diffusion_imgs(intermediate, root=self.config['Trainer']['save_dir'], cur_epoch=cur_epoch)


    def start_training(self):
        self.to(self.device)
        self.cur_step = 0
        for self.cur_step in range(self.start_step, self.max_epochs):
            self.meters.reset()
            with self.meters.focus_on("tra"):
                self.meters['lr'].add(self.optimizer.param_groups.__getitem__(0).get('lr'))

                train_metrics = self.train(trainS_loader=self.TrainS_loader, cur_epoch=self.cur_step)

            if self.cur_step % 20 == 0 or self.cur_step == self.max_epochs - 1:
                self.val(batch_size=4, cur_epoch=self.cur_step)

            with self._storage:
                self._storage.add_from_meter_interface(tra=train_metrics, epoch=self.cur_step)
                self.writer.add_scalars_from_meter_interface(tra=train_metrics, epoch=self.cur_step)

            self.save_checkpoint(self.state_dict(), current_epoch=self.cur_step, save_dir=self.save_dir)



