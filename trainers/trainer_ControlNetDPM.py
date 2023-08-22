import torch
from tqdm import tqdm

from DPM.utils import canny_edge_detection
from general_utils.image_save_fn_tool import save_shifted_Imgs, compress_and_delete_folder
from general_utils.path_tool import PROJECT_PATH
from general_utils.schedulers import GradualWarmupScheduler
from trainers.trainer_DPM import DPM_Trainer


class Control_DPM_Trainer(DPM_Trainer):

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
        DPM_Trainer.__init__(self, diffusion_model, optimizer, trainS_loader, max_epoch, iterations, save_dir, config, *args, **kwargs)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(90, 1), eta_min=1e-7)
        self.scheduler = GradualWarmupScheduler(optimizer, multiplier=300, total_epoch=10, after_scheduler=scheduler)

    def run_step(self, s_data, cur_batch):
        imgS, gt, filename = (
            s_data[0][0].to(self.device),
            s_data[0][1].to(self.device),
            s_data[1],
        )
        edge = canny_edge_detection(imgS)
        loss = self.diffusion_model(imgS, edge)
        return loss

    def val(self, data_loader, cur_epoch):
        self.diffusion_model.eval()
        data_indicator = tqdm(range(len(data_loader)))

        for batch_idx, (batchs_id, data) in enumerate(zip(data_indicator, data_loader)):
            img, gt, filename = (
                data[0][0].to(self.config['Trainer']['device']),
                data[0][1].to(self.config['Trainer']['device']),
                data[1]
            )
            edge = canny_edge_detection(img)
            if cur_epoch == self.max_epochs - 1:
                save_shifted_Imgs(edge.squeeze(1), filename, root=self.config['Trainer']['save_dir'], mode='S_edges', data_idx=self.config['Data']['name'])

            # The inputs are noise and edges
            img_r, intermediate = self.diffusion_model.p_sample_loop(img, edge, return_intermediates=True)
            save_shifted_Imgs(img_r.squeeze(1), filename, root=self.config['Trainer']['save_dir'], mode=f'S_{cur_epoch}', data_idx=self.config['Data']['name'])
            save_shifted_Imgs(intermediate, filename, root=self.config['Trainer']['save_dir'], mode=f'S_{cur_epoch}', data_idx=self.config['Data']['name'])

    def start_training(self):
        self.to(self.device)
        self.cur_step = 0
        for self.cur_step in range(self.start_step, self.max_epochs):
            self.meters.reset()
            with self.meters.focus_on("tra"):
                self.meters['lr'].add(self.optimizer.param_groups.__getitem__(0).get('lr'))

                train_metrics = self.train(trainS_loader=self.TrainS_loader, cur_epoch=self.cur_step)

                if self.config['Optim']['name'] in ['Radam']:
                    self.scheduler.step()

            if self.cur_step != 0 and self.cur_step % 50 == 0 or self.cur_step == self.max_epochs - 1:
                self.val(data_loader=self.TrainS_loader, cur_epoch=self.cur_step)

            with self._storage:
                self._storage.add_from_meter_interface(tra=train_metrics, epoch=self.cur_step)
                self.writer.add_scalars_from_meter_interface(tra=train_metrics, epoch=self.cur_step)

            self.save_checkpoint(self.state_dict(), current_epoch=self.cur_step, save_dir=self.save_dir)

        compress_and_delete_folder(folder_path=f'{PROJECT_PATH}/scripts_training/runs/{self.config["Trainer"]["save_dir"]}')





