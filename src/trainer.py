import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
from model.DocDiff import EMA, DocDiff
from schedule.diffusionSample import GaussianDiffusion
from schedule.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP, model_wrapper
from schedule.schedule import Schedule
from src.sobel import Laplacian
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from torchvision.utils import save_image
from tqdm import tqdm
from utils.perceptual_loss import PerceptualLoss
from utils.utils import get_A

# from utils.RGBuvHistBlock import RGBuvHistBlock
# from depth_anything.dpt import DPT_DINOv2


class Trainer:
    def __init__(self, config):
        self.mode = config.MODE
        self.schedule = Schedule(config.SCHEDULE, config.TIMESTEPS)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        in_channels = config.CHANNEL_X + config.CHANNEL_Y
        out_channels = config.CHANNEL_Y
        self.out_channels = out_channels
        self.network = DocDiff(
            input_channels=in_channels,
            output_channels=out_channels,
            n_channels=config.MODEL_CHANNELS,
            ch_mults=config.CHANNEL_MULT,
            n_blocks=config.NUM_RESBLOCKS,
        ).to(self.device)
        self.diffusion = GaussianDiffusion(
            self.network.denoiser, config.TIMESTEPS, self.schedule
        ).to(self.device)
        self.output_dir = config.OUTPUT_DIR
        self.test_path = os.path.join(config.OUTPUT_DIR, config.TEST_PATH)
        if not os.path.exists(self.test_path):
            os.makedirs(self.test_path)
        self.vis_path = os.path.join(config.OUTPUT_DIR, config.VIS_PATH)
        if not os.path.exists(self.vis_path):
            os.makedirs(self.vis_path)
        self.train_path = os.path.join(config.OUTPUT_DIR, config.TRAIN_PATH)
        if not os.path.exists(self.train_path):
            os.makedirs(self.train_path)
        self.weight_save_path = os.path.join(config.OUTPUT_DIR, config.WEIGHT_SAVE_PATH)
        if not os.path.exists(self.weight_save_path):
            os.makedirs(self.weight_save_path)

        self.pretrained_path_beta_predictor = config.PRETRAINED_PATH_BETA_PREDICTOR
        self.pretrained_path_denoiser = config.PRETRAINED_PATH_DENOISER
        self.pretrained_path_depth_estimator = config.PRETRAINED_PATH_DEPTH_ESTIMATOR

        self.continue_training = config.CONTINUE_TRAINING
        self.continue_training_steps = 0

        self.path_train_gt = config.PATH_GT
        self.path_train_img = config.PATH_IMG
        self.path_train_label_depth = config.PATH_GT_DEPTH
        self.path_train_hist = config.PATH_IMG_HIST

        self.path_test_img = config.PATH_TEST_IMG
        self.path_test_gt = config.PATH_TEST_GT
        self.path_test_label_depth = config.PATH_TEST_GT_DEPTH
        self.path_test_hist = config.PATH_TEST_IMG_HIST

        self.beta_loss = config.BETA_LOSS
        self.pre_ori = config.PRE_ORI
        self.high_low_freq = config.HIGH_LOW_FREQ
        self.image_size = config.IMAGE_SIZE
        self.native_resolution = config.NATIVE_RESOLUTION
        self.iteration_max = config.ITERATION_MAX
        self.LR = config.LR
        self.cross_entropy = nn.BCELoss()
        self.num_timesteps = config.TIMESTEPS
        self.ema_every = config.EMA_EVERY
        self.start_ema = config.START_EMA
        self.save_model_every = config.SAVE_MODEL_EVERY
        self.EMA_or_not = config.EMA
        self.DPM_SOLVER = config.DPM_SOLVER
        self.DPM_STEP = config.DPM_STEP

        if self.mode == 1 and self.continue_training == "True":
            print("Continue Training")
            self.network.beta_predictor.load_state_dict(
                torch.load(self.pretrained_path_beta_predictor)
            )
            self.network.denoiser.load_state_dict(
                torch.load(self.pretrained_path_denoiser)
            )
            self.continue_training_steps = config.CONTINUE_TRAINING_STEPS

        # if self.mode == 0:
        #     self.depth_estimator = DPT_DINOv2(
        #         encoder="vits",
        #         features=64,
        #         out_channels=[48, 96, 192, 384],
        #         localhub=True)
        #     self.depth_estimator.load_state_dict(
        #     torch.load(
        #         self.pretrained_path_depth_estimator,
        #         map_location="cpu",
        #     ),
        #     strict=True,
        #     )
        #     self.hist_estimator = RGBuvHistBlock(
        #         insz=config.IMAGE_SIZE[0],
        #         h=config.IMAGE_SIZE[1],
        #         resizing='sampling',
        #         method='inverse-quadratic',
        #         sigma=0.02,
        #         device=self.device)

        from data.data import UIEData

        if self.mode == 1:
            dataset_train = UIEData(
                self.path_train_img,
                self.path_train_gt,
                self.path_train_label_depth,
                self.path_train_hist,
                config.IMAGE_SIZE,
                self.mode,
            )
            self.dataloader_train = DataLoader(
                dataset_train,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                drop_last=False,
                num_workers=config.NUM_WORKERS,
            )
            dataset_eval = UIEData(
                self.path_test_img,
                self.path_test_gt,
                self.path_test_label_depth,
                self.path_test_hist,
                config.IMAGE_SIZE,
                mode=0,
            )
            self.dataloader_eval = DataLoader(
                dataset_eval,
                batch_size=config.BATCH_SIZE_VAL,
                shuffle=False,
                drop_last=False,
                num_workers=config.NUM_WORKERS,
            )
        else:
            dataset_test = UIEData(
                self.path_test_img,
                self.path_test_gt,
                self.path_test_label_depth,
                self.path_test_hist,
                config.IMAGE_SIZE,
                self.mode,
            )
            self.dataloader_test = DataLoader(
                dataset_test,
                batch_size=config.BATCH_SIZE_VAL,
                shuffle=False,
                drop_last=False,
                num_workers=config.NUM_WORKERS,
            )
        if self.mode == 1 and config.EMA == "True":
            self.EMA = EMA(0.9999)
            self.ema_model = copy.deepcopy(self.network).to(self.device)
        if config.LOSS == "L1":
            self.loss = nn.L1Loss()
        elif config.LOSS == "L2":
            self.loss = nn.MSELoss()
        else:
            print("Loss not implemented, setting the loss to L2 (default one)")
            self.loss = nn.MSELoss()
        if self.high_low_freq == "True":
            self.high_filter = Laplacian().to(self.device)
        self.perceptual_loss = PerceptualLoss()

    def test(self):
        def crop_concat(img, size=128):
            shape = img.shape
            correct_shape = (
                size * (shape[2] // size + 1),
                size * (shape[3] // size + 1),
            )
            one = torch.ones((shape[0], shape[1], correct_shape[0], correct_shape[1]))
            one[:, :, : shape[2], : shape[3]] = img
            # crop
            for i in range(shape[2] // size + 1):
                for j in range(shape[3] // size + 1):
                    if i == 0 and j == 0:
                        crop = one[
                            :, :, i * size : (i + 1) * size, j * size : (j + 1) * size
                        ]
                    else:
                        crop = torch.cat(
                            (
                                crop,
                                one[
                                    :,
                                    :,
                                    i * size : (i + 1) * size,
                                    j * size : (j + 1) * size,
                                ],
                            ),
                            dim=0,
                        )
            return crop

        def crop_concat_back(img, prediction, size=128):
            shape = img.shape
            for i in range(shape[2] // size + 1):
                for j in range(shape[3] // size + 1):
                    if j == 0:
                        crop = prediction[
                            (i * (shape[3] // size + 1) + j) * shape[0] : (
                                i * (shape[3] // size + 1) + j + 1
                            )
                            * shape[0],
                            :,
                            :,
                            :,
                        ]
                    else:
                        crop = torch.cat(
                            (
                                crop,
                                prediction[
                                    (i * (shape[3] // size + 1) + j) * shape[0] : (
                                        i * (shape[3] // size + 1) + j + 1
                                    )
                                    * shape[0],
                                    :,
                                    :,
                                    :,
                                ],
                            ),
                            dim=3,
                        )
                if i == 0:
                    crop_concat = crop
                else:
                    crop_concat = torch.cat((crop_concat, crop), dim=2)
            return crop_concat[:, :, : shape[2], : shape[3]]

        def min_max(array):
            return (array - array.min()) / (array.max() - array.min())

        with torch.no_grad():
            self.network.beta_predictor.load_state_dict(
                torch.load(self.pretrained_path_beta_predictor)
            )
            self.network.denoiser.load_state_dict(
                torch.load(self.pretrained_path_denoiser)
            )
            print("Test Model loaded")
            self.network.eval()
            tq = tqdm(self.dataloader_test)
            sampler = self.diffusion
            iteration = 0
            num_iter = 10
            for img, gt, label_depth, hist, name, in_size in tq:
                tq.set_description(
                    f"Iteration {iteration} / {len(self.dataloader_test.dataset)}"
                )
                iteration += 1
                pred_beta = self.network.beta_predictor(img.to(self.device))
                T_direct = torch.clamp((torch.exp(-pred_beta * label_depth)), 0, 1)
                T_scatter = torch.clamp((1 - torch.exp(-pred_beta * label_depth)), 0, 1)
                atm_light = [get_A(item) for item in img.to(self.device)]
                atm_light = torch.stack(atm_light).to(self.device)
                J = torch.clamp(
                    ((img.to(self.device) - T_scatter * atm_light) / T_direct), 0, 1
                )

                noisyImage = torch.randn_like(img.to(self.device))

                if self.DPM_SOLVER == "True":
                    sampledImgs = dpm_solver(
                        self.schedule.get_betas(),
                        self.network,
                        torch.cat((noisyImage, img.to(self.device)), dim=1),
                        self.DPM_STEP,
                    )
                else:
                    sampledImgs = sampler(
                        noisyImage.cuda(),
                        img.to(self.device),
                        J.to(self.device),
                        hist.to(self.device),
                        self.pre_ori,
                    )
                img_save = torch.cat(
                    [img, gt, J.cpu(), hist.cpu(), sampledImgs.cpu()], dim=3
                )

                if not os.path.exists(os.path.join(self.test_path, f"{num_iter}")):
                    os.makedirs(os.path.join(self.test_path, f"{num_iter}"))
                save_image(
                    img_save,
                    os.path.join(self.test_path, f"{num_iter}", f"{name[0]}"),
                    nrow=4,
                )
                if not os.path.exists(os.path.join(self.vis_path, f"{num_iter}")):
                    os.makedirs(os.path.join(self.vis_path, f"{num_iter}"))
                save_image(
                    Resize(in_size)(sampledImgs).cpu(),
                    os.path.join(self.vis_path, f"{num_iter}", f"{name[0]}"),
                )

    def evaluation(self, num_iter, beta_predictor_weight, denoiser_weight):
        def crop_concat(img, size=128):
            shape = img.shape
            correct_shape = (
                size * (shape[2] // size + 1),
                size * (shape[3] // size + 1),
            )
            one = torch.ones((shape[0], shape[1], correct_shape[0], correct_shape[1]))
            one[:, :, : shape[2], : shape[3]] = img
            # crop
            for i in range(shape[2] // size + 1):
                for j in range(shape[3] // size + 1):
                    if i == 0 and j == 0:
                        crop = one[
                            :, :, i * size : (i + 1) * size, j * size : (j + 1) * size
                        ]
                    else:
                        crop = torch.cat(
                            (
                                crop,
                                one[
                                    :,
                                    :,
                                    i * size : (i + 1) * size,
                                    j * size : (j + 1) * size,
                                ],
                            ),
                            dim=0,
                        )
            return crop

        def crop_concat_back(img, prediction, size=128):
            shape = img.shape
            for i in range(shape[2] // size + 1):
                for j in range(shape[3] // size + 1):
                    if j == 0:
                        crop = prediction[
                            (i * (shape[3] // size + 1) + j) * shape[0] : (
                                i * (shape[3] // size + 1) + j + 1
                            )
                            * shape[0],
                            :,
                            :,
                            :,
                        ]
                    else:
                        crop = torch.cat(
                            (
                                crop,
                                prediction[
                                    (i * (shape[3] // size + 1) + j) * shape[0] : (
                                        i * (shape[3] // size + 1) + j + 1
                                    )
                                    * shape[0],
                                    :,
                                    :,
                                    :,
                                ],
                            ),
                            dim=3,
                        )
                if i == 0:
                    crop_concat = crop
                else:
                    crop_concat = torch.cat((crop_concat, crop), dim=2)
            return crop_concat[:, :, : shape[2], : shape[3]]

        def min_max(array):
            return (array - array.min()) / (array.max() - array.min())

        with torch.no_grad():
            self.network.beta_predictor.load_state_dict(
                torch.load(beta_predictor_weight)
            )
            self.network.denoiser.load_state_dict(torch.load(denoiser_weight))

            print("Eval Model loaded")
            self.network.eval()
            tq = tqdm(self.dataloader_eval)
            sampler = self.diffusion
            iteration = 0
            for img, gt, label_depth, hist, name, in_size in tq:
                tq.set_description(
                    f"Iteration {iteration} / {len(self.dataloader_eval.dataset)}"
                )
                iteration += 1
                if self.native_resolution == "True":
                    temp = img
                    img = crop_concat(img)
                pred_beta = self.network.beta_predictor(img.to(self.device))
                depth = label_depth.to(self.device)
                T_direct = torch.clamp((torch.exp(-pred_beta * depth)), 0, 1)
                T_scatter = torch.clamp((1 - torch.exp(-pred_beta * depth)), 0, 1)
                atm_light = [get_A(item) for item in img.to(self.device)]
                atm_light = torch.stack(atm_light).to(self.device)
                J = torch.clamp(
                    ((img.to(self.device) - T_scatter * atm_light) / T_direct), 0, 1
                )
                noisyImage = torch.randn_like(img.to(self.device))

                if self.DPM_SOLVER == "True":
                    sampledImgs = dpm_solver(
                        self.schedule.get_betas(),
                        self.network,
                        torch.cat((noisyImage, img.to(self.device)), dim=1),
                        self.DPM_STEP,
                    )
                else:
                    sampledImgs = sampler(
                        noisyImage.cuda(),
                        img.to(self.device),
                        J.to(self.device),
                        hist.to(self.device),
                        self.pre_ori,
                    )
                img_save = torch.cat(
                    [
                        img,
                        gt,
                        J.cpu(),
                        hist.cpu(),
                        sampledImgs.cpu(),
                        T_direct.cpu(),
                        T_scatter.cpu(),
                        label_depth.cpu(),
                    ],
                    dim=3,
                )

                if not os.path.exists(os.path.join(self.test_path, f"{num_iter}")):
                    os.makedirs(os.path.join(self.test_path, f"{num_iter}"))
                save_image(
                    img_save,
                    os.path.join(self.test_path, f"{num_iter}", f"{name[0]}"),
                    nrow=4,
                )
                if not os.path.exists(os.path.join(self.vis_path, f"{num_iter}")):
                    os.makedirs(os.path.join(self.vis_path, f"{num_iter}"))
                save_image(
                    Resize(in_size)(sampledImgs).cpu(),
                    os.path.join(self.vis_path, f"{num_iter}", f"{name[0]}"),
                )

    def train(self):
        optimizer = optim.AdamW(
            self.network.parameters(), lr=self.LR, weight_decay=1e-4
        )
        iteration = self.continue_training_steps
        print("Starting Training", f"Step is {self.num_timesteps}")
        total_params = sum(
            p.numel() for p in self.network.parameters() if p.requires_grad
        )
        print(f"Trainable parameters: {total_params}")
        while iteration < self.iteration_max:
            tq = tqdm(self.dataloader_train)

            for img, gt, label_depth, hist, _, _ in tq:
                tq.set_description(f"Iteration {iteration} / {self.iteration_max}")
                self.network.train()
                optimizer.zero_grad()

                t = (
                    torch.randint(0, self.num_timesteps, (img.shape[0],))
                    .long()
                    .to(self.device)
                )
                J, noise_ref, denoised_J, T_direct, T_scatter = self.network(
                    gt.to(self.device),
                    img.to(self.device),
                    hist.to(self.device),
                    label_depth.to(self.device),
                    t,
                    self.diffusion,
                )
                if self.pre_ori == "True":
                    ddpm_loss = self.loss(denoised_J, gt.to(self.device))
                    perceptual_loss = self.perceptual_loss(
                        denoised_J, gt.to(self.device)
                    )
                else:
                    ddpm_loss = self.loss(denoised_J, noise_ref.to(self.device))
                    perceptual_loss = self.perceptual_loss(
                        denoised_J, noise_ref.to(self.device)
                    )

                loss = ddpm_loss + perceptual_loss
                loss.backward()
                optimizer.step()

                tq.set_postfix(
                    loss=loss.item(),
                    ddpm_loss=ddpm_loss.item(),
                    perceptual_loss=perceptual_loss.item(),
                )

                if iteration % 1000 == 0:
                    img_save = torch.cat(
                        [
                            img,
                            gt,
                            J.cpu(),
                            denoised_J.cpu(),
                            hist.cpu(),
                            T_direct.cpu(),
                            T_scatter.cpu(),
                            label_depth.cpu(),
                        ],
                        dim=3,
                    )
                    save_image(
                        img_save,
                        os.path.join(self.train_path, f"{iteration}.png"),
                        nrow=4,
                    )
                iteration += 1
                if self.EMA_or_not == "True":
                    if iteration % self.ema_every == 0 and iteration > self.start_ema:
                        print("EMA update")
                        self.EMA.update_model_average(self.ema_model, self.network)

                if iteration % self.save_model_every == 0:
                    print("Saving models")
                    if not os.path.exists(self.weight_save_path):
                        os.makedirs(self.weight_save_path)
                    torch.save(
                        self.network.beta_predictor.state_dict(),
                        os.path.join(
                            self.weight_save_path,
                            f"model_beta_predictor_{iteration}.pth",
                        ),
                    )
                    torch.save(
                        self.network.denoiser.state_dict(),
                        os.path.join(
                            self.weight_save_path, f"model_denoiser_{iteration}.pth"
                        ),
                    )

                    self.evaluation(
                        iteration,
                        os.path.join(
                            self.weight_save_path,
                            f"model_beta_predictor_{iteration}.pth",
                        ),
                        os.path.join(
                            self.weight_save_path, f"model_denoiser_{iteration}.pth"
                        ),
                    )


def dpm_solver(betas, model, x_T, steps, model_kwargs):
    # You need to firstly define your model and the extra inputs of your model,
    # And initialize an `x_T` from the standard normal distribution.
    # `model` has the format: model(x_t, t_input, **model_kwargs).
    # If your model has no extra inputs, just let model_kwargs = {}.

    # If you use discrete-time DPMs, you need to further define the
    # beta arrays for the noise schedule.

    # model = ....
    # model_kwargs = {...}
    # x_T = ...
    # betas = ....

    # 1. Define the noise schedule.
    noise_schedule = NoiseScheduleVP(schedule="discrete", betas=betas)

    # 2. Convert your discrete-time `model` to the continuous-time
    # noise prediction model. Here is an example for a diffusion model
    # `model` with the noise prediction type ("noise") .
    model_fn = model_wrapper(
        model,
        noise_schedule,
        model_type="noise",  # or "x_start" or "v" or "score"
        model_kwargs=model_kwargs,
    )

    # 3. Define dpm-solver and sample by singlestep DPM-Solver.
    # (We recommend singlestep DPM-Solver for unconditional sampling)
    # You can adjust the `steps` to balance the computation
    # costs and the sample quality.
    dpm_solver = DPM_Solver(
        model_fn,
        noise_schedule,
        algorithm_type="dpmsolver++",
        correcting_x0_fn="dynamic_thresholding",
    )
    # Can also try
    # dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

    # You can use steps = 10, 12, 15, 20, 25, 50, 100.
    # Empirically, we find that steps in [10, 20] can generate quite good samples.
    # And steps = 20 can almost converge.
    x_sample = dpm_solver.sample(
        x_T,
        steps=steps,
        order=1,
        skip_type="time_uniform",
        method="singlestep",
    )
    return x_sample
