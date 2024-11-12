import copy
import os
import time
from typing import Dict

import cv2
import hydra
import numpy as np
import omegaconf
import open3d as o3d
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from taxpose.datasets.point_cloud_data_module import MultiviewDataModule
from taxpose.nets.transformer_flow import ResidualFlow_DiffEmbTransformer
from taxpose.training.flow_equivariance_training_module_nocentering import \
    EquivarianceTrainingModule


class LearnPlace:

    cfg: DictConfig = None

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.project_dir = os.path.join(os.path.dirname(__file__), '../../')

    def learn(self) -> None:
        torch.set_float32_matmul_precision("high")
        torch.autograd.set_detect_anomaly(True)
        torch.cuda.empty_cache()
        torch.multiprocessing.set_sharing_strategy("file_system")

        cfg = self.cfg
        # print(OmegaConf.to_yaml(cfg, resolve=True))

        # torch.set_float32_matmul_precision("medium")
        pl.seed_everything(cfg.seed)
        logger = WandbLogger(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            group=cfg.wandb.group,
            save_dir=cfg.wandb.save_dir,
            job_type=cfg.job_type,
            save_code=True,
            log_model=True,
            config=omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            ),
        )
        # logger.log_hyperparams(cfg)
        # logger.log_hyperparams({"working_dir": os.getcwd()})
        trainer = pl.Trainer(
            logger=logger,
            accelerator="gpu",
            devices=[0],
            log_every_n_steps=cfg.training.log_every_n_steps,
            check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
            # reload_dataloaders_every_n_epochs=1,
            # callbacks=[SaverCallbackModel(), SaverCallbackEmbnnActionAnchor()],
            callbacks=[
                # This checkpoint callback saves the latest model during training, i.e. so we can resume if it crashes.
                # It saves everything, and you can load by referencing last.ckpt.
                ModelCheckpoint(
                    dirpath=cfg.lightning.checkpoint_dir,
                    filename="{epoch}-{step}",
                    monitor="step",
                    mode="max",
                    save_weights_only=False,
                    save_last=True,
                    every_n_epochs=1,
                ),
                # This checkpoint will get saved to WandB. The Callback mechanism in lightning is poorly designed, so we have to put it last.
                ModelCheckpoint(
                    dirpath=cfg.lightning.checkpoint_dir,
                    filename="{epoch}-{step}-{train_loss:.2f}-weights-only",
                    monitor="val_loss",
                    mode="min",
                    save_weights_only=True,
                ),
            ],
            max_epochs=cfg.training.max_epochs,
        )

        dm = MultiviewDataModule(
            batch_size=cfg.training.batch_size,
            num_workers=cfg.resources.num_workers,
            cfg=cfg.dm,
        )

        dm.setup()

        network = ResidualFlow_DiffEmbTransformer(
            emb_dims=cfg.model.emb_dims,
            emb_nn=cfg.model.emb_nn,
            return_flow_component=cfg.model.return_flow_component,
            center_feature=cfg.model.center_feature,
            pred_weight=cfg.model.pred_weight,
            multilaterate=cfg.model.multilaterate,
            sample=cfg.model.mlat_sample,
            mlat_nkps=cfg.model.mlat_nkps,
            break_symmetry=cfg.break_symmetry,
        )

        model = EquivarianceTrainingModule(
            network,
            lr=cfg.training.lr,
            image_log_period=cfg.training.image_logging_period,
            displace_loss_weight=cfg.training.displace_loss_weight,
            consistency_loss_weight=cfg.training.consistency_loss_weight,
            direct_correspondence_loss_weight=cfg.training.direct_correspondence_loss_weight,
            weight_normalize=cfg.task.phase.weight_normalize,
            sigmoid_on=cfg.training.sigmoid_on,
            softmax_temperature=cfg.task.phase.softmax_temperature,
            flow_supervision=cfg.training.flow_supervision,
        )

        model.cuda()
        model.train()
        if cfg.training.load_from_checkpoint:
            print("loaded checkpoint from")
            print(cfg.training.checkpoint_file)
            model.load_state_dict(
                torch.load(hydra.utils.to_absolute_path(cfg.training.checkpoint_file))[
                    "state_dict"
                ]
            )

        else:
            # Might be empty and not have those keys defined.
            # TODO: move this pretraining into the model itself.
            # TODO: figure out if we can get rid of the dictionary and make it null.
            if cfg.model.pretraining:
                if cfg.model.pretraining.checkpoint_file_action is not None:
                    # # Check to see if it's a wandb checkpoint.
                    # TODO: need to retrain a few things... checkpoint didn't stick...
                    emb_nn_action_state_dict = self.__load_emb_weights(
                        cfg.pretraining.checkpoint_file_action, cfg.wandb, logger.experiment
                    )
                    # checkpoint_file_fn = maybe_load_from_wandb(
                    #     cfg.pretraining.checkpoint_file_action, cfg.wandb, logger.experiment.run
                    # )

                    model.model.emb_nn_action.load_state_dict(
                        emb_nn_action_state_dict)
                    print(
                        "-----------------------Pretrained EmbNN Action Model Loaded!-----------------------"
                    )
                    print(
                        "Loaded Pretrained EmbNN Action: {}".format(
                            cfg.pretraining.checkpoint_file_action
                        )
                    )
                if cfg.pretraining.checkpoint_file_anchor is not None:
                    emb_nn_anchor_state_dict = self.__load_emb_weights(
                        cfg.pretraining.checkpoint_file_anchor, cfg.wandb, logger.experiment
                    )
                    model.model.emb_nn_anchor.load_state_dict(
                        emb_nn_anchor_state_dict)
                    print(
                        "-----------------------Pretrained EmbNN Anchor Model Loaded!-----------------------"
                    )
                    print(
                        "Loaded Pretrained EmbNN Anchor: {}".format(
                            cfg.pretraining.checkpoint_file_anchor
                        )
                    )
        trainer.fit(model, dm)

    def get_pre_placement_offset(self):

        project_dir = self.project_dir
        cfg = self.cfg

        data_dir = os.path.join(project_dir, cfg.data_dir)

        num_demos = cfg.num_demos

        teach_pose_dir = os.path.join(data_dir, "teach_data/pose_data")

        pre_placement_offsets = []

        # Load pose data
        for i in range(num_demos):
            placement_pose_file = f"demo{i+1}_{cfg.training.target.view}_pose.npy"
            pre_placement_pose_file = f"demo{i+1}_pre_{cfg.training.target.view}_pose.npy"

            placement_pose = np.load(os.path.join(
                teach_pose_dir, placement_pose_file))
            pre_placement_pose = np.load(os.path.join(
                teach_pose_dir, pre_placement_pose_file))
            pre_placement_offset = np.linalg.inv(
                placement_pose) @ pre_placement_pose

            pre_placement_offsets.append(pre_placement_offset)

        # Save dataset
        pre_placement_offset = np.mean(pre_placement_offsets, axis=0)
        save_dir = os.path.join(data_dir, "learn_data")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, f"pre_{cfg.training.target.view}_offset.npy"),
                pre_placement_offset)

    def prepare_dataset_old(self) -> None:

        project_dir = self.project_dir
        cfg = self.cfg
        data_dir = os.path.join(project_dir, cfg.data_dir)
        num_demos = cfg.training.num_demos
        train_demos = np.ceil(
            num_demos * (1-cfg.training.test_ratio)).astype(int)
        test_demos = num_demos - train_demos
        action_class = cfg.training.action.class_idx
        anchor_class = cfg.training.anchor.class_idx
        
        teach_pcd_dir = os.path.join(data_dir, "teach_data/pcd_data")
        teach_pose_dir = os.path.join(data_dir, "teach_data/pose_data")
        train_save_dir = os.path.join(data_dir, "learn_data/train")
        test_save_dir = os.path.join(data_dir, "learn_data/test")

        data_template = {
            'action':
                {
                    'pcd': None,
                    'tf': None,
                },
            'anchor':
                {
                    'pcd': None,
                    'tf': None,
                }
        }

        data_list = [copy.deepcopy(data_template) for _ in range(num_demos)]

        # Load pose data
        for i in range(num_demos):
            action_pose_file = f"demo{i+1}_{cfg.training.action.view}_pose.npy"
            anchor_pose_file = f"demo{i+1}_{cfg.training.anchor.view}_pose.npy"
            target_pose_file = f"demo{i+1}_{cfg.training.target.view}_pose.npy"

            action_pose = np.load(os.path.join(teach_pose_dir, action_pose_file))
            anchor_pose = np.load(os.path.join(teach_pose_dir, anchor_pose_file))
            target_pose = np.load(os.path.join(teach_pose_dir, target_pose_file))
            target_bias = np.asarray(cfg.training.target.bias)

            # transform for action pose
            action_cam = cfg.training.action.camera
            action_cam_type = cfg.training.action.camera_type
            action_tf = None
            if action_cam_type == 'on_base':
                T_base2cam_file = cfg.devices.cameras[action_cam].setup.T_base2cam 
                T_base2cam = np.load(os.path.join(
                    project_dir, T_base2cam_file))
                action_tf = T_base2cam
            elif action_cam_type == 'on_eef':
                T_eef2cam_file = cfg.devices.cameras[action_cam].setup.T_eef2cam 
                T_eef2cam = np.load(os.path.join(project_dir, T_eef2cam_file))
                action_tf = action_pose @ T_eef2cam

            # transform for anchor pose
            anchor_cam = cfg.training.anchor.camera
            anchor_cam_type = cfg.training.anchor.camera_type
            anchor_tf = None
            if anchor_cam_type == 'on_base':
                T_base2cam_file = cfg.devices.cameras[anchor_cam].setup.T_base2cam 
                T_base2cam = np.load(os.path.join(
                    project_dir, T_base2cam_file))
                anchor_tf = T_base2cam
            elif anchor_cam_type == 'on_hand':
                T_eef2cam_file = cfg.devices.cameras[anchor_cam].setup.T_eef2cam 
                T_eef2cam = np.load(os.path.join(project_dir, T_eef2cam_file))
                anchor_tf = anchor_pose @ T_eef2cam
            else:
                raise ValueError(f"Invalid camera type for anchor camera: {anchor_cam_type}")

            action_target = target_pose @ np.linalg.inv(action_pose)
            data_list[i]['action']['tf'] = (action_target @ action_tf) + target_bias
            data_list[i]['anchor']['tf'] = anchor_tf

        # Load point cloud data
        for i in range(num_demos):
            action_view = cfg.training.action.view
            action_cam = cfg.training.action.camera
            anchor_view = cfg.training.anchor.view
            anchor_cam = cfg.training.anchor.camera
            anchor_pcd_file = f"demo{i+1}_{anchor_view}_{anchor_cam}_pointcloud.ply"
            action_pcd_file = f"demo{i+1}_{action_view}_{action_cam}_pointcloud.ply"

            action_pcd = o3d.io.read_point_cloud(
                os.path.join(teach_pcd_dir, action_pcd_file))
            anchor_pcd = o3d.io.read_point_cloud(
                os.path.join(teach_pcd_dir, anchor_pcd_file))

            # crop and filter point cloud data
            action_crop_box = cfg.training.action.crop_box
            action_pcd = self.__crop_point_cloud(action_pcd, action_crop_box)
            action_pcd = self.__filter_point_cloud(action_pcd)
            action_pcd.points = o3d.utility.Vector3dVector(
                np.asarray(action_pcd.points)/1000.0)

            anchor_crop_box = cfg.training.anchor.crop_box
            anchor_pcd = self.__crop_point_cloud(anchor_pcd, anchor_crop_box)
            anchor_pcd = self.__filter_point_cloud(anchor_pcd)
            anchor_pcd.points = o3d.utility.Vector3dVector(
                np.asarray(anchor_pcd.points)/1000.0)

            # transform point cloud data
            anchor_pcd = anchor_pcd.transform(data_list[i]['anchor']['tf'])
            action_pcd = action_pcd.transform(data_list[i]['action']['tf'])

            data_list[i]['action']['pcd'] = np.asarray(action_pcd.points)
            data_list[i]['anchor']['pcd'] = np.asarray(anchor_pcd.points)

        # Save dataset(s)
        for i in range(num_demos):
            file_name = f'{i}_teleport_obj_points.npz'
            if not os.path.exists(train_save_dir):
                os.makedirs(train_save_dir)
            if not os.path.exists(test_save_dir):
                os.makedirs(test_save_dir)

            action_points = data_list[i]['action']['pcd']
            anchor_points = data_list[i]['anchor']['pcd']
            clouds = np.concatenate([action_points, anchor_points], axis=0)
            classes = np.concatenate(
                [np.full(action_points.shape[0], action_class),
                 np.full(anchor_points.shape[0], anchor_class)],
                axis=0)

            save_dir = train_save_dir if i < train_demos else test_save_dir
            np.savez_compressed(
                os.path.join(save_dir, file_name),
                clouds=clouds,
                classes=classes,
                colors=None,
                shapenet_ids=None)
            
        breakpoint()

    def prepare_dataset(self) -> None:
        project_dir = self.project_dir
        cfg = self.cfg
        data_dir = os.path.join(project_dir, cfg.data_dir)
        num_demos = cfg.training.num_demos
        train_demos = np.ceil(
            num_demos * (1-cfg.training.test_ratio)).astype(int)
        test_demos = num_demos - train_demos
        action_class = cfg.training.action.class_idx
        anchor_class = cfg.training.anchor.class_idx
        ctr = 0
        
        teach_pcd_dir = os.path.join(data_dir, "teach_data/pcd_data")
        teach_pose_dir = os.path.join(data_dir, "teach_data/pose_data")
        train_save_dir = os.path.join(data_dir, "learn_data/train")
        test_save_dir = os.path.join(data_dir, "learn_data/test")
        
        for demo in range(num_demos):
            print(f"Processing demo {demo}")
            anchor_points_list = []
            action_points_list = []
            
            anchor_pcd_file = os.path.join(teach_pcd_dir, f"demo{demo}_{cfg.training.anchor.view}0_{cfg.training.anchor.camera}_pointcloud.ply")
            anchor_pcd = o3d.io.read_point_cloud(anchor_pcd_file)
            anchor_pcd = self.__filter_point_cloud(anchor_pcd)
            
            #scale point cloud
            anchor_points = np.asarray(anchor_pcd.points) / 1000
            anchor_pcd.points = o3d.utility.Vector3dVector(anchor_points)
            
            anchor_view_pose_file = os.path.join(teach_pose_dir, f"demo{demo}_{cfg.training.anchor.view}0_pose.npy")
            T_eef2cam_file = cfg.devices.cameras[cfg.training.anchor.camera].setup.T_eef2cam
            
            # anchor view point
            T_base2eef = np.load(anchor_view_pose_file)
            T_eef2cam = np.load(os.path.join(project_dir, T_eef2cam_file))
            T_base2cam = np.dot(T_base2eef, T_eef2cam)
            
            # target pose
            T_base2targeteef = np.load(os.path.join(teach_pose_dir, f"demo0_placement_pose.npy"))
            T_ee2target = np.asarray(self.cfg.devices.gripper.T_ee2target)
            T_base2target = np.dot(T_base2targeteef, T_ee2target)
            
            T_base2placeeef = np.load(os.path.join(teach_pose_dir, f"demo{demo}_placement_pose.npy"))
            T_base2place = np.dot(T_base2placeeef, T_ee2target)
            
            # transform anchor pcd in target pose frame for easy cropping
            crop_tf = np.dot(np.linalg.inv(T_base2target), T_base2cam)
            anchor_pcd = anchor_pcd.transform(crop_tf)
            
            # crop_points
            anchor_bounds = cfg.training.anchor.object_bounds
            object_bounds = {
                'min': np.array(anchor_bounds.min)/1000,
                'max': np.array(anchor_bounds.max)/1000
            }
            anchor_pcd = self.__crop_point_cloud(anchor_pcd, object_bounds)
            anchor_points_list.append(np.asarray(anchor_pcd.points))    
            
            # transform target pose frame to placement pose
            place_tf = np.linalg.inv(T_base2place) @ T_base2target
                        
            for var in range(cfg.training.action.view_variations.count):
                # action point clouds @ target
                action_pcd_file = os.path.join(teach_pcd_dir, f"demo{demo}_{cfg.training.action.view}{var}_{cfg.training.action.camera}_pointcloud.ply")
                action_pcd = o3d.io.read_point_cloud(action_pcd_file)
                action_pcd = self.__filter_point_cloud(action_pcd)

                #scale point cloud
                action_points = np.asarray(action_pcd.points) / 1000
                action_pcd.points = o3d.utility.Vector3dVector(action_points)
                
                action_view_pose_file = os.path.join(teach_pose_dir, f"demo{demo}_{cfg.training.action.view}{var}_pose.npy")
                T_base2action = np.load(action_view_pose_file)
                T_base2cam_file = cfg.devices.cameras[cfg.training.action.camera].setup.T_base2cam
                T_base2cam = np.load(os.path.join(project_dir, T_base2cam_file))
                action_tf = np.linalg.inv(T_ee2target) @ (np.linalg.inv(T_base2action) @ T_base2cam)
                action_pcd = action_pcd.transform(action_tf)
                action_pcd = action_pcd.transform(np.linalg.inv(place_tf))
                
                # crop_points
                action_bounds = cfg.training.action.object_bounds
                object_bounds = {
                    'min': np.array(action_bounds.min)/1000,
                    'max': np.array(action_bounds.max)/1000
                }
                action_points = np.asarray(action_pcd.points)
                action_points = self.__crop_points(action_points, object_bounds)
                action_points_list.append(action_points)
            
            # save data to train or test dir
            if not os.path.exists(train_save_dir):
                os.makedirs(train_save_dir)
            if not os.path.exists(test_save_dir):
                os.makedirs(test_save_dir)            

            for anchor in anchor_points_list:
                anchor[:, 2] = -anchor[:, 2]
                for action in action_points_list:
                    action[:, 2] = -action[:, 2]
                    clouds = np.concatenate([action, anchor], axis=0)
                    classes = np.concatenate(
                        [np.full(action.shape[0], action_class),
                        np.full(anchor.shape[0], anchor_class)],
                        axis=0)
                    file_name = f'{ctr}_teleport_obj_points.npz'
                    save_dir = train_save_dir if demo < train_demos else test_save_dir
                    np.savez_compressed(
                        os.path.join(save_dir, file_name),
                        clouds=clouds,
                        classes=classes,
                        colors=None,
                        shapenet_ids=None)
                    ctr += 1
            
    def __crop_points(self, points: np.ndarray, crop_box: Dict[str, float]) -> np.ndarray:

        mask = np.logical_and.reduce(
            (points[:, 0] >= crop_box['min'][0],
                points[:, 0] <= crop_box['max'][0],
                points[:, 1] >= crop_box['min'][1],
                points[:, 1] <= crop_box['max'][1],
                points[:, 2] >= crop_box['min'][2],
                points[:, 2] <= crop_box['max'][2])
        )
        points = points[mask]

        return points

    def __crop_point_cloud(self, pcd: o3d.geometry.PointCloud, crop_box: Dict[str, float]) -> o3d.geometry.PointCloud:

        points = np.asarray(pcd.points)
        mask = np.logical_and.reduce(
            (points[:, 0] >= crop_box['min'][0],
             points[:, 0] <= crop_box['max'][0],
             points[:, 1] >= crop_box['min'][1],
             points[:, 1] <= crop_box['max'][1],
             points[:, 2] >= crop_box['min'][2],
             points[:, 2] <= crop_box['max'][2])
        )
        points = points[mask]

        pcd_t = o3d.geometry.PointCloud()
        pcd_t.points = o3d.utility.Vector3dVector(points)

        return pcd_t

    def __filter_point_cloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        return pcd

    def __load_emb_weights(self, checkpoint_reference, wandb_cfg=None, run=None):
        if checkpoint_reference.startswith(wandb_cfg.entity):
            artifact_dir = os.path.join(
                wandb_cfg.artifact_dir, checkpoint_reference)
            if run is None or not isinstance(run, wandb.sdk.wandb_run.Run):
                # Download without a run
                api = wandb.Api()
                artifact = api.artifact(checkpoint_reference, type="model")
            else:
                artifact = run.use_artifact(checkpoint_reference)
            checkpoint_path = artifact.get_path(
                "model.ckpt").download(root=artifact_dir)
            weights = torch.load(checkpoint_path)["state_dict"]
            # remove "model.emb_nn" prefix from keys
            weights = {k.replace("model.emb_nn.", "")
                                 : v for k, v in weights.items()}
            return weights
        else:
            return torch.load(hydra.utils.to_absolute_path(checkpoint_reference))[
                "embnn_state_dict"
            ]
