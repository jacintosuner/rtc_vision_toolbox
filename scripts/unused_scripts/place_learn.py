import argparse
import os
import time
from typing import Dict
import copy

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

from rtc_core.place_skill.place_learn import LearnPlace


class LearnPlace_:

    cfg: DictConfig = None

    @classmethod
    def learn(cls) -> None:
        torch.set_float32_matmul_precision("high")
        torch.autograd.set_detect_anomaly(True)
        torch.cuda.empty_cache()
        torch.multiprocessing.set_sharing_strategy("file_system")
       
        cfg = cls.cfg 
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
                    emb_nn_action_state_dict = cls.__load_emb_weights(
                        cfg.pretraining.checkpoint_file_action, cfg.wandb, logger.experiment
                    )
                    # checkpoint_file_fn = maybe_load_from_wandb(
                    #     cfg.pretraining.checkpoint_file_action, cfg.wandb, logger.experiment.run
                    # )

                    model.model.emb_nn_action.load_state_dict(emb_nn_action_state_dict)
                    print(
                        "-----------------------Pretrained EmbNN Action Model Loaded!-----------------------"
                    )
                    print(
                        "Loaded Pretrained EmbNN Action: {}".format(
                            cfg.pretraining.checkpoint_file_action
                        )
                    )
                if cfg.pretraining.checkpoint_file_anchor is not None:
                    emb_nn_anchor_state_dict = cls.__load_emb_weights(
                        cfg.pretraining.checkpoint_file_anchor, cfg.wandb, logger.experiment
                    )
                    model.model.emb_nn_anchor.load_state_dict(emb_nn_anchor_state_dict)
                    print(
                        "-----------------------Pretrained EmbNN Anchor Model Loaded!-----------------------"
                    )
                    print(
                        "Loaded Pretrained EmbNN Anchor: {}".format(
                            cfg.pretraining.checkpoint_file_anchor
                        )
                    )
        trainer.fit(model, dm)
        
    @classmethod
    def __load_emb_weights(cls, checkpoint_reference, wandb_cfg=None, run=None):
        if checkpoint_reference.startswith(wandb_cfg.entity):
            artifact_dir = os.path.join(wandb_cfg.artifact_dir, checkpoint_reference)
            if run is None or not isinstance(run, wandb.sdk.wandb_run.Run):
                # Download without a run
                api = wandb.Api()
                artifact = api.artifact(checkpoint_reference, type="model")
            else:
                artifact = run.use_artifact(checkpoint_reference)
            checkpoint_path = artifact.get_path("model.ckpt").download(root=artifact_dir)
            weights = torch.load(checkpoint_path)["state_dict"]
            # remove "model.emb_nn" prefix from keys
            weights = {k.replace("model.emb_nn.", ""): v for k, v in weights.items()}
            return weights
        else:
            return torch.load(hydra.utils.to_absolute_path(checkpoint_reference))[
                "embnn_state_dict"
            ]
    
    @classmethod
    def get_pre_placement_offset(cls):
        data_dir = "/home/mfi/repos/rtc_vision_toolbox/data/demonstrations/06-20-wp/"
        num_demos = 5
    
        teach_pose_dir = os.path.join(data_dir, "teach_data/pose_data")
        
        pre_placement_offsets = []
        
        # Load pose data
        for i in range(num_demos):
            placement_pose_file = f"demo{i+1}_placement_pose.npy"
            pre_placement_pose_file = f"demo{i+1}_pre_placement_pose.npy"
            
            placement_pose = np.load(os.path.join(teach_pose_dir, placement_pose_file))
            pre_placement_pose = np.load(os.path.join(teach_pose_dir, pre_placement_pose_file))
            pre_placement_offset = np.linalg.inv(placement_pose) @ pre_placement_pose
            
            pre_placement_offsets.append(pre_placement_offset)
            
        # Save dataset
        pre_placement_offset = np.mean(pre_placement_offsets, axis=0)
        save_dir = os.path.join(data_dir, "learn_data")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, "pre_placement_offset.npy"), pre_placement_offset)
                            
    @classmethod
    def prepare_dataset(cls) -> None:
            
        data_dir = "/home/mfi/repos/rtc_vision_toolbox/data/demonstrations/06-20-wp/"
        num_demos = 5
        test_demos = 1
        action_class = 0
        anchor_class = 1        
        
        teach_pcd_dir = os.path.join(data_dir, "teach_data/pcd_data")
        teach_pose_dir = os.path.join(data_dir, "teach_data/pose_data")
        calib_dir = os.path.join(data_dir, "calib_data")
        train_save_dir = os.path.join(data_dir,"learn_data/train")
        test_save_dir = os.path.join(data_dir,"learn_data/test")
        
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
        
        # data_list = [data_template for _ in range(num_demos)]
        data_list = []
        
        # Load pose data
        T_base2cam2 = np.load(os.path.join(calib_dir, "T_base2cam2.npy"))
        T_eef2cam3 = np.load(os.path.join(calib_dir, "T_eef2cam3.npy"))        
        
        for i in range(num_demos):
            data = copy.deepcopy(data_template)
            target_pose_file = f"demo{i+1}_placement_pose.npy"
            action_pose_file = f"demo{i+1}_gripper_close_up_pose.npy"
            anchor_pose_file = f"demo{i+1}_ih_camera_view_pose.npy" 
            
            target_pose = np.load(os.path.join(teach_pose_dir, target_pose_file))
            action_pose = np.load(os.path.join(teach_pose_dir, action_pose_file))
            anchor_pose = np.load(os.path.join(teach_pose_dir, anchor_pose_file))
            
            T = target_pose @ np.linalg.inv(action_pose) @ T_base2cam2
            T[2,3] = T[2,3] + 0.035
            T[1,3] = T[1,3] + 0.001
            T[0,3] = T[0,3] - 0.003
            data['action']['tf'] = T
            data['anchor']['tf'] = anchor_pose @ T_eef2cam3       
            
            data_list.append(data) 
        
        # Load point cloud data
        for i in range(num_demos):
            anchor_pcd_file = f"demo{i+1}_ih_camera_view_cam3_gripper_pointcloud.ply"
            action_pcd_file= f"demo{i+1}_gripper_close_up_cam2_closeup_pointcloud.ply"
            
            action_pcd = o3d.io.read_point_cloud(os.path.join(teach_pcd_dir, action_pcd_file))
            anchor_pcd = o3d.io.read_point_cloud(os.path.join(teach_pcd_dir, anchor_pcd_file))

            # crop and filter point cloud data
            action_crop_box = {
                'min_bound': np.array([-200, -60, 0.0]),
                'max_bound': np.array([200, 60, 200]),

            }
            action_pcd = cls.__crop_point_cloud(action_pcd, action_crop_box)
            action_pcd = cls.__filter_point_cloud(action_pcd)
            action_pcd.points = o3d.utility.Vector3dVector(np.asarray(action_pcd.points)/1000.0)
            
            anchor_crop_box = {
                'min_bound': np.array([-53, -28, 0.0]),
                'max_bound': np.array([53, 25, 300])
            }
            anchor_pcd = cls.__crop_point_cloud(anchor_pcd, anchor_crop_box)
            anchor_pcd = cls.__filter_point_cloud(anchor_pcd)
            anchor_pcd.points = o3d.utility.Vector3dVector(np.asarray(anchor_pcd.points)/1000.0)
            
            # transform point cloud data
            anchor_pcd = anchor_pcd.transform(data_list[i]['anchor']['tf'])
            action_pcd = action_pcd.transform(data_list[i]['action']['tf'])
            
            data_list[i]['action']['pcd'] = np.asarray(action_pcd.points)
            data_list[i]['anchor']['pcd'] = np.asarray(anchor_pcd.points)

        # Save dataset
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
            if i < test_demos:
                save_dir = test_save_dir
            else:
                save_dir = train_save_dir
            np.savez_compressed(
                os.path.join(save_dir, file_name),
                clouds=clouds,
                classes=classes,
                colors=None,
                shapenet_ids=None)
    
    @classmethod
    def __crop_point_cloud(cls, pcd: o3d.geometry.PointCloud, crop_box: Dict[str, float]) -> o3d.geometry.PointCloud:
        
        points = np.asarray(pcd.points)
        mask = np.logical_and.reduce(
            (points[:, 0] >= crop_box['min_bound'][0],
             points[:, 0] <= crop_box['max_bound'][0],
             points[:, 1] >= crop_box['min_bound'][1],
             points[:, 1] <= crop_box['max_bound'][1],
             points[:, 2] >= crop_box['min_bound'][2],
             points[:, 2] <= crop_box['max_bound'][2])
        )
        points = points[mask]
        
        pcd_t = o3d.geometry.PointCloud()
        pcd_t.points = o3d.utility.Vector3dVector(points)
    
        return pcd_t

    @classmethod
    def __filter_point_cloud(cls, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        return pcd

@hydra.main(config_path="../models/taxpose/configs", config_name="commands/mfi/waterproof/train_taxpose_06-20-wp_place")
def get_cfg(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    
    # LearnPlace.cfg = cfg

if __name__ == "__main__":
     
    # config = get_cfg()
    
    # LearnPlace.cfg = config
    
    # LearnPlace.get_pre_placement_offset()
    # LearnPlace.prepare_dataset()
    # LearnPlace.learn()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    
    # Read configuration file
    config_file = args.config
    config_dir = os.path.dirname(config_file)
    config_name = os.path.basename(config_file)
    
    print(f"Reading configuration file: {config_file}")
    print(f"Configuration directory: {config_dir}")
    print(f"Configuration name: {config_name}")
    
    hydra.initialize(config_path=config_dir, version_base="1.3")
    config: DictConfig = hydra.compose(config_name)
      
    place_learn = LearnPlace(config)
    place_learn.prepare_dataset()
    place_learn.get_pre_placement_offset()
