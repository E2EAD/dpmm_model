import os
import sys
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import json
import math
import carla
import pickle
from loguru import logger
from typing import List, Dict, Any

# Import necessary functions from your existing code
# 获取当前脚本的上两级目录（LEGION/my_code）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # parent_dir=my_code
sys.path.append(parent_dir)
from tool.my_config import get_cfg
from tool.bev_tool import calculate_pixel_per_meter
from dataset.b2d_143_dataset import (
    convert_veh_coord, scale_and_crop_image, tokenize_action, 
    tokenize_traj, get_depth, my_update_intrinsics, ProcessImage
)

class AbilityDataset(Dataset):
    def __init__(self, data_root: str, ability: str, is_train: bool, config: Any, 
                 scen_skill_desc_list: List[Dict], scenario_dirs: List[str] = None):
        """
        Dataset that loads scenarios based on a specific ability/skill.
        
        Args:
            data_root: Root directory containing all scenario directories
            ability: The ability to filter scenarios by (e.g., 'Overtaking')
            is_train: Whether this is training data
            config: Configuration object
            scen_skill_desc_list: List of scenario descriptions with skill information
            scenario_dirs: Optional list of specific scenario directories to use
        """
        super(AbilityDataset, self).__init__()
        self.cfg = config
        self.ability = ability
        self.is_train = is_train
        self.data_root = data_root
        self.scen_skill_desc_list = scen_skill_desc_list

        self.BOS_token = self.cfg.token_nums - 3
        self.EOS_token = self.BOS_token + 1
        self.PAD_token = self.EOS_token + 1

        self.token_nums = self.cfg.token_nums
        self.rgb_crop = self.cfg.image_crop
        self.rgb_scale = self.cfg.image_scale
        self.image_process_rgb = ProcessImage(self.rgb_scale, self.rgb_crop)

        self.ability2label = {}
        for label, ability in enumerate(self.cfg.ability_list):
            self.ability2label[ability] = label
        
        # Filter scenarios by the specified ability
        if scenario_dirs is None:
            # Get all scenario directories if not provided
            scenario_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) 
                           if os.path.isdir(os.path.join(data_root, d))]
        # print(f'scenario directories:')
        # for d in scenario_dirs:
        #     print(d)
        
        # Filter scenarios by ability using the scen_skill_desc_list
        self.scenario_dirs = []
        for scen_dir in scenario_dirs:
            scen_name = os.path.basename(scen_dir).split('_')[0]
            # print(f'scen name: {scen_name}')
            matched = [item for item in scen_skill_desc_list if item['scen_name'] == scen_name]
            # print(f'matched: {len(matched)}')
            # print(f'matched: {matched}')
            
            # if matched and matched[0]['skill'] == ability:
            #     print(f'append ability related scenario: {scen_dir}')
            #     self.scenario_dirs.append(scen_dir)
            if matched:
                for m in matched:
                    if self.ability in m['skill']:
                        print(f'append ability related scenario: {scen_dir}')
                        single_scenario_dirs = [os.path.join(scen_dir, d) for d in os.listdir(scen_dir) 
                           if os.path.isdir(os.path.join(scen_dir, d))]
                        self.scenario_dirs.extend(single_scenario_dirs)
        print(f'{self.ability} contains {len(self.scenario_dirs)} senario-town-route-weather clips')
        
        if not self.scenario_dirs:
            logger.warning(f"No scenarios found for ability: {self.ability}")
        
        # Initialize data storage lists
        self._init_data_arrays()
        
        # Load data from all filtered scenarios
        self.get_data()
        
        logger.info(f"Loaded {len(self.rgb_front_paths)} sequences for ability: {self.ability}")

    def _init_data_arrays(self):
        """Initialize all data storage arrays."""
        # Image paths
        self.rgb_front_paths = []
        self.rgb_front_left_paths = []
        self.rgb_front_right_paths = []
        self.rgb_back_paths = []
        self.rgb_back_left_paths = []
        self.rgb_back_right_paths = []

        self.depth_front_paths = []
        self.depth_front_left_paths = []
        self.depth_front_right_paths = []
        self.depth_back_paths = []
        self.depth_back_left_paths = []
        self.depth_back_right_paths = []

        self.bev_paths = []

        # Trajectory data
        self.all_trajectory_points = []
        self.near_trajectory_points = []
        self.target_points = []
        self.near_traj_tokens = []

        # Ego state data
        self.ego_position = []
        self.speed = []
        self.rad_yaw = []
        self.acceleration = []
        self.angular_velocity = []
        self.ego_motion = []

        # Control data
        self.control_tokens = []
        self.traj_tokens = []

        # Navigation data
        self.nav_cmd = []
        self.nav_cmd_near = []
        self.nav_cmd_far = []

        # Other data
        self.frame_ids = []
        self.bev_pix_per_meter = []
        self.enco_descriptions = []
        self.surroundings = []
        
        # Camera parameters (will be set per scenario)
        self.intrinsics = []
        self.extrinsics = []

        self.ability_one_hot = []

    def get_data(self):
        """Load data from all scenarios for this ability."""
        # Calculate trajectory indices (same as in ScenarioDataset)
        self.traj_time_near = self.cfg.traj_time_near
        self.traj_time_far = self.cfg.traj_time_far
        self.traj_interval_near = self.cfg.traj_interval_near
        self.traj_interval_far = self.cfg.traj_interval_far
        self.fps = 10
        
        near_end_frame = int(self.traj_time_near * self.fps)
        far_end_frame = int(self.traj_time_far * self.fps)
        
        near_indices = list(range(self.traj_interval_near, near_end_frame + self.traj_interval_near, 
                                self.traj_interval_near))
        far_indices = list(range(near_end_frame + self.traj_interval_far, 
                               far_end_frame + self.traj_interval_far, self.traj_interval_far))
        
        self.traj_all_indices = near_indices + far_indices
        self.traj_near_indices = near_indices
        
        # Process each scenario
        for scenario_dir in self.scenario_dirs:
            self._process_scenario(scenario_dir)

    def _process_scenario(self, scenario_dir):
        """Process a single scenario directory."""
        scen_name = os.path.basename(scenario_dir).split('_')[0]
        matched = [item for item in self.scen_skill_desc_list if item['scen_name'] == scen_name]
        
        if not matched:
            logger.warning(f"No matching scen in scen_skill_desc_list for {scen_name}")
            return

        for item in self.scen_skill_desc_list:
            if item['scen_name'] == scen_name:
                abilities = item['skill']  
                ability_one_hot = np.zeros((5,), dtype=np.float32)
                idx = []
                for a in abilities:
                    idx.append(self.ability2label[a])
                ability_one_hot[idx] = 1.0
                ability_one_hot = ability_one_hot
                # print(f'ability_one_hot for this sceanario: {ability_one_hot}')

        enco_description = torch.from_numpy(matched[0]['enco_description']).float().squeeze(0)
        
        # Paths for this scenario
        anno_path = os.path.join(scenario_dir, 'extracted_anno')
        camera_path = os.path.join(scenario_dir, 'camera')

        rgb_front_path_base = os.path.join(camera_path, 'rgb_front')
        rgb_front_left_path_base = os.path.join(camera_path, 'rgb_front_left')
        rgb_front_right_path_base = os.path.join(camera_path, 'rgb_front_right')
        rgb_back_path_base = os.path.join(camera_path, 'rgb_back')
        rgb_back_left_path_base = os.path.join(camera_path, 'rgb_back_left')
        rgb_back_right_path_base = os.path.join(camera_path, 'rgb_back_right')

        depth_front_path_base = os.path.join(camera_path, 'depth_front')
        depth_front_left_path_base = os.path.join(camera_path, 'depth_front_left')
        depth_front_right_path_base = os.path.join(camera_path, 'depth_front_right')
        depth_back_path_base = os.path.join(camera_path, 'depth_back')
        depth_back_left_path_base = os.path.join(camera_path, 'depth_back_left')
        depth_back_right_path_base = os.path.join(camera_path, 'depth_back_right')

        bev_path_base = os.path.join(camera_path, 'rgb_top_down')
        
        # Check if required directories exist
        if not (os.path.exists(anno_path) and os.path.exists(camera_path)):
            logger.warning(f"Missing required directories in {scenario_dir}")
            return
            
        # Get frame files
        try:
            frame_files = [f for f in os.listdir(anno_path) if f.endswith('.json')]
            total_frames = len(frame_files)
            if total_frames < (self.traj_time_far * self.fps):
                logger.warning(f"Not enough frames in {scenario_dir}")
                return
        except Exception as e:
            logger.error(f"Error reading frames in {scenario_dir}: {e}")
            return
            
        # Process camera parameters for this scenario
        cam_para_processed = False
        intrinsic = None
        extrinsic = None
        
        # Process all valid frames in this scenario
        for frame_idx in range(0, total_frames - self.traj_time_far * self.fps):
            # Check frame validity (same as in ScenarioDataset)
            skip_frame = False
            for i in self.traj_all_indices:
                json_path = os.path.join(anno_path, f"{frame_idx + i:05d}.json")
                if not os.path.exists(json_path):
                    skip_frame = True
                    break
                try:
                    with open(json_path, 'r') as f:
                        json.load(f)
                except Exception:
                    skip_frame = True
                    break
            if skip_frame:
                continue
                
            # Load current frame data
            current_json_path = os.path.join(anno_path, f"{frame_idx:05d}.json")
            try:
                with open(current_json_path, 'r') as f:
                    current_data = json.load(f)
                self.frame_ids.append(os.path.join(scenario_dir, 'extracted_anno', f"{frame_idx:05d}"))
            except Exception as e:
                logger.error(f"Error loading {current_json_path}: {e}")
                continue

            # Ego state check
            if np.isnan(current_data['x']) or np.isnan(current_data['y']) or np.isnan(current_data['theta']):
                skip_frame = True
                logger.warning(f"Nan x or y or theta for frame {frame_idx} in {scenario_dir}")
                continue

            # BEV pixel per meter
            bev_cam_config = current_data["sensors"]["TOP_DOWN"]
            self.bev_pix_per_meter.append(calculate_pixel_per_meter(bev_cam_config))
                
            # Process camera parameters if not already done for this scenario
            if not cam_para_processed:
                cam_front_config = current_data["sensors"]["CAM_FRONT"]
                cam_front_left_config = current_data["sensors"]["CAM_FRONT_LEFT"]
                cam_front_right_config = current_data["sensors"]["CAM_FRONT_RIGHT"]
                cam_back_config = current_data["sensors"]["CAM_BACK"]
                
                intrinsic = torch.from_numpy(np.array([ 
                    my_update_intrinsics(cam_front_config['intrinsic'], self.cfg.image_scale, self.cfg.image_crop),
                    my_update_intrinsics(cam_front_left_config['intrinsic'], self.cfg.image_scale, self.cfg.image_crop),
                    my_update_intrinsics(cam_front_right_config['intrinsic'], self.cfg.image_scale, self.cfg.image_crop),
                    my_update_intrinsics(cam_back_config['intrinsic'], self.cfg.image_scale, self.cfg.image_crop),
                ], dtype=np.float32))
                
                extrinsic = torch.from_numpy(np.array([
                    cam_front_config['cam2ego'], 
                    cam_front_left_config['cam2ego'], 
                    cam_front_right_config['cam2ego'], 
                    cam_back_config['cam2ego']
                ], dtype=np.float32))
                
                cam_para_processed = True
                
            # Store camera parameters for this frame
            self.intrinsics.append(intrinsic)
            self.extrinsics.append(extrinsic)
            
            # Process the rest of the frame data (similar to ScenarioDataset)
            # ... (Copy the relevant parts from your ScenarioDataset.get_data method here)
            
            # Add the encoded description for this frame
            self.enco_descriptions.append(enco_description)
            self.ability_one_hot.append(ability_one_hot)
            
            # Image paths
            rgb_front_file = f"{frame_idx:05d}.jpg"
            rgb_front_full_path = os.path.join(rgb_front_path_base, rgb_front_file)
            self.rgb_front_paths.append(rgb_front_full_path)
        
            rgb_front_left_file = f"{frame_idx:05d}.jpg"
            rgb_front_left_full_path = os.path.join(rgb_front_left_path_base, rgb_front_left_file)
            self.rgb_front_left_paths.append(rgb_front_left_full_path)

            rgb_front_right_file = f"{frame_idx:05d}.jpg"
            rgb_front_right_full_path = os.path.join(rgb_front_right_path_base, rgb_front_right_file)
            self.rgb_front_right_paths.append(rgb_front_right_full_path)

            rgb_back_file = f"{frame_idx:05d}.jpg"
            rgb_back_full_path = os.path.join(rgb_back_path_base, rgb_back_file)
            self.rgb_back_paths.append(rgb_back_full_path)
        
            rgb_back_left_file = f"{frame_idx:05d}.jpg"
            rgb_back_left_full_path = os.path.join(rgb_back_left_path_base, rgb_back_left_file)
            self.rgb_back_left_paths.append(rgb_back_left_full_path)

            rgb_back_right_file = f"{frame_idx:05d}.jpg"
            rgb_back_right_full_path = os.path.join(rgb_back_right_path_base, rgb_back_right_file)
            self.rgb_back_right_paths.append(rgb_back_right_full_path)

            # Depth paths
            depth_front_file = f"{frame_idx:05d}.png"
            depth_front_full_path = os.path.join(depth_front_path_base, depth_front_file)
            self.depth_front_paths.append(depth_front_full_path)
        
            depth_front_left_file = f"{frame_idx:05d}.png"
            depth_front_left_full_path = os.path.join(depth_front_left_path_base, depth_front_left_file)
            self.depth_front_left_paths.append(depth_front_left_full_path)

            depth_front_right_file = f"{frame_idx:05d}.png"
            depth_front_right_full_path = os.path.join(depth_front_right_path_base, depth_front_right_file)
            self.depth_front_right_paths.append(depth_front_right_full_path)

            depth_back_file = f"{frame_idx:05d}.png"
            depth_back_full_path = os.path.join(depth_back_path_base, depth_back_file)
            self.depth_back_paths.append(depth_back_full_path)
        
            depth_back_left_file = f"{frame_idx:05d}.png"
            depth_back_left_full_path = os.path.join(depth_back_left_path_base, depth_back_left_file)
            self.depth_back_left_paths.append(depth_back_left_full_path)

            depth_back_right_file = f"{frame_idx:05d}.png"
            depth_back_right_full_path = os.path.join(depth_back_right_path_base, depth_back_right_file)
            self.depth_back_right_paths.append(depth_back_right_full_path)


            bev_file = f"{frame_idx:05d}.jpg"
            bev_full_path = os.path.join(bev_path_base, bev_file)

            if not (os.path.exists(rgb_front_full_path) and os.path.exists(bev_full_path)):
                logger.warning(f"Missing image file for frame {frame_idx} in {scenario_dir}")
                continue

            self.bev_paths.append(bev_full_path)

            # Ego state
            ego_x = current_data['x']
            ego_y = current_data['y']
            ego_z = 0.0
            self.ego_position.append(np.array([ego_x, ego_y], dtype=np.float32))
            ego_yaw_rad = current_data['theta'] - (math.pi/2)
            self.rad_yaw.append(ego_yaw_rad)
            ego_transform = carla.Transform(
                carla.Location(x=ego_x, y=ego_y, z=ego_z),
                carla.Rotation(roll=0, pitch=0, yaw=math.degrees(ego_yaw_rad)))
            ego_transform_matrix = np.array(ego_transform.get_matrix())

            # Motion state
            self.acceleration.append(np.array(current_data['acceleration'], dtype=np.float32))
            self.angular_velocity.append(np.array(current_data['angular_velocity'], dtype=np.float32))
            self.ego_motion.append(np.concatenate([np.atleast_1d(np.array(current_data['speed'], dtype=np.float32)), np.atleast_1d(np.array(current_data['theta'] - math.pi/2, dtype=np.float32)), 
                                                    np.array(current_data['acceleration'], dtype=np.float32), np.array(current_data['angular_velocity'], dtype=np.float32)], axis=0))

            # Target point
            target_x_world = current_data['x_command_near']
            target_y_world = current_data['y_command_near']
            target_point_ego = convert_veh_coord(target_x_world, target_y_world, 0.0, ego_transform)
            self.target_points.append(np.array([target_point_ego[0], target_point_ego[1]], dtype=np.float32))

            #Nav command
            nav_cmd = self._load_nav_cmd(current_data)
            self.nav_cmd.append(nav_cmd)

            # Trajectory points
            all_traj_points_ego = []
            near_traj_points_ego = []
            # All points
            for i in (self.traj_all_indices):
                traj_idx = frame_idx + i
                json_path = os.path.join(anno_path, f"{traj_idx:05d}.json")
                # try:
                #     with open(json_path, 'r') as f:
                #         data = json.load(f)
                #     point_ego = convert_veh_coord(data['x'], data['y'], 0.0, ego_transform)
                #     all_traj_points_ego.append([point_ego[0], point_ego[1]])
                #     # print(f'ego position: {self.ego_position[-1]}, yaw: {self.rad_yaw[-1]*(180/np.pi)}, wp_world: {[data["x"], data["y"]]}, wp_ego: {point_ego[0:2]}')
                # except Exception:
                #     all_traj_points_ego.append([0.0, 0.0])
                #     print('traj append 0,0')

                with open(json_path, 'r') as f:
                    data = json.load(f)
                point_ego = convert_veh_coord(data['x'], data['y'], 0.0, ego_transform)
                all_traj_points_ego.append([point_ego[0], point_ego[1]])
                # print(f'ego position: {self.ego_position[-1]}, yaw: {self.rad_yaw[-1]*(180/np.pi)}, wp_world: {[data["x"], data["y"]]}, wp_ego: {point_ego[0:2]}')


            # Near points
            for i in (self.traj_near_indices):
                near_traj_idx = frame_idx + i
                json_path = os.path.join(anno_path, f"{near_traj_idx:05d}.json")
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    point_ego = convert_veh_coord(data['x'], data['y'], 0.0, ego_transform)
                    near_traj_points_ego.append([point_ego[0], point_ego[1]])
                except Exception:
                    near_traj_points_ego.append([0.0, 0.0])

            self.all_trajectory_points.append(np.array(all_traj_points_ego, dtype=np.float32))
            self.near_trajectory_points.append(np.array(near_traj_points_ego, dtype=np.float32))
            self.near_traj_tokens.append(np.array(tokenize_traj(self.cfg.bev_x_bound[1], self.cfg.bev_y_bound[1], near_traj_points_ego), dtype=np.int64))

            # Control signals
            controls_tokens = []
            for i in self.traj_near_indices:
                control_idx = frame_idx + i
                future_json_path = os.path.join(anno_path, f"{control_idx:05d}.json")
                try:
                    with open(future_json_path, 'r') as f:
                        future_data = json.load(f)
                    tokens = tokenize_action(
                        future_data.get('throttle', 0.0),
                        future_data.get('brake', 0.0),
                        future_data.get('steer', 0.0),
                        1 if future_data.get('reverse', False) else 0,
                        self.token_nums
                    )
                    controls_tokens.extend(tokens)
                except Exception:
                    controls_tokens.extend([self.PAD_token - 4] * 3)

            # Add sequence markers
            controls_tokens.insert(0, self.BOS_token)
            controls_tokens.append(self.EOS_token)
            # Pad to fixed length
            target_len = len(self.traj_near_indices) * 3 + 2
            current_len = len(controls_tokens)
            if current_len < target_len:
                controls_tokens.extend([self.PAD_token] * (target_len - current_len))
            elif current_len > target_len:
                controls_tokens = controls_tokens[:target_len]
            self.control_tokens.append(np.array(controls_tokens, dtype=np.int64))

    def _one_hot_encode(self, command, num_classes=6):
        one_hot = torch.zeros(num_classes)
        one_hot[command-1] = 1
        return one_hot
    
    def _load_nav_cmd(self, data):
        cmd_far = self._one_hot_encode(data['command_far'])
        cmd_near  = self._one_hot_encode(data['command_near'])
        cmd_next = self._one_hot_encode(data['next_command'])
        cmd_list = [cmd_near, cmd_next, cmd_far]
        return torch.cat(cmd_list, dim=0) # 创建一维张量            

    def __len__(self):
        return len(self.rgb_front_paths)

    def __getitem__(self, index):
        # Similar to ScenarioDataset.__getitem__ but without the device assignment
        data = {}
        
        # Process images (only the 4 views we need)
        rgb_tensor_f, _ = self.image_process_rgb(self.rgb_front_paths[index])
        rgb_tensor_fl, _ = self.image_process_rgb(self.rgb_front_left_paths[index])
        rgb_tensor_fr, _ = self.image_process_rgb(self.rgb_front_right_paths[index])
        rgb_tensor_b, _ = self.image_process_rgb(self.rgb_back_paths[index])
        data['images'] = torch.cat([rgb_tensor_f, rgb_tensor_fl, rgb_tensor_fr, rgb_tensor_b], dim=0)
        
        # Process depths (only the 4 views we need)
        depth_tensor_f = get_depth(self.depth_front_paths[index])
        depth_tensor_fl = get_depth(self.depth_front_left_paths[index])
        depth_tensor_fr = get_depth(self.depth_front_right_paths[index])
        depth_tensor_b = get_depth(self.depth_back_paths[index])
        data['depths'] = torch.cat([depth_tensor_f, depth_tensor_fl, depth_tensor_fr, depth_tensor_b], dim=0)
        
        # Add other data
        data['all_trajectory_points'] = torch.from_numpy(self.all_trajectory_points[index])
        data['near_trajectory_points'] = torch.from_numpy(self.near_trajectory_points[index])
        data['flatten_all_trajectory_points'] = torch.from_numpy(self.all_trajectory_points[index].flatten())
        data['flatten_near_trajectory_points'] = torch.from_numpy(self.near_trajectory_points[index].flatten())
        data['target_point'] = torch.from_numpy(self.target_points[index])
        data['ego_motion'] = torch.from_numpy(self.ego_motion[index]).unsqueeze(0)
        data['gt_control'] = torch.from_numpy(self.control_tokens[index])
        data['enco_description'] = self.enco_descriptions[index]
        data['nav_cmd'] = self.nav_cmd[index]
        data['gt_near_trajectory'] = torch.from_numpy(self.near_traj_tokens[index])
        # data['surroundings'] = torch.from_numpy(self.surroundings[index])
        data['intrinsic'] = self.intrinsics[index]
        data['extrinsic'] = self.extrinsics[index]

        data['dpmm_data'] = torch.from_numpy(self.all_trajectory_points[index].flatten())

        data['ability_one_hot'] = torch.from_numpy(np.array(self.ability_one_hot[index]))
        
        return data


# Example usage
if __name__ == '__main__':
    # Load scenario skill descriptions
    with open('../text_enco/scen_skill_desc_list.pkl', 'rb') as f:
        scen_skill_desc_list = pickle.load(f)
    
    # Load config
    with open('../config/my_training.yaml') as yaml_file:
        cfg_yaml = yaml.safe_load(yaml_file)
    config = get_cfg(cfg_yaml)
    
    # Create dataset for a specific ability
    data_root = "../../b2d_143_train"
    ability = 'EmergencyBrake'
    
    dataset = AbilityDataset(
        data_root=data_root,
        ability=ability,
        is_train=True,
        config=config,
        scen_skill_desc_list=scen_skill_desc_list
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )
    
    # Now you can iterate through the dataloader
    # for batch in dataloader:
    #     # Your training code here
    #     pass

    print('checking the ability dataloader')
    idx = 100
    print(f'dpmm_data: {dataloader.dataset[idx]["flatten_all_trajectory_points"]}')

    print(f'target_point: {dataloader.dataset[idx]["target_point"].shape}')

    print(f'ego_motion: {dataloader.dataset[idx]["ego_motion"].shape}')

    print(f'gt_near_trajectory (tokenized traj): {dataloader.dataset[idx]["gt_near_trajectory"].shape}')
    print(f'flatten_near_trajectory: {dataloader.dataset[idx]["near_trajectory_points"].shape}')

    print(f'images: {dataloader.dataset[idx]["images"].shape}')

    print(f'depths: {dataloader.dataset[idx]["depths"].shape}')

    print(f'intrinsic: {dataloader.dataset[idx]["intrinsic"].shape}')
    print(f'extrinsic: {dataloader.dataset[idx]["extrinsic"].shape}')