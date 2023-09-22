import os
import math
import cv2

import random
import mmcv
import numpy as np
import torch
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from nuscenes.utils.data_classes import Box
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset

__all__ = ['NuscMVDetDataset']

map_name_from_general_to_detection = {
    'vehicle.car': 'Vehicle',
    'vehicle.truck': 'Vehicle',
    'vehicle.van': 'Vehicle',
    'vehicle.bus': 'Vehicle', 
    'vehicle.cyclist': 'Cyclist',
    'vehicle.tricyclist': 'Cyclist',
    'vehicle.motorcyclist': 'Cyclist',
    'vehicle.barrowlist': 'Cyclist',
    'human.pedestrian.adult': 'Pedestrian',       
}

map_name_from_detection_to_prompt_class = {
    'Vehicle': 1,
    'Cyclist': 2,
    'Pedestrian': 3,
}

def equation_plane(points): 
    x1, y1, z1 = points[0, 0], points[0, 1], points[0, 2]
    x2, y2, z2 = points[1, 0], points[1, 1], points[1, 2]
    x3, y3, z3 = points[2, 0], points[2, 1], points[2, 2]
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return np.array([a, b, c, d])

def get_denorm(sweepego2sweepsensor):
    ground_points_lidar = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    ground_points_lidar = np.concatenate((ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), axis=1)
    ground_points_cam = np.matmul(sweepego2sweepsensor, ground_points_lidar.T).T
    denorm = -1 * equation_plane(ground_points_cam)
    return denorm

def get_sensor2virtual(denorm):
    origin_vector = np.array([0, 1, 0])    
    target_vector = -1 * np.array([denorm[0], denorm[1], denorm[2]])
    target_vector_norm = target_vector / np.sqrt(target_vector[0]**2 + target_vector[1]**2 + target_vector[2]**2)       
    sita = math.acos(np.inner(target_vector_norm, origin_vector))
    n_vector = np.cross(target_vector_norm, origin_vector) 
    n_vector = n_vector / np.sqrt(n_vector[0]**2 + n_vector[1]**2 + n_vector[2]**2)
    n_vector = n_vector.astype(np.float32)
    rot_mat, _ = cv2.Rodrigues(n_vector * sita)
    rot_mat = rot_mat.astype(np.float32)
    sensor2virtual = np.eye(4)
    sensor2virtual[:3, :3] = rot_mat
    return sensor2virtual.astype(np.float32)

def get_reference_height(denorm):
    ref_height = np.abs(denorm[3]) / np.sqrt(denorm[0]**2 + denorm[1]**2 + denorm[2]**2)
    return ref_height.astype(np.float32)

def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

def img_intrin_extrin_transform(img, ratio, roll, transform_pitch, intrin_mat):
    center = intrin_mat[:2, 2].astype(np.int32) 
    center = (int(center[0]), int(center[1]))

    W, H = img.size[0], img.size[1]
    new_W, new_H = (int(W * ratio), int(H * ratio))
    img = img.resize((new_W, new_H), Image.ANTIALIAS)
    
    h_min = int(center[1] * abs(1.0 - ratio))
    w_min = int(center[0] * abs(1.0 - ratio))
    if ratio <= 1.0:
        image = Image.new(mode='RGB', size=(W, H))
        image.paste(img, (w_min, h_min,  w_min + new_W, h_min + new_H))
    else:
        image = img.crop((w_min, h_min,  w_min + W, h_min + H))
    img = image.rotate(-roll, expand=0, center=center, translate=(0, transform_pitch), fillcolor=(0,0,0), resample=Image.BICUBIC)
    return img

def img_transform(img, resize, resize_dims, crop, flip, rotate):
    ida_rot = torch.eye(2)
    ida_tran = torch.zeros(2)
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    ida_rot *= resize
    ida_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
    A = get_rot(rotate / 180 * np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    ida_rot = A.matmul(ida_rot)
    ida_tran = A.matmul(ida_tran) + b
    ida_mat = ida_rot.new_zeros(4, 4)
    ida_mat[3, 3] = 1
    ida_mat[2, 2] = 1
    ida_mat[:2, :2] = ida_rot
    ida_mat[:2, 3] = ida_tran
    return img, ida_mat


def bev_transform(gt_boxes, rotate_angle, scale_ratio, flip_dx, flip_dy):
    rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
    rot_sin = torch.sin(rotate_angle)
    rot_cos = torch.cos(rotate_angle)
    rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                            [0, 0, 1]])
    scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                              [0, 0, scale_ratio]])
    flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dx:
        flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dy:
        flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    rot_mat = flip_mat @ (scale_mat @ rot_mat)
    if gt_boxes.shape[0] > 0:
        gt_boxes[:, :3] = (rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
        gt_boxes[:, 3:6] *= scale_ratio
        gt_boxes[:, 6] += rotate_angle
        if flip_dx:
            gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
        if flip_dy:
            gt_boxes[:, 6] = -gt_boxes[:, 6]
        gt_boxes[:, 7:] = (
            rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
    return gt_boxes, rot_mat


def depth_transform(cam_depth, resize, resize_dims, crop, flip, rotate):
    """Transform depth based on ida augmentation configuration.

    Args:
        cam_depth (np array): Nx3, 3: x,y,d.
        resize (float): Resize factor.
        resize_dims (list): Final dimension.
        crop (list): x1, y1, x2, y2
        flip (bool): Whether to flip.
        rotate (float): Rotation value.

    Returns:
        np array: [h/down_ratio, w/down_ratio, d]
    """

    H, W = resize_dims
    cam_depth[:, :2] = cam_depth[:, :2] * resize
    cam_depth[:, 0] -= crop[0]
    cam_depth[:, 1] -= crop[1]
    if flip:
        cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

    cam_depth[:, 0] -= W / 2.0
    cam_depth[:, 1] -= H / 2.0

    h = rotate / 180 * np.pi
    rot_matrix = [
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ]
    cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

    cam_depth[:, 0] += W / 2.0
    cam_depth[:, 1] += H / 2.0

    depth_coords = cam_depth[:, :2].astype(np.int16)

    depth_map = np.zeros(resize_dims)
    valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                  & (depth_coords[:, 0] < resize_dims[1])
                  & (depth_coords[:, 1] >= 0)
                  & (depth_coords[:, 0] >= 0))
    depth_map[depth_coords[valid_mask, 1],
              depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

    return torch.Tensor(depth_map)


class NuscMVDetDataset(Dataset):
    def __init__(self,
                 ida_aug_conf,
                 classes,
                 data_root,
                 info_path,
                 is_train,
                 use_cbgs=False,
                 num_sweeps=1,
                 img_conf=dict(img_mean=[123.675, 116.28, 103.53],
                               img_std=[58.395, 57.12, 57.375],
                               to_rgb=True),
                 return_depth=False,
                 sweep_idxes=list(),
                 key_idxes=list()):
        """Dataset used for bevdetection task.
        Args:
            ida_aug_conf (dict): Config for ida augmentation.
            classes (list): Class names.
            use_cbgs (bool): Whether to use cbgs strategy,
                Default: False.
            num_sweeps (int): Number of sweeps to be used for each sample.
                default: 1.
            img_conf (dict): Config for image.
            return_depth (bool): Whether to use depth gt.
                default: False.
            sweep_idxes (list): List of sweep idxes to be used.
                default: list().
            key_idxes (list): List of key idxes to be used.
                default: list().
        """
        super().__init__()
        self.infos = mmcv.load(info_path)
        self.is_train = is_train
        self.ida_aug_conf = ida_aug_conf
        self.data_root = data_root
        self.classes = classes
        self.use_cbgs = use_cbgs
        if self.use_cbgs:
            self.cat2id = {name: i for i, name in enumerate(self.classes)}
            self.sample_indices = self._get_sample_indices()
        self.num_sweeps = num_sweeps
        self.img_mean = np.array(img_conf['img_mean'], np.float32)
        self.img_std = np.array(img_conf['img_std'], np.float32)
        self.to_rgb = img_conf['to_rgb']
        self.return_depth = return_depth
        assert sum([sweep_idx >= 0 for sweep_idx in sweep_idxes]) \
            == len(sweep_idxes), 'All `sweep_idxes` must greater \
                than or equal to 0.'

        self.sweeps_idx = sweep_idxes
        assert sum([key_idx < 0 for key_idx in key_idxes]) == len(key_idxes),\
            'All `key_idxes` must less than 0.'
        self.key_idxes = [0] + key_idxes

        self.ratio_range = [1.0, 0.20]
        self.roll_range = [0.0, 2.00]
        self.pitch_range = [0.0, 0.67]

    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx, info in enumerate(self.infos):
            gt_names = set(
                [ann_info['category_name'] for ann_info in info['ann_infos']])
            for gt_name in gt_names:
                gt_name = map_name_from_general_to_detection[gt_name]
                if gt_name not in self.classes:
                    continue
                class_sample_idxs[self.cat2id[gt_name]].append(idx)
        duplicated_samples = sum(
            [len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_idxs.items()
        }

        sample_indices = []
        frac = 1.0 / len(self.classes)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(cls_inds,
                                               int(len(cls_inds) *
                                                   ratio)).tolist()
        return sample_indices

    def degree2rad(self, degree):
        return degree * np.pi / 180

    def get_M(self, R, K, R_r, K_r):
        R_inv = np.linalg.inv(R)
        K_inv = np.linalg.inv(K)
        M = np.matmul(K_r, R_r)
        M = np.matmul(M, R_inv)
        M = np.matmul(M, K_inv)
        return M

    def sample_intrin_extrin_augmentation(self, intrin_mat, sweepego2sweepsensor):
        intrin_mat, sweepego2sweepsensor = intrin_mat.numpy(), sweepego2sweepsensor.numpy()
        # rectify intrin_mat
        ratio = np.random.normal(self.ratio_range[0], self.ratio_range[1])
        intrin_mat_rectify = intrin_mat.copy()
        intrin_mat_rectify[:2,:2] = intrin_mat[:2,:2] * ratio
        
        # rectify sweepego2sweepsensor by roll
        roll = np.random.normal(self.roll_range[0], self.roll_range[1])
        roll_rad = self.degree2rad(roll)
        rectify_roll = np.array([[math.cos(roll_rad), -math.sin(roll_rad), 0, 0], 
                                 [math.sin(roll_rad), math.cos(roll_rad), 0, 0], 
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
        sweepego2sweepsensor_rectify_roll = np.matmul(rectify_roll, sweepego2sweepsensor)
        
        # rectify sweepego2sweepsensor by pitch
        pitch = np.random.normal(self.pitch_range[0], self.pitch_range[1])
        pitch_rad = self.degree2rad(pitch)
        rectify_pitch = np.array([[1, 0, 0, 0],
                                  [0,math.cos(pitch_rad), -math.sin(pitch_rad), 0], 
                                  [0,math.sin(pitch_rad), math.cos(pitch_rad), 0],
                                  [0, 0, 0, 1]])
        sweepego2sweepsensor_rectify_pitch = np.matmul(rectify_pitch, sweepego2sweepsensor_rectify_roll)
        M = self.get_M(sweepego2sweepsensor_rectify_roll[:3,:3], intrin_mat_rectify[:3,:3], sweepego2sweepsensor_rectify_pitch[:3,:3], intrin_mat_rectify[:3,:3])
        center = intrin_mat_rectify[:2, 2]  # w, h
        center_ref = np.array([center[0], center[1], 1.0])
        center_ref = np.matmul(M, center_ref.T)[:2]
        transform_pitch = int(center_ref[1] - center[1])

        intrin_mat_rectify, sweepego2sweepsensor_rectify = torch.Tensor(intrin_mat_rectify), torch.Tensor(sweepego2sweepsensor_rectify_pitch)
        return intrin_mat_rectify, sweepego2sweepsensor_rectify, ratio, roll, transform_pitch

    def sample_ida_augmentation(self):
        """Generate ida augmentation values based on ida_config."""
        H, W = self.ida_aug_conf['H'], self.ida_aug_conf['W']
        fH, fW = self.ida_aug_conf['final_dim']
        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int(
            (1 - np.mean(self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate_ida = 0
        return resize, resize_dims, crop, flip, rotate_ida

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        rotate_bda = 0
        scale_bda = 1.0
        flip_dx = False
        flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy
    
    def get_image(self, cam_infos, cams):
        """Given data and cam_names, return image data needed.

        Args:
            sweeps_data (list): Raw data used to generate the data we needed.
            cams (list): Camera names.

        Returns:
            Tensor: Image data after processing.
            Tensor: Transformation matrix from camera to ego.
            Tensor: Intrinsic matrix.
            Tensor: Transformation matrix for ida.
            Tensor: Transformation matrix from key
                frame camera to sweep frame camera.
            Tensor: timestamps.
            dict: meta infos needed for evaluation.
        """
        assert len(cam_infos) > 0
        sweep_imgs = list()
        sweep_imgs_ori = list()
        sweep_sensor2ego_mats = list()
        sweep_intrin_mats = list()
        sweep_ida_mats = list()
        sweep_sensor2sensor_mats = list()
        sweep_sensor2virtual_mats = list()
        sweep_timestamps = list()
        sweep_reference_heights = list()
        gt_depth = list()
        for cam in cams:
            imgs = list()
            imgs_ori = list()
            sensor2ego_mats = list()
            intrin_mats = list()
            ida_mats = list()
            sensor2sensor_mats = list()
            sensor2virtual_mats=list()
            reference_heights = list()
            timestamps = list()
            key_info = cam_infos[0]
            resize, resize_dims, crop, flip, \
                rotate_ida = self.sample_ida_augmentation(
                    )
            for sweep_idx, cam_info in enumerate(cam_infos):
                img = Image.open(
                    os.path.join(self.data_root, cam_info[cam]['filename']))
                if "rotation_matrix" in cam_info[cam]['calibrated_sensor'].keys():
                    sweepsensor2sweepego_rot = torch.Tensor(cam_info[cam]['calibrated_sensor']['rotation_matrix'])
                else:
                    w, x, y, z = cam_info[cam]['calibrated_sensor']['rotation']
                    # sweep sensor to sweep ego
                    sweepsensor2sweepego_rot = torch.Tensor(
                        Quaternion(w, x, y, z).rotation_matrix)
                sweepsensor2sweepego_tran = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['translation'])
                sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros(
                    (4, 4))
                sweepsensor2sweepego[3, 3] = 1
                sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
                sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
                
                sweepego2sweepsensor = sweepsensor2sweepego.inverse()
                denorm = get_denorm(sweepego2sweepsensor.numpy())
                # sweep ego to global
                w, x, y, z = cam_info[cam]['ego_pose']['rotation']
                sweepego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                sweepego2global_tran = torch.Tensor(
                    cam_info[cam]['ego_pose']['translation'])
                sweepego2global = sweepego2global_rot.new_zeros((4, 4))
                sweepego2global[3, 3] = 1
                sweepego2global[:3, :3] = sweepego2global_rot
                sweepego2global[:3, -1] = sweepego2global_tran
                
                intrin_mat = torch.zeros((4, 4))
                intrin_mat[3, 3] = 1
                intrin_mat[:3, :3] = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['camera_intrinsic'])
                sweepego2sweepsensor = sweepsensor2sweepego.inverse()
                
                if self.is_train and random.random() < 0.5:
                    intrin_mat, sweepego2sweepsensor, ratio, roll, transform_pitch = self.sample_intrin_extrin_augmentation(intrin_mat, sweepego2sweepsensor)
                    img = img_intrin_extrin_transform(img, ratio, roll, transform_pitch, intrin_mat.numpy())
                
                denorm = get_denorm(sweepego2sweepsensor.numpy())
                sweepsensor2sweepego = sweepego2sweepsensor.inverse()

                # global sensor to cur ego
                w, x, y, z = key_info[cam]['ego_pose']['rotation']
                keyego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                keyego2global_tran = torch.Tensor(
                    key_info[cam]['ego_pose']['translation'])
                keyego2global = keyego2global_rot.new_zeros((4, 4))
                keyego2global[3, 3] = 1
                keyego2global[:3, :3] = keyego2global_rot
                keyego2global[:3, -1] = keyego2global_tran
                global2keyego = keyego2global.inverse()

                # cur ego to sensor
                if "rotation_matrix" in key_info[cam]['calibrated_sensor'].keys():
                    keysensor2keyego_rot = torch.Tensor(key_info[cam]['calibrated_sensor']['rotation_matrix'])
                else:
                    w, x, y, z = key_info[cam]['calibrated_sensor']['rotation']                    
                    keysensor2keyego_rot = torch.Tensor(
                        Quaternion(w, x, y, z).rotation_matrix)
                keysensor2keyego_tran = torch.Tensor(
                    key_info[cam]['calibrated_sensor']['translation'])
                keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
                keysensor2keyego[3, 3] = 1
                keysensor2keyego[:3, :3] = keysensor2keyego_rot
                keysensor2keyego[:3, -1] = keysensor2keyego_tran
                keyego2keysensor = keysensor2keyego.inverse()
                keysensor2sweepsensor = (
                    keyego2keysensor @ global2keyego @ sweepego2global
                    @ sweepsensor2sweepego).inverse()
                sweepsensor2keyego = global2keyego @ sweepego2global @\
                    sweepsensor2sweepego
                sensor2virtual = torch.Tensor(get_sensor2virtual(denorm))
                sensor2ego_mats.append(sweepsensor2keyego)
                sensor2sensor_mats.append(keysensor2sweepsensor)
                sensor2virtual_mats.append(sensor2virtual)

                if self.return_depth and sweep_idx == 0:
                    file_name = os.path.split(cam_info[cam]['filename'])[-1]
                    point_depth = np.fromfile(os.path.join(
                        self.data_root, 'depth_gt', f'{file_name}.bin'),
                                              dtype=np.float32,
                                              count=-1).reshape(-1, 3)
                    point_depth_augmented = depth_transform(
                        point_depth, resize, self.ida_aug_conf['final_dim'],
                        crop, flip, rotate_ida)
                    gt_depth.append(point_depth_augmented)
                img, ida_mat = img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate_ida,
                )
                img_ori = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
                img_ori = torch.from_numpy(img_ori).permute(2, 0, 1)
                imgs_ori.append(img_ori)                
                ida_mats.append(ida_mat)
                img = mmcv.imnormalize(np.array(img), self.img_mean,
                                       self.img_std, self.to_rgb)
                img = torch.from_numpy(img).permute(2, 0, 1)
                imgs.append(img)
                intrin_mats.append(intrin_mat)
                timestamps.append(cam_info[cam]['timestamp'])
                reference_heights.append(get_reference_height(denorm))
                
            sweep_imgs.append(torch.stack(imgs))
            sweep_imgs_ori.append(torch.stack(imgs_ori))
            sweep_sensor2ego_mats.append(torch.stack(sensor2ego_mats))
            sweep_intrin_mats.append(torch.stack(intrin_mats))
            sweep_ida_mats.append(torch.stack(ida_mats))
            sweep_sensor2sensor_mats.append(torch.stack(sensor2sensor_mats))
            sweep_sensor2virtual_mats.append(torch.stack(sensor2virtual_mats))
            sweep_timestamps.append(torch.tensor(timestamps))
            sweep_reference_heights.append(torch.tensor(reference_heights))
            
        # Get mean pose of all cams.
        ego2global_rotation = np.mean(
            [key_info[cam]['ego_pose']['rotation'] for cam in cams], 0)
        ego2global_translation = np.mean(
            [key_info[cam]['ego_pose']['translation'] for cam in cams], 0)
        img_metas = dict(
            box_type_3d=LiDARInstance3DBoxes,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
        )

        ret_list = [
            torch.stack(sweep_imgs_ori).permute(1, 0, 2, 3, 4),
            torch.stack(sweep_imgs).permute(1, 0, 2, 3, 4),
            torch.stack(sweep_sensor2ego_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_intrin_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_ida_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_sensor2sensor_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_sensor2virtual_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_timestamps).permute(1, 0),
            torch.stack(sweep_reference_heights).permute(1, 0),
            img_metas,
        ]
        if self.return_depth:
            ret_list.append(torch.stack(gt_depth))
        return ret_list

    def get_gt(self, info, cams):
        """Generate gt labels from info.

        Args:
            info(dict): Infos needed to generate gt labels.
            cams(list): Camera names.

        Returns:
            Tensor: GT bboxes.
            Tensor: GT labels.
        """
        ego2global_rotation = np.mean(
            [info['cam_infos'][cam]['ego_pose']['rotation'] for cam in cams],
            0)
        ego2global_translation = np.mean([
            info['cam_infos'][cam]['ego_pose']['translation'] for cam in cams
        ], 0)
        trans = -np.array(ego2global_translation)
        rot = Quaternion(ego2global_rotation).inverse
        gt_boxes = list()
        gt_labels = list()
        gt_classes = list()
        gt_corners = list()        
        for ann_info in info['ann_infos']:
            # Use ego coordinate.
            if (map_name_from_general_to_detection[ann_info['category_name']]
                    not in self.classes
                    or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <=
                    0):
                continue
            box = Box(
                ann_info['translation'],
                ann_info['size'][[1, 0, 2]],
                Quaternion(ann_info['rotation']),
                velocity=ann_info['velocity'],
            )
            box.translate(trans)
            box.rotate(rot)
            box_xyz = np.array(box.center)
            box_dxdydz = np.array(box.wlh)
            box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
            box_velo = np.array(box.velocity[:2])
            gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
            gt_corner = np.array(box.corners().transpose(1, 0))
            gt_boxes.append(gt_box)
            gt_corners.append(gt_corner)
            gt_labels.append(
                self.classes.index(map_name_from_general_to_detection[
                    ann_info['category_name']]))
            gt_classes.append(
                    map_name_from_detection_to_prompt_class[map_name_from_general_to_detection[
                    ann_info['category_name']]])
        gt_classes = torch.tensor(gt_classes)
        gt_classes = gt_classes.expand(512, gt_classes.shape[0]).permute(1, 0).unsqueeze(1)           
        return torch.Tensor(gt_boxes), torch.tensor(gt_labels), gt_classes, torch.tensor(gt_corners)

    def get_2d_from_2D_detector(self, frame_name):
        self.result_2d_dir = '/home/zjlab/wsq/detrex/results/labels21/'
        score_2d_thresh_dict = {
            '0': 0.4,
            '1': 0.4,
            '2': 0.4
        }
        gt_labels, gt_classes, gt_boxes_2d = list(), list(), list()
        gt_scores = list()
        file_flag = os.path.exists(self.result_2d_dir+frame_name+'.txt')
        if file_flag == False:
            print("Not 2d file: ", self.result_2d_dir+frame_name+'.txt')
            return torch.zeros([0, ], dtype=torch.float32), torch.zeros([0, ], dtype=torch.float32), torch.zeros([0, 4], dtype=torch.float32)
        with open(self.result_2d_dir+frame_name+'.txt', 'r') as f:
            objs_list = f.readlines()
            for aid, objs in enumerate(objs_list):
                temp_list = objs.split(' ')
                cls_id, x1, y1, x2, y2, score = temp_list
                score = float(score[:-1])
                if score < score_2d_thresh_dict[cls_id]:
                    continue
                gt_labels.append(int(cls_id))
                gt_classes.append(int(cls_id)+1)
                gt_boxes_2d.append([float(x1)*0.8, float(y1)*0.8, float(x2)*0.8, float(y2)*0.8])
                gt_scores.append(score)
        gt_labels = torch.LongTensor(gt_labels)
        gt_classes = torch.LongTensor(gt_classes)
        gt_classes = gt_classes.expand(512, gt_classes.shape[0]).permute(1, 0).unsqueeze(1)
        gt_boxes_2d = torch.Tensor(gt_boxes_2d)
        gt_scores = torch.Tensor(gt_scores)
        return gt_labels, gt_classes, gt_boxes_2d, gt_scores

    def lidar2img(self, points_lidar, sweep_sensor2ego_mats, sweep_intrins, sweep_ida_mats):
        points_lidar_homogeneous = \
            np.concatenate([points_lidar,
                            np.ones((points_lidar.shape[0], 1),
                                    dtype=points_lidar.dtype)], axis=1)
        camera2lidar = sweep_sensor2ego_mats
        lidar2camera = np.linalg.inv(camera2lidar)
        points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
        points_camera = points_camera_homogeneous[:, :3]  
            
        valid = np.ones((points_camera.shape[0]), dtype=bool)
        valid = np.logical_and(points_camera[:, -1] > 0.5, valid)

        points_camera = points_camera / points_camera[:, 2:3]
        camera2img = sweep_intrins
        points_img = points_camera @ camera2img.T
        points_img = points_img[:, :2]
        
        points_img = points_img @ sweep_ida_mats.T
        return points_img, valid

    def check_point_in_img(self, points, height, width):
        valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
        valid = np.logical_and(valid,
                            np.logical_and(points[:, 0] < width,
                                           points[:, 1] < height))
        return valid

    def choose_cams(self):
        """Choose cameras randomly.

        Returns:
            list: Cameras to be used.
        """
        if self.is_train and self.ida_aug_conf['Ncams'] < len(
                self.ida_aug_conf['cams']):
            cams = np.random.choice(self.ida_aug_conf['cams'],
                                    self.ida_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.ida_aug_conf['cams']
        return cams

    def __getitem__(self, idx):
        if self.use_cbgs:
            idx = self.sample_indices[idx]
        cam_infos = list()
        # TODO: Check if it still works when number of cameras is reduced.
        cams = self.choose_cams()
        for key_idx in self.key_idxes:
            cur_idx = key_idx + idx
            # Handle scenarios when current idx doesn't have previous key
            # frame or previous key frame is from another scene.
            if cur_idx < 0:
                cur_idx = idx
            elif self.infos[cur_idx]['scene_token'] != self.infos[idx][
                    'scene_token']:
                cur_idx = idx
            info = self.infos[cur_idx]
            cam_infos.append(info['cam_infos'])
            for sweep_idx in self.sweeps_idx:
                if len(info['sweeps']) == 0:
                    cam_infos.append(info['cam_infos'])
                else:
                    # Handle scenarios when current sweep doesn't have all
                    # cam keys.
                    for i in range(min(len(info['sweeps']) - 1, sweep_idx), -1,
                                   -1):
                        if sum([cam in info['sweeps'][i]
                                for cam in cams]) == len(cams):
                            cam_infos.append(info['sweeps'][i])
                            break
        image_data_list = self.get_image(cam_infos, cams)
        ret_list = list()
        (
            sweep_imgs_ori,
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            sweep_sensor2virtual_mats,
            sweep_timestamps,
            sweep_reference_heights,
            img_metas,
        ) = image_data_list[:10]
        img_metas['token'] = self.infos[idx]['sample_token']
        if self.is_train:
            gt_boxes, gt_labels, gt_classes, gt_corners = self.get_gt(self.infos[idx], cams)
            
            # Transform 3D box to 2D Box
            gt_2d_list = list()
            gt_flag = np.zeros([gt_boxes.shape[0]], dtype=np.bool)
            gt_corners = gt_corners.numpy().astype('float32').reshape(-1, 3)
            sweep_imgs_ori_array = sweep_imgs_ori.squeeze().numpy().transpose(1, 2, 0).copy()
            sweep_sensor2ego_mats_array = sweep_sensor2ego_mats.squeeze().numpy()
            sweep_intrins_array = sweep_intrins.squeeze().numpy()
            sweep_ida_mats_array = sweep_ida_mats.squeeze().numpy()
            corners_img_gt, valid_z_gt = self.lidar2img(gt_corners, sweep_sensor2ego_mats_array, 
                                                        sweep_intrins_array[:3, :3], sweep_ida_mats_array[:2, :2])
            valid_shape_gt = self.check_point_in_img(corners_img_gt, sweep_imgs_ori_array.shape[0], sweep_imgs_ori_array.shape[1])     
            valid_all_gt = np.logical_and(valid_z_gt, valid_shape_gt)      
            valid_z_gt = valid_z_gt.reshape(-1, 8)
            valid_shape_gt = valid_shape_gt.reshape(-1, 8)
            valid_all_gt = valid_all_gt.reshape(-1, 8)
            corners_img_gt = corners_img_gt.reshape(-1, 8, 2).astype(np.int)
            for aid in range(valid_all_gt.shape[0]):              
                if valid_z_gt[aid].sum() >= 1:
                    min_col = max(min(corners_img_gt[aid, valid_z_gt[aid], 0].min(), sweep_imgs_ori_array.shape[1]), 0)
                    max_col = max(min(corners_img_gt[aid, valid_z_gt[aid], 0].max(), sweep_imgs_ori_array.shape[1]), 0)
                    min_row = max(min(corners_img_gt[aid, valid_z_gt[aid], 1].min(), sweep_imgs_ori_array.shape[0]), 0)
                    max_row = max(min(corners_img_gt[aid, valid_z_gt[aid], 1].max(), sweep_imgs_ori_array.shape[0]), 0)                  
                    if (max_col - min_col) == 0 or (max_row - min_row) == 0:
                        continue     
                    gt_flag[aid] = True 
                    gt_2d_list.append([min_col, min_row, max_col, max_row])

            gt_boxes = gt_boxes[gt_flag]
            gt_labels = gt_labels[gt_flag]
            gt_classes = gt_classes[gt_flag]
            gt_boxes_2d = torch.Tensor(gt_2d_list)
            if gt_boxes.shape[0] == 0:
                gt_boxes = sweep_imgs.new_zeros(0, 7)
                gt_labels = sweep_imgs.new_zeros(0, )
                gt_classes = sweep_imgs.new_zeros(0, )
                gt_boxes_2d = sweep_imgs.new_zeros(0, 4)
            else:                      
                gt_boxes = gt_boxes
                gt_labels = gt_labels
                gt_classes = gt_classes
                gt_boxes_2d = gt_boxes_2d
        # Temporary solution for test.
        else:
            # Get from 2D Detections
            img_name = img_metas['token'].split('/')[1]
            frame_name = img_name.split('.jpg')[0]
            sweep_imgs_ori_array = sweep_imgs_ori.squeeze().numpy().transpose(1, 2, 0).copy()
            dt_labels, dt_classes, dt_boxes_2d, dt_scores = self.get_2d_from_2D_detector(frame_name)
            dt_boxes = sweep_imgs.new_zeros(0, 7)
            if dt_labels.shape[0] == 0:
                gt_boxes = sweep_imgs.new_zeros(0, 7)
                gt_labels = sweep_imgs.new_zeros(0, )
                gt_classes = sweep_imgs.new_zeros(0, )
                gt_boxes_2d = sweep_imgs.new_zeros(0, 4)
                gt_scores = sweep_imgs.new_zeros(0, )
            else:                   
                gt_boxes = dt_boxes
                gt_labels = dt_labels
                gt_classes = dt_classes
                gt_boxes_2d = dt_boxes_2d
                gt_scores = dt_scores
        
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(
        )

        bda_mat = sweep_imgs.new_zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = bev_transform(gt_boxes, rotate_bda, scale_bda,
                                          flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        ret_list = [
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            sweep_sensor2virtual_mats,
            bda_mat,
            sweep_timestamps,
            sweep_reference_heights,
            img_metas,
            gt_boxes,
            gt_labels,
            gt_classes,
            gt_boxes_2d,            
        ]
        if self.return_depth:
            ret_list.append(image_data_list[9])
        return ret_list

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: \
            {"train" if self.is_train else "val"}.
                    Augmentation Conf: {self.ida_aug_conf}"""

    def __len__(self):
        if self.use_cbgs:
            return len(self.sample_indices)
        else:
            return len(self.infos)

def collate_fn(data, is_return_depth=False):
    imgs_batch = list()
    sensor2ego_mats_batch = list()
    intrin_mats_batch = list()
    ida_mats_batch = list()
    sensor2sensor_mats_batch = list()
    sensor2virtual_mats_batch = list()
    bda_mat_batch = list()
    timestamps_batch = list()
    reference_heights_batch = list()
    gt_boxes_batch = list()
    gt_labels_batch = list()
    gt_classes_batch = list()
    gt_boxes_2d_batch = list()    
    img_metas_batch = list()
    depth_labels_batch = list()
    for iter_data in data:
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            sweep_sensor2virtual_mats,
            bda_mat,
            sweep_timestamps,
            sweep_reference_heights,
            img_metas,
            gt_boxes,
            gt_labels,
            gt_classes,
            gt_boxes_2d,            
        ) = iter_data[:14]
        if is_return_depth:
            gt_depth = iter_data[14]
            depth_labels_batch.append(gt_depth)
        imgs_batch.append(sweep_imgs)
        sensor2ego_mats_batch.append(sweep_sensor2ego_mats)
        intrin_mats_batch.append(sweep_intrins)
        ida_mats_batch.append(sweep_ida_mats)
        sensor2sensor_mats_batch.append(sweep_sensor2sensor_mats)
        sensor2virtual_mats_batch.append(sweep_sensor2virtual_mats)
        bda_mat_batch.append(bda_mat)
        timestamps_batch.append(sweep_timestamps)
        reference_heights_batch.append(sweep_reference_heights)
        img_metas_batch.append(img_metas)
        gt_boxes_batch.append(gt_boxes)
        gt_labels_batch.append(gt_labels)
        gt_classes_batch.append(gt_classes)
        gt_boxes_2d_batch.append(gt_boxes_2d)        
    mats_dict = dict()
    mats_dict['sensor2ego_mats'] = torch.stack(sensor2ego_mats_batch)
    mats_dict['intrin_mats'] = torch.stack(intrin_mats_batch)
    mats_dict['ida_mats'] = torch.stack(ida_mats_batch)
    mats_dict['reference_heights'] = torch.stack(reference_heights_batch)
    mats_dict['sensor2sensor_mats'] = torch.stack(sensor2sensor_mats_batch)
    mats_dict['sensor2virtual_mats'] = torch.stack(sensor2virtual_mats_batch)
    mats_dict['bda_mat'] = torch.stack(bda_mat_batch)
    mats_dict['prompt_2d'] = gt_boxes_2d_batch
    mats_dict['prompt_class'] = gt_classes_batch    
    ret_list = [
        torch.stack(imgs_batch),
        mats_dict,
        torch.stack(timestamps_batch),
        img_metas_batch,
        gt_boxes_batch,
        gt_labels_batch,
    ]
    if is_return_depth:
        ret_list.append(torch.stack(depth_labels_batch))
    return ret_list