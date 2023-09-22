import os
import math
import json
import mmcv
import cv2
import torch
from PIL import Image

import numpy as np
from pyquaternion import Quaternion
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from nuscenes.utils.data_classes import Box
from tqdm import tqdm

name2nuscenceclass = {
    "car": "vehicle.car",
    "truck": "vehicle.truck",    
    "van": "vehicle.van",
    "bus": "vehicle.bus",   
    "cyclist": "vehicle.cyclist",
    "tricyclist": "vehicle.tricyclist", 
    "motorcyclist": "vehicle.motorcyclist", 
    "barrowlist": "vehicle.barrowlist",   
    "pedestrian": "human.pedestrian.adult",             
}

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

classes = [
    'Vehicle',
    'Cyclist',
    'Pedestrian',
]

conf = dict(
    point_cloud_range=[0, -51.2, -5, 102.4, 51.2, 3],
    grid_size=[1024, 1024, 1],
    voxel_size=[0.1, 0.1, 8],
    out_size_factor=4,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
)

H = 1080
W = 1920
final_dim = (1080, 1920)
img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
                to_rgb=True)
ida_aug_conf = {
    'final_dim':
    final_dim,
    'H':
    H,
    'W':
    W,
    'bot_pct_lim': (0.0, 0.0),
    'cams': ['CAM_FRONT'],
    'Ncams': 1,
}
img_mean = np.array(img_conf['img_mean'], np.float32)
img_std = np.array(img_conf['img_std'], np.float32)
to_rgb = img_conf['to_rgb']

def read_json(path_json):
    with open(path_json, "r") as load_f:
        my_json = json.load(load_f)
    return my_json

def get_velo2cam(path):
    my_json = read_json(path)
    t_velo2cam = np.array(my_json["translation"])
    r_velo2cam = np.array(my_json["rotation"])
    return r_velo2cam, t_velo2cam

def get_P(path):
    my_json = read_json(path)
    P = np.array(my_json["cam_K"]).reshape(3,3)
    return P

def get_annos(path):
    my_json = read_json(path)
    gt_names = []
    gt_boxes = []
    for item in my_json:
        gt_names.append(item["type"].lower())
        x, y, z = float(item["3d_location"]["x"]), float(item["3d_location"]["y"]), float(item["3d_location"]["z"])
        h, w, l = float(item["3d_dimensions"]["h"]), float(item["3d_dimensions"]["w"]), float(item["3d_dimensions"]["l"])                                                            
        lidar_yaw = float(item["rotation"])
        gt_boxes.append([x, y, z, l, w, h, lidar_yaw])
    gt_boxes = np.array(gt_boxes)
    return gt_names, gt_boxes

def load_data(dair_root, token):
    sample_id = token.split('/')[1].split('.')[0]
    img_pth = os.path.join(dair_root, token)
    camera_intrinsic_path = os.path.join(dair_root, "calib", "camera_intrinsic", sample_id + ".json")
    virtuallidar_to_camera_path = os.path.join(dair_root, "calib", "virtuallidar_to_camera", sample_id + ".json")
    label_path = os.path.join(dair_root, "label", "camera", sample_id + ".json")
    r_velo2cam, t_velo2cam = get_velo2cam(virtuallidar_to_camera_path)
    P = get_P(camera_intrinsic_path)
    gt_names, gt_boxes = get_annos(label_path)
    return r_velo2cam, t_velo2cam, P, gt_names, gt_boxes, img_pth

def cam2velo(r_velo2cam, t_velo2cam):
    Tr_velo2cam = np.eye(4)
    Tr_velo2cam[:3, :3] = r_velo2cam
    Tr_velo2cam[:3 ,3] = t_velo2cam.flatten()
    Tr_cam2velo = np.linalg.inv(Tr_velo2cam)
    r_cam2velo = Tr_cam2velo[:3, :3]
    t_cam2velo = Tr_cam2velo[:3, 3]
    return r_cam2velo, t_cam2velo
    
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

def get_denorm(rotation_matrix, translation):
    lidar2cam = np.eye(4)
    lidar2cam[:3, :3] = rotation_matrix
    lidar2cam[:3, 3] = translation.flatten()
    ground_points_lidar = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    ground_points_lidar = np.concatenate((ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), axis=1)
    ground_points_cam = np.matmul(lidar2cam, ground_points_lidar.T).T
    denorm = -1 * equation_plane(ground_points_cam)
    
    return denorm

def get_denorm_2d(sweepego2sweepsensor):
    ground_points_lidar = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    ground_points_lidar = np.concatenate((ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), axis=1)
    ground_points_cam = np.matmul(sweepego2sweepsensor, ground_points_lidar.T).T
    denorm = -1 * equation_plane(ground_points_cam)
    return denorm

def get_reference_height(denorm):
    ref_height = np.abs(denorm[3]) / np.sqrt(denorm[0]**2 + denorm[1]**2 + denorm[2]**2)
    return ref_height.astype(np.float32)

def sample_ida_augmentation():
    """Generate ida augmentation values based on ida_config."""
    H, W = ida_aug_conf['H'], ida_aug_conf['W']
    fH, fW = ida_aug_conf['final_dim']
    resize = max(fH / H, fW / W)
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    crop_h = int(
        (1 - np.mean(ida_aug_conf['bot_pct_lim'])) * newH) - fH
    crop_w = int(max(0, newW - fW) / 2)
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    flip = False
    rotate_ida = 0
    return resize, resize_dims, crop, flip, rotate_ida

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

def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

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

def get_image(cam_infos, cams):
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
    data_root = "data/dair-v2x-i/"
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
            rotate_ida = sample_ida_augmentation(
                )
        for sweep_idx, cam_info in enumerate(cam_infos):
            img = Image.open(
                os.path.join(data_root, cam_info[cam]['filename']))
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
            denorm = get_denorm_2d(sweepego2sweepsensor.numpy())
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
            
            denorm = get_denorm_2d(sweepego2sweepsensor.numpy())
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
            img = mmcv.imnormalize(np.array(img), img_mean,
                                    img_std, to_rgb)
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
    return ret_list

def get_gt(info, cams):
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
    gt_corners = list()
    for idx, ann_info in enumerate(info['ann_infos']):
        if (map_name_from_general_to_detection[ann_info['category_name']]
                not in classes
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
        gt_corner = np.array(box.corners().transpose(1, 0))
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        
        gt_labels.append(
                classes.index(map_name_from_general_to_detection[
                ann_info['category_name']]))

        gt_corners.append(gt_corner)
    return torch.Tensor(gt_boxes), torch.tensor(gt_labels), torch.tensor(gt_corners)

def lidar2img(points_lidar, sweep_sensor2ego_mats, sweep_intrins, sweep_ida_mats):
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

def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(valid,
                        np.logical_and(points[:, 0] < width,
                                        points[:, 1] < height))
    return valid

def get_2d(info):
    cam_infos = list()
    cams = ['CAM_FRONT']
    for key_idx in [0]:
        cam_infos.append(info['cam_infos'])
    image_data_list = get_image(cam_infos, cams)
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
    img_metas['token'] = info['sample_token']      
    gt_boxes, gt_labels, gt_corners = get_gt(info, cams)

    gt_flag = np.zeros([gt_boxes.shape[0]], dtype=np.bool)
    gt_2d_list = list()
    gt_corners = gt_corners.numpy().astype('float32').reshape(-1, 3)
    sweep_imgs_ori_array = sweep_imgs_ori.squeeze().numpy().transpose(1, 2, 0).copy()
    sweep_sensor2ego_mats_array = sweep_sensor2ego_mats.squeeze().numpy()
    sweep_intrins_array = sweep_intrins.squeeze().numpy()
    sweep_ida_mats_array = sweep_ida_mats.squeeze().numpy()
            
    # Transform 3D box to 2D Box
    corners_img_gt, valid_z_gt = lidar2img(gt_corners, sweep_sensor2ego_mats_array, 
                                            sweep_intrins_array[:3, :3], sweep_ida_mats_array[:2, :2])
    valid_shape_gt = check_point_in_img(corners_img_gt, sweep_imgs_ori_array.shape[0], sweep_imgs_ori_array.shape[1])     
    valid_all_gt = np.logical_and(valid_z_gt, valid_shape_gt)      
    valid_z_gt = valid_z_gt.reshape(-1, 8)
    valid_shape_gt = valid_shape_gt.reshape(-1, 8)
    valid_all_gt = valid_all_gt.reshape(-1, 8)
    corners_img_gt = corners_img_gt.reshape(-1, 8, 2).astype(np.int)

    gt_boxes = gt_boxes.numpy()
    
    for aid in range(valid_all_gt.shape[0]):             
        x, y, z = gt_boxes[aid][0], gt_boxes[aid][
                    1], gt_boxes[aid][2]
        coor_x = (
            x - conf['point_cloud_range'][0]
        ) / conf['voxel_size'][0]
        coor_y = (
            y - conf['point_cloud_range'][1]
        ) / conf['voxel_size'][1]

        center = torch.tensor([coor_x, coor_y],
                                dtype=torch.float32)
        center_int = center.to(torch.int32)

        if not (0 <= center_int[0] < (conf['grid_size'])[0]
           and 0 <= center_int[1] < (conf['grid_size'])[1]):
            continue 
        if valid_z_gt[aid].sum() >= 1:
            min_col = max(min(corners_img_gt[aid, valid_z_gt[aid], 0].min(), sweep_imgs_ori_array.shape[1]), 0)
            max_col = max(min(corners_img_gt[aid, valid_z_gt[aid], 0].max(), sweep_imgs_ori_array.shape[1]), 0)
            min_row = max(min(corners_img_gt[aid, valid_z_gt[aid], 1].min(), sweep_imgs_ori_array.shape[0]), 0)
            max_row = max(min(corners_img_gt[aid, valid_z_gt[aid], 1].max(), sweep_imgs_ori_array.shape[0]), 0)
            if (max_col - min_col) == 0 or (max_row - min_row) == 0:
                continue     
            gt_flag[aid] = True 
            gt_2d_list.append([min_col, min_row, max_col, max_row])  
    return gt_flag

def generate_info_dair(dair_root, split):    
    infos = mmcv.load("data/single-infrastructure-split-data.json")

    split_list = infos[split]

    infos = list()
    split_num = 20
    for sample_id in tqdm(split_list):
        token = "image/" + sample_id + ".jpg"
        r_velo2cam, t_velo2cam, camera_intrinsic, gt_names, gt_boxes, img_pth = load_data(dair_root, token)

        info = dict()
        cam_info = dict()
        info['sample_token'] = token
        info['timestamp'] = 1000000
        info['scene_token'] = token
        cam_names = ['CAM_FRONT']
        lidar_names = ['LIDAR_TOP']
        cam_infos, lidar_infos = dict(), dict()
        for cam_name in cam_names:
            cam_info = dict()
            cam_info['sample_token'] = token
            cam_info['timestamp'] = 1000000
            cam_info['is_key_frame'] = True
            cam_info['height'] = 1080
            cam_info['width'] = 1920
            cam_info['filename'] = token
            ego_pose = {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0], "token": token, "timestamp": 1000000}
            cam_info['ego_pose'] = ego_pose
            
            denorm = get_denorm(r_velo2cam, t_velo2cam)
            r_cam2velo, t_cam2velo = cam2velo(r_velo2cam, t_velo2cam)
            calibrated_sensor = {"token": token, "sensor_token": token, "translation": t_cam2velo.flatten(), "rotation_matrix": r_cam2velo, "camera_intrinsic": camera_intrinsic}
            cam_info['calibrated_sensor'] = calibrated_sensor
            cam_info['denorm'] = denorm
            cam_infos[cam_name] = cam_info                  
        for lidar_name in lidar_names:
            lidar_info = dict()
            lidar_info['sample_token'] = token
            ego_pose = {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0], "token": token, "timestamp": 1000000}
            lidar_info['ego_pose'] = ego_pose
            lidar_info['timestamp'] = 1000000
            lidar_info['filename'] = "velodyne/" + sample_id + ".pcd"
            lidar_info['calibrated_sensor'] = calibrated_sensor
            lidar_infos[lidar_name] = lidar_info            
        info['cam_infos'] = cam_infos
        info['lidar_infos'] = lidar_infos
        info['sweeps'] = list()

        gt_flag = np.zeros([gt_boxes.shape[0]], dtype=np.bool)
        ann_infos = list()
        for idx in range(gt_boxes.shape[0]):
            category_name = gt_names[idx]
            if category_name not in name2nuscenceclass.keys():
                continue
            gt_flag[idx] = True
            gt_box = gt_boxes[idx]
            lwh = gt_box[3:6]
            loc = gt_box[:3]    # need to certify
            yaw_lidar = gt_box[6]
            rot_mat = np.array([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], 
                                [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], 
                                [0, 0, 1]])    
            rotation = Quaternion(matrix=rot_mat)
            ann_info = dict()
            ann_info["category_name"] = name2nuscenceclass[category_name]
            ann_info["translation"] = loc
            ann_info["rotation"] = rotation
            ann_info["yaw_lidar"] = yaw_lidar
            ann_info["size"] = lwh
            ann_info["prev"] = ""
            ann_info["next"] = ""
            ann_info["sample_token"] = token
            ann_info["instance_token"] = token
            ann_info["token"] = token
            ann_info["visibility_token"] = "0"
            ann_info["num_lidar_pts"] = 3
            ann_info["num_radar_pts"] = 0            
            ann_info['velocity'] = np.zeros(3)
            ann_infos.append(ann_info)
        info['ann_infos'] = ann_infos        
        gt_boxes = gt_boxes[gt_flag]
        gt_names = list(np.array(gt_names)[gt_flag])
        gt_flag = get_2d(info)
        gt_boxes = gt_boxes[gt_flag]
        gt_names = list(np.array(gt_names)[gt_flag])
        if gt_boxes.shape[0] == 0:
            print("000000")


        loop_num = int(len(gt_names) / split_num)
        last_num = len(gt_names) % split_num
        for loop_index in range(loop_num):
            gt_boxes_split = gt_boxes[loop_index*split_num:(loop_index+1)*split_num]
            gt_names_split = gt_names[loop_index*split_num:(loop_index+1)*split_num]
            if gt_boxes_split.shape[0] > 20:
                a = 0
            ann_infos = list()
            for idx in range(gt_boxes_split.shape[0]):
                category_name = gt_names_split[idx]
                if category_name not in name2nuscenceclass.keys():
                    continue
                gt_box = gt_boxes_split[idx]
                lwh = gt_box[3:6]
                loc = gt_box[:3]    # need to certify
                yaw_lidar = gt_box[6]
                rot_mat = np.array([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], 
                                    [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], 
                                    [0, 0, 1]])    
                rotation = Quaternion(matrix=rot_mat)
                ann_info = dict()
                ann_info["category_name"] = name2nuscenceclass[category_name]
                ann_info["translation"] = loc
                ann_info["rotation"] = rotation
                ann_info["yaw_lidar"] = yaw_lidar
                ann_info["size"] = lwh
                ann_info["prev"] = ""
                ann_info["next"] = ""
                ann_info["sample_token"] = token
                ann_info["instance_token"] = token
                ann_info["token"] = token
                ann_info["visibility_token"] = "0"
                ann_info["num_lidar_pts"] = 3
                ann_info["num_radar_pts"] = 0            
                ann_info['velocity'] = np.zeros(3)
                ann_infos.append(ann_info)
            info = info.copy()
            info['ann_infos'] = ann_infos
            infos.append(info)

        if last_num == 0:
            continue        
        ann_infos = list()
        gt_boxes_split = gt_boxes[-last_num:]
        gt_names_split = gt_names[-last_num:]        
        for idx in range(gt_boxes_split.shape[0]):
            category_name = gt_names_split[idx]
            if category_name not in name2nuscenceclass.keys():
                continue
            gt_box = gt_boxes_split[idx]
            lwh = gt_box[3:6]
            loc = gt_box[:3]    # need to certify
            yaw_lidar = gt_box[6]
            rot_mat = np.array([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], 
                                [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], 
                                [0, 0, 1]])    
            rotation = Quaternion(matrix=rot_mat)
            ann_info = dict()
            ann_info["category_name"] = name2nuscenceclass[category_name]
            ann_info["translation"] = loc
            ann_info["rotation"] = rotation
            ann_info["yaw_lidar"] = yaw_lidar
            ann_info["size"] = lwh
            ann_info["prev"] = ""
            ann_info["next"] = ""
            ann_info["sample_token"] = token
            ann_info["instance_token"] = token
            ann_info["token"] = token
            ann_info["visibility_token"] = "0"
            ann_info["num_lidar_pts"] = 3
            ann_info["num_radar_pts"] = 0            
            ann_info['velocity'] = np.zeros(3)
            ann_infos.append(ann_info)
        info = info.copy()
        info['ann_infos'] = ann_infos
        infos.append(info)
        
    return infos

def main():
    dair_root = "data/dair-v2x-i"
    train_infos = generate_info_dair(dair_root, split='train')
    mmcv.dump(train_infos, './data/dair-v2x-i/dair_12hz_infos_train_2d_20.pkl')

if __name__ == '__main__':
    main()
