# mymmdetection3d
My fork and modified version of OpenMMLab's mmdetection3d for general 3D object detection.

## Install the mymmdetection3d on SJSU HPC
The original mmdetection3d does not work on SJSU HPC, you need to use our forked version to install mymmdetection3d on SJSU HPC. I changed the setup.py file and add the cublas_v2.h path.

Check how to add multiple class to mmdetection3d: https://mmdetection3d.readthedocs.io/en/stable/index.html, 
Configure meaning: https://mmdetection3d.readthedocs.io/en/stable/tutorials/config.html?highlight=workers_per_gpu#an-example-of-votenet


## Dataset

### Kitti dataset conversion and generate infos.pkl
[create_kitti_info_file](/tools/data_converter/kitti_converter.py) calls [get_kitti_image_info](/tools/data_converter/kitti_data_utils.py) to get kitti_infos_train then calls [_calculate_num_points_in_gt](/tools/data_converter/kitti_data_utils.py)

The detailed info dictionary generated in kitti_infos_train.pkl is
info['image']=image_info 
* image_info['image_path']
* image_info['image_shape']
* image_info['image_idx']: idx
info['point_cloud'] = pc_info
* pc_info['velodyne_path']
* pc_info['num_features']= 4
info['calib'] = calib_info (actual calibration data)
* calib_info['P0'] = P0
* calib_info['P1'] = P1
* calib_info['P2'] = P2
* calib_info['P3'] = P3
* calib_info['R0_rect'] = rect_4x4
* calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
* calib_info['Tr_imu_to_velo'] = Tr_imu_to_velo
info['pose'] actual pose data
info['annos'] actual annotations
* annos['num_points_in_gt'] obtained from _calculate_num_points_in_gt

There are four info .pkl files are created: kitti_infos_test.pkl, kitti_infos_train.pkl, kitti_infos_trainval.pkl, kitti_infos_val.pkl
 
After the info .pkl files, create_groundtruth_database function in [create_gt_database.py](/tools/data_converter/create_gt_database.py) will create kitti_dbinfos_train.pkl, kitti_gt_database folder ({image_idx}_{names[i]}_{i}.bin, e.g., 1196_Car_40.bin, 1012143_Pedestrian_9.bin) and gt.bin (a big file) as the ground truth database
1. Define dataset_cfg, which contains the data configuration and pipelines: LoadPointsFromFile (load_dim, use_dim), LoadAnnotations3D), then build the dataset
2. Iterate the dataset: for j in track_iter_progress(list(range(len(dataset)))), the dataset definition is in [class KittiDataset(Custom3DDataset)](/mmdet3d/datasets/kitti_dataset.py)
3. Save {image_idx}_{names[i]}_{i}.bin file to kitti_gt_database folder
4. Save the following db_info to all_db_infos list, then save to .pkl file
```bash
db_info = {
      'name': names[i], #used_classes
      'path': rel_filepath,# relative path for the {image_idx}_{names[i]}_{i}.bin file in kitti_gt_database folder
      'image_idx': image_idx,
      'gt_idx': i, #i-th object
      'box3d_lidar': gt_boxes_3d[i], #from annos['gt_bboxes_3d']
      'num_points_in_gt': gt_points.shape[0],
      'difficulty': difficulty[i],
  }
```

[class KittiDataset(Custom3DDataset)](/mmdet3d/datasets/kitti_dataset.py) is under mmdet3d/datasets and used to register the Dataset. It contains the key functions 
* get_data_info(self, index): Get data info from the previous info .pkl file according to the given index, it will return a dict as

```bash
input_dict = dict(
    sample_idx=sample_idx, #info['image']['image_idx']
    pts_filename=pts_filename, #self._get_pts_filename(sample_idx)
    img_prefix=None,
    img_info=dict(filename=img_filename), #info['image']['image_path']
    lidar2img=lidar2img) #Transformations
```

### Waymo dataset conversion and generate infos.pkl
[class Waymo2KITTI](/tools/data_converter/waymo_converter.py) converts the Waymo dataset to Kitti format. [myconvert_waymo2kitti.py](/tools/myconvert_waymo2kitti.py) used [class Waymo2KITTI](/tools/data_converter/waymo_converter.py) to convert the waymo dataset and create the info (.pkl) files
* [def create_waymo_info_file](/tools/data_converter/kitti_converter.py) create info file of waymo dataset, it mainly calls [get_waymo_image_info](/tools/data_converter/kitti_data_utils.py) and [_calculate_num_points_in_gt](/tools/data_converter/kitti_data_utils.py) in kitti_data_utils.py
* [get_waymo_image_info](/tools/data_converter/kitti_data_utils.py) creates info dictionary for each idx, return waymo_infos_train is a list for idx, each idx element is a info dictionary data for all data
file save to waymo_infos_train.pkl
* [_calculate_num_points_in_gt](/tools/data_converter/kitti_data_utils.py) used information in the info dictionary and get velodyne points. It can box_np_ops.remove_outside_points and obtain 3D bounding box in the Velodyne coordinate via gt_boxes_lidar = box_np_ops.box_camera_to_lidar(gt_boxes_camera, rect, Trv2c). Finally, it gets annos['num_points_in_gt']

The detailed info dictionary generated in waymo_infos_train.pkl is
info['image']=image_info 
* image_info['image_path']
* image_info['image_shape']
* image_info['image_idx']: idx
info['point_cloud'] = pc_info
* pc_info['velodyne_path']
* pc_info['num_features']= 6
info['timestamp'] # get it from the last line of velodyne (not available in our conversion)
info['calib'] = calib_info (actual calibration data)
* calib_info['P0'] = P0
* calib_info['P1'] = P1
* calib_info['P2'] = P2
* calib_info['P3'] = P3
* calib_info['P4'] = P4
* calib_info['R0_rect'] = rect_4x4
* calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
info['pose'] actual pose data
info['annos'] actual annotations
info['annos']['camera_id']
* annos['num_points_in_gt'] obtained from _calculate_num_points_in_gt
info['sweeps'] previous multiple velodyne path

After the info .pkl files, create_groundtruth_database function in [create_gt_database.py](/tools/data_converter/create_gt_database.py) will create waymo_dbinfos_train.pkl, waymo_gt_database folder ({image_idx}_{names[i]}_{i}.bin, e.g., 1196_Car_40.bin, 1012143_Pedestrian_9.bin) and gt.bin (a big file) as the ground truth database
* Define dataset_cfg, which contains the data configuration and pipelines: LoadPointsFromFile (load_dim, use_dim), LoadAnnotations3D)


### Training
#### Kitti Training
Use the following code to start the training via the configuration file (hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py):
```bash
(venvpy37cu10) [010796032@g8 mymmdetection3d]$ python tools/train.py configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py --work-dir ./mypointpillar_kitticar/
```
The evaluation results is:
Car AP@0.70, 0.70, 0.70:
bbox AP:98.5525, 90.6131, 89.9877
bev  AP:98.2510, 90.1967, 89.3395
3d   AP:90.2396, 87.9322, 79.6033
aos  AP:98.39, 90.22, 89.38
Car AP@0.70, 0.50, 0.50:
bbox AP:98.5525, 90.6131, 89.9877
bev  AP:98.8110, 90.7028, 90.1898
3d   AP:98.7234, 90.6936, 90.1576
aos  AP:98.39, 90.22, 89.38

2021-01-15 19:46:33,686 - mmdet - INFO - Epoch(val) [80][2992]  KITTI/Car_3D_easy_strict: 90.2396, KITTI/Car_BEV_easy_strict: 98.2510, KITTI/Car_2D_easy_strict: 98.5525, KITTI/Car_3D_moderate_strict: 87.9322, KITTI/Car_BEV_moderate_strict: 90.1967, KITTI/Car_2D_moderate_strict: 90.6131, KITTI/Car_3D_hard_strict: 79.6033, KITTI/Car_BEV_hard_strict: 89.3395, KITTI/Car_2D_hard_strict: 89.9877, KITTI/Car_3D_easy_loose: 98.7234, KITTI/Car_BEV_easy_loose: 98.8110, KITTI/Car_2D_easy_loose: 98.5525, KITTI/Car_3D_moderate_loose: 90.6936, KITTI/Car_BEV_moderate_loose: 90.7028, KITTI/Car_2D_moderate_loose: 90.6131, KITTI/Car_3D_hard_loose: 90.1576, KITTI/Car_BEV_hard_loose: 90.1898, KITTI/Car_2D_hard_loose: 89.9877


### Inference
In mmdet3d/core/visualizer/show_results.py, _write_ply will write points into ``ply`` format for meshlab visualization

The predicted results via model from model zoo:

![image](https://user-images.githubusercontent.com/6676586/111930963-64b11400-8a77-11eb-93d2-221321687014.png)

The predicted results via our own trained model:

![image](https://user-images.githubusercontent.com/6676586/111930977-71356c80-8a77-11eb-9937-55834b83e46b.png)


