
from mmdet3d.apis import init_detector, inference_detector

config_file = 'configs/pointpillars/myhv_pointpillars_secfpn_6x8_160e_waymokitti-3d-3class.py'
checkpoint_file = 'mypointpillar_waymokitti/epoch_45.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
point_cloud = '/data/cmpe249-fa21/4c_train5678/training/velodyne/000000.bin'
result, data = inference_detector(model, point_cloud)
# visualize the results and save the results in 'results' folder
model.show_results(data, result, out_dir='results')
