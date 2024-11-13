import torch
from thop import profile

from ultralytics import YOLO

model = YOLO("ultralytics/models/v8/yolov8n.yaml")
model.train(**{'cfg': 'ultralytics/yolo/cfg/default.yaml'})
#

# model = YOLO("result/best.pt")
# model.val(project='result/val', name='t1')

# model = YOLO("ultralytics/models/v8/yolov8s-test.yaml")
# print(model.model)

# inputs = torch.randn(1, 3, 640, 640)
# flops, params = profile(model, inputs=(inputs, ))
# print(flops/1e9, params/1e6)
