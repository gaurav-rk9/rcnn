import detectron2
import cv2
import os

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils import visualizer
from detectron2 import model_zoo


def setup():
	cfg = get_cfg()
	cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
	cfg.MODEL.WEIGHTS = "output2/model_final.pth"
	cfg.OUTPUT_DIR = "output2/predictions/video-Chrih8BX8fbZLmsoX-frame-010771-AXAgnLg4GMyrzBQvo"

	return cfg

cfg = setup()
pred = DefaultPredictor(cfg)
inputs = cv2.imread("../FLIR/images_rgb_val/data/video-Chrih8BX8fbZLmsoX-frame-010771-AXAgnLg4GMyrzBQvo.jpg")
#print(inputs)
outputs = pred(inputs)

x = visualizer.Visualizer(inputs)
y = x.draw_instance_predictions(outputs['instances'].to('cpu'))
y.save(cfg.OUTPUT_DIR)

#print(outputs)
