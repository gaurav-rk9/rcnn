import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor,launch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator


class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

register_coco_instances("my_dataset_train", {}, "../FLIR/images_rgb_train/coco.json", "../FLIR/images_rgb_train")
register_coco_instances("my_dataset_val", {}, "../FLIR/images_rgb_val/coco.json", "../FLIR/images_rgb_val")


def conf():
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))
  cfg.DATASETS.TRAIN = ("my_dataset_train",)
  cfg.DATASETS.TEST = ("my_dataset_val",)
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
  return cfg


cfg = conf()
'''os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)  
#print(cfg.OUTPUT_DIR)
#print(cfg.MODEL.WEIGHTS)
model = CocoTrainer.build_model(cfg)
#print(model.state_dict())
res = CocoTrainer.test(cfg, model)
print(res)'''


print(COCOEvaluator("my_dataset_val",cfg).evaluate())
