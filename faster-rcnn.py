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

register_coco_instances("my_dataset_train", {}, "../FLIR/images_rgb_train/coco.json", "../FLIR/images_rgb_train")
register_coco_instances("my_dataset_val", {}, "../FLIR/images_rgb_val/coco.json", "../FLIR/images_rgb_val")

def conf():
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))
  cfg.DATASETS.TRAIN = ("my_dataset_train",)
  cfg.DATASETS.TEST = ("my_dataset_val",)

  cfg.DATALOADER.NUM_WORKERS = 32
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")  # Let training initialize from model zoo
  cfg.SOLVER.IMS_PER_BATCH = 8 
  cfg.SOLVER.BASE_LR = 0.0001


  cfg.SOLVER.WARMUP_ITERS = 1000
  cfg.SOLVER.MAX_ITER = 2000 #adjust up if val mAP is still rising, adjust down if overfit
  cfg.SOLVER.STEPS = (1000, 1500)
  cfg.SOLVER.GAMMA = 0.05
    
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80

  cfg.TEST.EVAL_PERIOD = 500
  return cfg


class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)


def ghi():
  cfg = conf()
  trainer = CocoTrainer(cfg) 
  trainer.resume_or_load(resume=False)
  return trainer.train()

if __name__ == '__main__':

  launch(ghi,4)
  '''trainer = DefaultTrainer(cfg) 
  trainer.resume_or_load(resume=False)
  trainer.train()'''


  

  
  #os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
  cfg = conf()
  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
  trainer1 = CocoTrainer.build_model(cfg)
  #predictor = DefaultPredictor(cfg)
  evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir="../output_2/")
  val_loader = build_detection_test_loader(cfg, "my_dataset_val")
  print(inference_on_dataset(trainer1, val_loader, evaluator))

