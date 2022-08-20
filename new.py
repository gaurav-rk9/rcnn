import detectron2

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer,launch
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator

register_coco_instances("my_dataset_train", {}, "../FLIR/images_rgb_train/coco.json", "../FLIR/images_rgb_train")
register_coco_instances("my_dataset_val", {}, "../FLIR/images_rgb_val/coco.json", "../FLIR/images_rgb_val")


def setup():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_train","my_dataset_val")

    cfg.DATALOADER.NUM_WORKERS = 16
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.03


    #cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 40000 #adjust up if val mAP is still rising, adjust down if overfit
    #cfg.SOLVER.STEPS = (20000,30000)
    cfg.SOLVER.GAMMA = 0.05
    cfg.SOLVER.WEIGHT_DECAY = 0

    #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80

    cfg.TEST.EVAL_PERIOD = 2500
    cfg.OUTPUT_DIR = "output3"
    return cfg

class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name,("bbox",))

def ghi():
    cfg = setup()
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == '__main__':
    launch(ghi,4)
