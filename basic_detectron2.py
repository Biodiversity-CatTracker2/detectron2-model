"""Source: https://github.com/facebookresearch/detectron2 With minor custom changes"""
import shlex
import subprocess
from glob import glob
from pathlib import Path

import cv2
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer


def _config():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return cfg, predictor


def custom_vis(file,
               output_dir,
               save=False,
               alpha=0.0,
               _linewidth=10):
    im = cv2.imread(file)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                   scale=1.2)
    num_labels = outputs['instances'].get_fields()['pred_classes'].shape[0]
    out = draw_instance_predictions(
        v,
        outputs["instances"].to("cpu"),
        _alpha=alpha,
        _colors=['#d9534f'],
        labels=['detection'] * num_labels,
        _linewidth=_linewidth,
    )
    out_img = out.get_image()[:, :, ::-1]
    if save:
        cv2.imwrite(f'output/{Path(file).name}', out_img)
    return out, out_img


if __name__ == '__main__':
    setup_logger()

    # TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    # CUDA_VERSION = torch.__version__.split("+")[-1]
    # print("torch: ", TORCH_VERSION, "; cuda: ",
    #       CUDA_VERSION)  # Install detectron2 that matches this pytorch version

    cfg, predictor = _config()
    Path('output').mkdir(exist_ok=True)
    files = glob('<folder>')  # replace

    for file in files:
        custom_vis(_files, 'output', save=True)
