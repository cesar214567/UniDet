# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import sys
import tqdm
import numpy as np
import shutil
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from PIL import Image

from unidet.predictor import UnifiedVisualizationDemo
from unidet.config import add_unidet_config

# constants
WINDOW_NAME = "Unified detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.MODEL.DEVICE='cpu'
    add_unidet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = UnifiedVisualizationDemo(cfg)
    print(demo.metadata.thing_classes)
    counter = 0
    #demo.export_onnx_model()
    if args.input:
        if (args.output):
            columns=np.array(['ALB','BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT','missing'])
            for folder in columns: 
                try:
                    shutil.rmtree(f'{args.output}{folder}')
                    os.mkdir(f'{args.output}{folder}')
                except:
                    os.mkdir(f'{args.output}{folder}')
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            #if len(predictions['instances'])>0: 
            #    logger.info(predictions['instances'])
            #    logger.info(predictions['instances'][0].pred_boxes)
            #    logger.info(predictions['instances'][0].pred_classes)

            if args.output:
                

                if len(predictions['instances'])>0: 
                    instances = predictions['instances']
                    for bounding_box,label,score in zip(instances.pred_boxes,instances.pred_classes,instances.scores): 
                        text_label =demo.metadata.thing_classes[label]
                        #logger.info(bounding_box,text_label,score)
                        if 'Fish' in text_label or 'Animal' in text_label:
                            new_path = f'{args.output}{path.split("/")[4]}/{counter}.jpg'
                            logger.info(new_path)
                            #logger.info(bounding_box,text_label,score)
                            counter+=1
                            temp_img = Image.fromarray(img)
                            b, g, r = temp_img.split()
                            temp_img = Image.merge("RGB", (r, g, b))
                            #img = img[:,:,::-1]
                            #temp_img = Image.fromarray(img)

                            [x,y,x_max,y_max] = bounding_box.numpy()
                            x_diff = x_max - x
                            y_diff = y_max - y
                            bounding_box[0] = max(0,x-x_diff*0.15)
                            bounding_box[1] = max(0,y-y_diff*0.15)
                            bounding_box[2] = min(img.shape[1],x_max+x_diff*0.3)
                            bounding_box[3] = min(img.shape[0],y_max+y_diff*0.3)
                            
                            #logger.info(', '.join(map(str,int([x,y,x_max,y_max]))))
                            temp_img = temp_img.crop(tuple(bounding_box.numpy()))
                            #temp_img.show()
                            temp_img.save(new_path)
                        
                
            else:       
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
