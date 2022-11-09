import tensorrt as trt
import pycuda.driver as cuda 
import pycuda.autoinit
import cv2
import numpy as np 
import time

import argparse
import os
from loguru import logger

from yolox.utils import multiclass_nms
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="480x640_fp16.engine",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--video_path",
        type=str,
        default='videos/palace.mp4',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.1,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "-n",
        "--nms_thr",
        type=float,
        default=0.7,
        help="NMS threshould.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="480,640",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

class ModelTRT():
    def __init__(self,weights: str):
        
        ### init engine ###
        with open(weights, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()

        self.bindings = []
        for binding in engine:
            size = self.context.get_tensor_shape(binding)
            dtype = trt.nptype(engine.get_tensor_dtype(binding))
            if engine.binding_is_input(binding):
                self.input_buffer = np.ascontiguousarray(np.zeros(size, dtype))
                self.input_memory = cuda.mem_alloc(self.input_buffer.nbytes)
                self.bindings.append(int(self.input_memory))
            else:
                self.output_buffer = np.ascontiguousarray(np.zeros(size, dtype))
                self.output_memory = cuda.mem_alloc(self.output_buffer.nbytes)
                self.bindings.append(int(self.output_memory))

        res = self.warmup(np.zeros((480,640,3)))


    def warmup(self, image: np.ndarray):
        np.copyto(self.input_buffer, image.astype('float32'))
        cuda.memcpy_htod(self.input_memory, self.input_buffer)
        self.context.execute_v2(bindings=self.bindings)
        cuda.memcpy_dtoh(self.output_buffer, self.output_memory)
        # print('warmup done, output shape is ', self.output_buffer.shape)
        return self.output_buffer


def preproc(img, input_size):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])

    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_NEAREST,
    )

    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = padded_img[:, :, ::-1]

    return padded_img, r


class Predictor(object):
    def __init__(self, args):
        self.args = args
        self.model = ModelTRT(args.model)

        self.input_shape = tuple(map(int, args.input_shape.split(',')))

    def inference(self, ori_img, timer):
        timer.tic()
        t1 = time.time()
        img_info = {"id": 0}
        height, width = ori_img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = ori_img

        # preprocess with resize
        img, ratio = preproc(ori_img, self.input_shape)
        # no resize
        # img = ori_img
        # ratio = 1
        img_info["ratio"] = ratio

        t2 = time.time()
        output = self.model.warmup(img)

        t3 = time.time()
        predictions = output[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self.args.nms_thr, score_thr=self.args.score_thr)

        print("prepro %.5f pro %.5f postpro %.5f"%(t2 - t1, t3-t2, time.time()-t3))
        return dets[:, :-1], img_info


def imageflow_demo(predictor, args):
    cap = cv2.VideoCapture(args.video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = args.output_dir
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, args.video_path.split("/")[-1])
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    # vid_writer = cv2.VideoWriter(
    #     save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (640,480)
    # )
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            online_targets = tracker.update(outputs, [img_info['height'], img_info['width']], [img_info['height'], img_info['width']])
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            timer.toc()
            results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1,
                                      fps=1. / timer.average_time)
            vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1
    

if __name__ == '__main__':
    args = make_parser().parse_args()

    predictor = Predictor(args)
    imageflow_demo(predictor, args)


