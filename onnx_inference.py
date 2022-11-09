import argparse
import os

import cv2
import numpy as np
from loguru import logger

import onnxruntime

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
        default="480x640.onnx",
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
        # self.session = onnxruntime.InferenceSession(args.model)

        providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 6 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }),
        'CPUExecutionProvider',
        ]
        # model = "480x640_fp16.onnx"
        # self.dtype = np.float16

        model = "480x640.onnx"
        self.dtype = np.float32

        self.session = onnxruntime.InferenceSession(model, providers=providers)
        self.io_binding = self.session.io_binding()
        self.cpu_input = np.ones((480,640,3), dtype=np.float32)
        # self.io_binding.bind_cpu_input('images', self.cpu_input)
        self.io_binding.bind_output('output')

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
        img = img.astype(self.dtype)
        # no resize
        # img = ori_img.astype(np.float16)
        # ratio = 1
        img_info["ratio"] = ratio

        t11 = time.time()
        # self.io_binding.bind_cpu_input('images', img)
        self.io_binding.bind_cpu_input('images', img)
        t2 = time.time()
        # ort_inputs = {self.session.get_inputs()[0].name: img}
        # output = self.session.run(None, ort_inputs)

        self.session.run_with_iobinding(self.io_binding)
        output = self.io_binding.copy_outputs_to_cpu()[0]

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

        # print("prepro %.5f binding %.5f pro %.5f postpro %.5f"%(t11-t1, t2 - t11, t3-t2, time.time()-t3))
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

            # padded_img, _ = preprocess(frame, (480, 640), 1, 1) 
            # padded_img = padded_img.astype(np.uint8)         
            # padded_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)  
            # vid_writer.write(padded_img)
        else:
            break
        frame_id += 1
    

if __name__ == '__main__':
    args = make_parser().parse_args()

    predictor = Predictor(args)
    imageflow_demo(predictor, args)
