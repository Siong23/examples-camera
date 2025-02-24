"""A demo that runs object detection on RTSP stream via SOCKS5 proxy using OpenCV and streams frames via Flask-based HTTP. 
This allows you to view the processed video stream in a web browser.

Run with CPU:
python3 detectv3.py

Run with Coral accelerator:
python3 detectv3.py \
  --edgetpu \
  --model ../all_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite

"""

import argparse
import cv2
import os
import subprocess
import numpy as np
from flask import Flask, Response
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
from tflite_runtime.interpreter import Interpreter

app = Flask(__name__)

# Get RTSP address from environment variable, default to the hardcoded value
rtsp_address = os.getenv("RTSP_ADDRESS", "rtsp://192.168.0.159:30000/unicast")

def generate_frames(args):
    """Generator function to fetch frames from RTSP stream via SOCKS5 proxy and perform inference."""
    
    # Use FFmpeg to fetch frames
    ffmpeg_cmd = [
        "proxychains", "ffmpeg",
        "-i", rtsp_address,
        "-f", "image2pipe",
        "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo",
        "-"
    ]
    
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

    frame_width, frame_height = 640, 480  # Adjust based on your stream resolution

    print(f'Loading {args.model} with {args.labels} labels.')

    # Load the model
    interpreter = make_interpreter(args.model) if args.edgetpu else Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    while True:
        raw_frame = process.stdout.read(frame_width * frame_height * 3)  # Read one frame
        if not raw_frame:
            print("Error: Cannot open RTSP stream via SOCKS5 proxy")
            break
        
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((frame_height, frame_width, 3))

        # Perform inference
        cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)

        if args.edgetpu:
            run_inference(interpreter, cv2_im_rgb.tobytes())
        else:
            input_details = interpreter.get_input_details()
            interpreter.set_tensor(input_details[0]['index'], [cv2_im_rgb])
            interpreter.invoke()

        objs = get_objects(interpreter, args.threshold)[:args.top_k]
        frame = append_objs_to_img(frame, inference_size, objs, labels)

        # Encode frame as JPEG for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    process.terminate()

@app.route('/video_feed')
def video_feed():
    """Flask route to serve video over HTTP."""
    args = get_args()  # Get the parsed arguments
    return Response(generate_frames(args), mimetype='multipart/x-mixed-replace; boundary=frame')

def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, _ = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = f'{percent}% {labels.get(obj.id, obj.id)}'

        cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(cv2_im, label, (x0, y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess.tflite'
    default_labels = 'coco_labels.txt'

    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                        action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
