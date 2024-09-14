import cv2
import numpy as np
import onnxruntime as ort
from random import randint
from .utils import non_max_suppression

class YOLO:
    def __init__(self, model_path, labels, model_h, model_w, thred_nms=0.45, thred_cond=0.25):
        self.model_h = model_h
        self.model_w = model_w
        self.labels = labels
        self.thred_nms = thred_nms
        self.thred_cond = thred_cond

        providers = ['CUDAExecutionProvider']
        self.net = ort.InferenceSession(model_path, providers=providers)
        self.input_name=self.get_input_name()
        self.output_name=self.get_output_name()

    def get_input_name(self):
        input_name=[]
        for node in self.net.get_inputs():
            input_name.append(node.name)

        return input_name

    def get_output_name(self):
        output_name=[]
        for node in self.net.get_outputs():
            output_name.append(node.name)

        return output_name

    def get_input_feed(self,image_tensor):
        input_feed={}
        for name in self.input_name:
            input_feed[name]=image_tensor

        return input_feed

    @staticmethod
    def plot_one_box(x, img, color=None, label=None, line_thickness=None):
        tl = (line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)
        color = color or [randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def preprocess(self, img):
        
        shape = img.shape[:2]
        new_shape = (self.model_h, self.model_w)

        r = min(new_shape[1] / shape[1], new_shape[0] / shape[0])

        new_unpad = (int(shape[1] * r), int(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
        
        if shape[::-1] != new_unpad:
            img_pre = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            img_pre = img

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img_pre = cv2.copyMakeBorder(img_pre, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[114, 114, 114])

        img_pre = img_pre[:, :, ::-1].transpose(2, 0, 1)
        img_pre = np.ascontiguousarray(img_pre, dtype=np.float32)
        img_pre /= 255.0
        img_pre = np.expand_dims(img_pre, axis=0)

        return img_pre
    
    def afterprocess(self, outs, src_h, src_w):
        outs[:, :4] = self.scale_coords((self.model_h, self.model_w), outs[:, :4], (src_h, src_w)).round()
        boxes = outs[:, :4]
        confs = outs[:, 4]
        ids = outs[:, 5].astype(np.int32)
        return np.array(boxes), np.array(confs), np.array(ids)
    

    def infer_img(self, img):
        img_pre = self.preprocess(img)
        img_feed = self.get_input_feed(img_pre)
        outs = self.net.run(None, img_feed)[0].squeeze(axis=0)
        outs = non_max_suppression(outs, self.thred_cond, self.thred_nms)
        src_h, src_w, _ = img.shape
        if outs is not None:
            return self.afterprocess(outs, src_h, src_w)
        else:
            return [], [], []

    def draw_detections(self, img, boxes, confs, ids):
        for box, score, id in zip(boxes, confs, ids):
            label = '%s:%.2f' % (self.labels[id], score)
            self.plot_one_box(box.astype(np.int16), img, color=(255, 0, 0), label=label, line_thickness=None)

    def scale_coords(self,img1_shape, coords, img0_shape):


        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2

        coords[:, [0, 2]] -= pad[0]
        coords[:, [1, 3]] -= pad[1]
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords

    def clip_coords(self, boxes, img_shape):

        boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])
        boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])
        boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])
        boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])
