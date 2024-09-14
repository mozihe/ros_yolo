import numpy as np

def xywh2xyxy(x):

    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def box_iou(box1, box2):

    x11, y11, x12, y12 = np.split(box1, 4, axis=1)
    x21, y21, x22, y22 = np.split(box2, 4, axis=1)

    xa = np.maximum(x11, np.transpose(x21))
    xb = np.minimum(x12, np.transpose(x22))
    ya = np.maximum(y11, np.transpose(y21))
    yb = np.minimum(y12, np.transpose(y22))

    area_inter = np.maximum(0, (xb - xa + 1)) * np.maximum(0, (yb - ya + 1))

    area_1 = (x12 - x11 + 1) * (y12 - y11 + 1)
    area_2 = (x22 - x21 + 1) * (y22 - y21 + 1)
    area_union = area_1 + np.transpose(area_2) - area_inter

    iou = area_inter / area_union
    return iou

def numpy_cpu_nms(dets, scores, thresh):

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    order = scores.argsort()[::-1]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        over = (w * h) / (area[i] + area[order[1:]] - w * h)
        index = np.where(over <= thresh)[0]
        order = order[index + 1]
    return np.array(keep)


def non_max_suppression(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        max_det=300):
    
    xc = prediction[:, 4] > conf_thres
    max_nms = 25200
    x = prediction[xc]
    if not x.shape[0]:
        return None
    x[:, 5:] *= x[:, 4:5]
    box = xywh2xyxy(x[:, :4])
    conf, j = x[:, 5:].max(1), x[:, 5:].argmax(1)
    conf = conf.reshape(conf.shape[0], 1)
    j = j.reshape(j.shape[0], 1)
    x = np.concatenate((box, conf, j), 1)[conf.flatten() > conf_thres]
    if x.shape[0] > max_nms:
        x = x[x[:, 4].argsort()[::-1][:max_nms]]
    boxes, scores = x[:, :4], x[:, 4]
    i = numpy_cpu_nms(boxes, scores, iou_thres).astype(np.int)
    if i.shape[0] > max_det:
        i = i[:max_det]
    return x[i]
