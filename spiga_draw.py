import numpy as np
import tqdm
from PIL import Image
from datasets import load_dataset
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
from facelib import FaceDetector
import cv2
import os

processor = SPIGAFramework(ModelConfig("300wpublic"))

def center_crop(image, size):
    width, height = image.size
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image

def resize(image, size):
    width, height = image.size
    if width > height:
        # 按宽度进行比例调整
        new_width = size
        new_height = int(height * (size / width))
    else:
        # 按高度进行比例调整
        new_height = size
        new_width = int(width * (size / height))
    resized_image = image.resize((new_width, new_height))
    return resized_image

def preprocess(example, name, path):
    image = resize(example, 512)
    # 调用中心剪裁函数
    cropped_image = center_crop(image, 512)
    # 保存剪裁后的图像
    cropped_image.save(path+name)
    return cropped_image

# We obtain the bbox from the existing landmarks in the dataset.
# We could use `dlib`, but this should be faster.
# Note that the `landmarks` are stored as strings.

def get_landmarks(image, detector):
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    faces, boxes, scores, landmarks = detector.detect_align(image)  # 一定要用align啊
    boxes = boxes.cpu().numpy()
    box_ls = []
    for box in boxes:
        x, y, x1, y1 = box
        box = x, y, x1 - x, y1 - y
        box_ls.append(box)
    if len(box_ls) == 0:
        return []
    else:
        features = processor.inference(image, box_ls)
        landmarks = np.array(features['landmarks'])
        return landmarks


def parse_landmarks(landmarks):
    ldm = []
    for landmark in landmarks:
        ldm.append([(float(x), float(y)) for x, y in landmark])
    return ldm


def bbox_from_landmarks(landmarks_):
    landmarks = parse_landmarks(landmarks_)
    bbox = []
    for ldm in landmarks:
        landmarks_x, landmarks_y = zip(*ldm)

        x_min, x_max = min(landmarks_x), max(landmarks_x)
        y_min, y_max = min(landmarks_y), max(landmarks_y)
        width = x_max - x_min
        height = y_max - y_min

        # Give it a little room; I think it works anyway
        x_min -= 5
        y_min -= 5
        width += 10
        height += 10
        bbox.append((x_min, y_min, width, height))
    return bbox


def spiga_process(example, detector):
    ldms = get_landmarks(example, detector)

    if len(ldms) == 0:
        return False

    else:
        image = example
        image = np.array(image)
        # BGR
        image = image[:, :, ::-1]
        bbox = bbox_from_landmarks(ldms)
        features = processor.inference(image, [*bbox])
        landmarks = features["landmarks"]
        spigas = landmarks
        return spigas


# For some reason this map doesn't work with num_proc > 1 :(
# TODO: run inference on GPU


# ## "Segmentation"

# We use bezier paths to draw contours and areas.

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import PIL


def get_patch(landmarks, color='lime', closed=False):
    contour = landmarks
    ops = [Path.MOVETO] + [Path.LINETO] * (len(contour) - 1)
    facecolor = (0, 0, 0, 0)  # Transparent fill color, if open
    if closed:
        contour.append(contour[0])
        ops.append(Path.CLOSEPOLY)
        facecolor = color
    path = Path(contour, ops)
    return patches.PathPatch(path, facecolor=facecolor, edgecolor=color, lw=4)


# Draw to a buffer.

def conditioning_from_landmarks(landmarks_, size=512):
    # Precisely control output image size
    dpi = 72
    fig, ax = plt.subplots(1, figsize=[size / dpi, size / dpi], tight_layout={'pad': 0})
    fig.set_dpi(dpi)

    black = np.zeros((size, size, 3))
    ax.imshow(black)

    for landmarks in landmarks_:
        face_patch = get_patch(landmarks[0:17])
        l_eyebrow = get_patch(landmarks[17:22], color='yellow')
        r_eyebrow = get_patch(landmarks[22:27], color='yellow')
        nose_v = get_patch(landmarks[27:31], color='orange')
        nose_h = get_patch(landmarks[31:36], color='orange')
        l_eye = get_patch(landmarks[36:42], color='magenta', closed=True)
        r_eye = get_patch(landmarks[42:48], color='magenta', closed=True)
        outer_lips = get_patch(landmarks[48:60], color='cyan', closed=True)
        inner_lips = get_patch(landmarks[60:68], color='blue', closed=True)

        ax.add_patch(face_patch)
        ax.add_patch(l_eyebrow)
        ax.add_patch(r_eyebrow)
        ax.add_patch(nose_v)
        ax.add_patch(nose_h)
        ax.add_patch(l_eye)
        ax.add_patch(r_eye)
        ax.add_patch(outer_lips)
        ax.add_patch(inner_lips)

        plt.axis('off')

        fig.canvas.draw()
    buffer, (width, height) = fig.canvas.print_to_buffer()
    assert width == height
    assert width == size

    buffer = np.frombuffer(buffer, np.uint8).reshape((height, width, 4))
    buffer = buffer[:, :, 0:3]
    plt.close(fig)
    return PIL.Image.fromarray(buffer)


def spiga_segmentation(spiga, size):
    landmarks = spiga
    spiga_seg = conditioning_from_landmarks(landmarks, size=size)
    return spiga_seg


if __name__ == '__main__':
    # ## Obtain SPIGA features
    processor = SPIGAFramework(ModelConfig("300wpublic"))
    detector = FaceDetector(weight_path="/share2/zhangyuxuan/project/train_ip_cn/datasets/make_kps/pretrained_models/mobilenet0.25_Final.pth")

    id_folder = "/share2/zhangyuxuan/project/train_ip_cn/test_img_2/id/"
    pose_folder = "/share2/zhangyuxuan/project/train_ip_cn/test_img_2/pose/"

    if not os.path.exists(pose_folder):
        os.makedirs(pose_folder)

    pbar = tqdm.tqdm(os.listdir(id_folder))
    for name in pbar:
        face = Image.open(id_folder+name).convert("RGB").resize((512, 512))
        face.save(id_folder+name)
        spigas = spiga_process(face, detector)
        if spigas == False:
            height = 512
            width = 512
            channels = 3
            black_image = np.zeros((height, width, channels), dtype=np.uint8)
            black_image_cv2 = cv2.cvtColor(black_image, cv2.COLOR_RGB2BGR)
            continue
        else:
            spigas_faces = spiga_segmentation(spigas)
            spigas_faces.save(pose_folder + name)