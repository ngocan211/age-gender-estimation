import cv2
import dlib
import numpy as np
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from src.factory import get_model
from tensorflow.keras.utils import get_file


def cropface(img_, d_, img_size_, img_h, img_w):
    margin = 0.4
    x1, y1, x2, y2, w, h = d_.left(), d_.top(), d_.right() + 1, d_.bottom() + 1, d_.width(), d_.height()
    xw1 = max(int(x1 - margin * w), 0)
    yw1 = max(int(y1 - margin * h), 0)
    xw2 = min(int(x2 + margin * w), img_w - 1)
    yw2 = min(int(y2 + margin * h), img_h - 1)
    cv2.rectangle(img_, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
    return cv2.resize(img_[yw1:yw2 + 1, xw1:xw2 + 1], (img_size_, img_size_))


pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
modhash = '6d7f7b7ced093a8b3ef6399163da6ece'

detector = dlib.get_frontal_face_detector()
weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pretrained_model, cache_subdir="pretrained_models",
                       file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))
model_name, img_size = Path(weight_file).stem.split("_")[:2]
img_size = int(img_size)
cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
model = get_model(cfg)
model.load_weights(weight_file)


def predict(image_path):
    img = cv2.imread(str(image_path), 1)
    if img is not None:
        h, w, _ = img.shape
        r = 640 / max(w, h)
        img = cv2.resize(img, (int(w * r), int(h * r)))

    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detected = detector(input_img, 1)
    # faces = np.empty((len(detected), img_size, img_size, 3))
    if len(detected) > 0:
        img_h, img_w, _ = np.shape(input_img)
        faces = [cropface(img, d, img_size, img_h, img_w) for i, d in enumerate(detected)]

        # predict ages and genders of the detected faces
        results = model.predict(faces)
        predicted_genders = results[0]
        predicted_ages = results[1].dot(np.arange(0, 101).reshape(101, 1)).flatten()

    # draw results
    for i, _ in enumerate(predicted_ages):
        label = "{}, {}".format(int(predicted_ages[i]), "M" if predicted_genders[i][0] < 0.5 else "F")
        print(label)


def get_input_lines():
    import sys
    lines = []
    for input_line in sys.stdin:
        input_line = input_line.strip()
        lines.append(input_line)
    return lines


pbar = tqdm(get_input_lines())
# pbar = tqdm(['/home/ubuntu/acis/irec/photo/./photo3/00/0000056f370350e2f401277a78127dbc_m.jpg'])
for line_ in pbar:
    pbar.set_description('predict %s' % line_)
    image_path = line_
    img = cv2.imread(str(image_path), 1)
    margin = 0.4
    if img is not None:
        h, w, _ = img.shape
        r = 640 / max(w, h)
        img = cv2.resize(img, (int(w * r), int(h * r)))

    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = np.shape(input_img)

    detected = detector(input_img, 1)
    faces = np.empty((len(detected), img_size, img_size, 3))

    if len(detected) > 0:
        for i, d in enumerate(detected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), img_w - 1)
            yw2 = min(int(y2 + margin * h), img_h - 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
            faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))

        # predict ages and genders of the detected faces
        results = model.predict(faces)
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()

        # draw results
        for i, d in enumerate(detected):
            label = "{},{}".format(int(predicted_ages[i]),
                                   "M" if predicted_genders[i][0] < 0.5 else "F")
            print("%s,%s" % (line_, label))
