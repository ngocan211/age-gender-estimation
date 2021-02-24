import logging

from flask import Flask, render_template, jsonify

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.ERROR)

app = Flask(__name__, static_folder='age-files')


# def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
#                font_scale=0.8, thickness=1):
#     size = cv2.getTextSize(label, font, font_scale, thickness)[0]
#     x, y = point
#     cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
#     cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
#
#
# pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
# modhash = '6d7f7b7ced093a8b3ef6399163da6ece'
#
# detector = dlib.get_frontal_face_detector()
# weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pretrained_model, cache_subdir="pretrained_models",
#                        file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))
# model_name, img_size = Path(weight_file).stem.split("_")[:2]
# img_size = int(img_size)
# cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
# model = get_model(cfg)
# model.load_weights(weight_file)


@app.route('/')
def index():
    return jsonify({'message': 'ACWork Image Search 1.0.0'})


@app.route('/age', methods=['GET'])
def age():
    return render_template('posts3.html', end_point='/age2', slice_pos='3')


# @app.route('/age2', methods=['GET'])
# def age2():
#     # url = 'https://ac-ai-services.s3-ap-northeast-1.amazonaws.com/ac-image-search/4a584287-cd22-472d-afea-c2aa5ac4b769/original.jpeg'
#     object_id = urllib.parse.unquote(request.args.get('object_id', ''))
#     offset = object_id.find('/')
#     s3_bucket = object_id[:offset]
#     s3_key = object_id[offset + 1:]
#     url = f'https://{s3_bucket}.s3-ap-northeast-1.amazonaws.com/{s3_key}'
#
#     resp = urllib.request.urlopen(url)
#     image = np.asarray(bytearray(resp.read()), dtype="uint8")
#     img = cv2.imdecode(image, cv2.IMREAD_COLOR)
#
#     # image_path = '/Users/acworks/PycharmProjects/age-gender-estimation/test/59ee12f8077e6c026366914437beaa42_t.jpeg'
#     # img = cv2.imread(str(image_path), 1)
#     margin = 0.4
#     if img is not None:
#         h, w, _ = img.shape
#         r = 640 / max(w, h)
#         img = cv2.resize(img, (int(w * r), int(h * r)))
#
#     input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_h, img_w, _ = np.shape(input_img)
#
#     detected = detector(input_img, 1)
#     faces = np.empty((len(detected), img_size, img_size, 3))
#
#     if len(detected) > 0:
#         for i, d in enumerate(detected):
#             x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
#             xw1 = max(int(x1 - margin * w), 0)
#             yw1 = max(int(y1 - margin * h), 0)
#             xw2 = min(int(x2 + margin * w), img_w - 1)
#             yw2 = min(int(y2 + margin * h), img_h - 1)
#             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
#             faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))
#
#         # predict ages and genders of the detected faces
#         results = model.predict(faces)
#         predicted_genders = results[0]
#         ages = np.arange(0, 101).reshape(101, 1)
#         predicted_ages = results[1].dot(ages).flatten()
#
#         # draw results
#         for i, d in enumerate(detected):
#             label = "{}, {}".format(int(predicted_ages[i]),
#                                     "M" if predicted_genders[i][0] < 0.5 else "F")
#             draw_label(img, (d.left(), d.top()), label)
#     output_path = "/age-files/%s.jpg" % str(uuid.uuid4())
#     cv2.imwrite("." + output_path, img)
#     return redirect(output_path, code=302)
#     # return jsonify({'message': 'Age Detection'})


@app.route('/age-range', methods=['GET'])
def age_range():
    return render_template('age-range.html')


if __name__ == "__main__":
    import sys

    port = (sys.argv[1:] + ["5000"])[0]
    app.run(host="0.0.0.0", port=int(port), debug=True, use_reloader=False)
