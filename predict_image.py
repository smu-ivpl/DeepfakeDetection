import argparse
import os
import re
import time

import torch
import pandas as pd
from kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video_set
from training.zoo.classifiers import DeepFakeClassifier, DeepFakeClassifier_Distill, DeepFakeClassifier_Video_Distill
from facenet_pytorch.models.mtcnn import MTCNN
import glob
import cv2
import numpy as np
from PIL import Image
from kernel_utils import isotropically_resize_image, put_to_center, normalize_transform

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Predict test videos")
    arg = parser.add_argument
    arg('--weights-dir', type=str, default="final/Efficient_ViT_Distill/weights/", help="path to directory with checkpoints")
    arg('--models', nargs='+', required=True, help="checkpoint files")
    arg('--test-dir', type=str, required=True, help="path to directory with images")
    arg('--output', type=str, required=False, help="path to output csv", default="submission_image.csv")
    arg('--distill', type=bool, required=False, default=False)
    args = parser.parse_args()

    models = []
    model_paths = [os.path.join(args.weights_dir, model) for model in args.models]
    for path in model_paths:
        if args.distill == True:
            model = DeepFakeClassifier_Distill(encoder="deit_distill_large_patch32_384").to("cuda")
        else:
            model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to("cuda")
        print("loading state dict {}".format(path))
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
        model.eval()
        del checkpoint
        models.append(model.half())


    #image load
    #image detect
    #model
    predictions =[]
    paths = [file for file in glob.glob(args.test_dir+'/*.png')]
    print("Predicting {} images".format(len(paths)))
    #images = [cv2.imread(file) for file in glob.glob(args.test_dir+'/*.png')]
    detector = MTCNN(margin=0, thresholds=[0.7, 0.8, 0.8], device="cuda")
    input_size = 384
    strategy = confident_strategy
    stime = time.time()

    for path in paths:
        frame = cv2.imread(path)
        h, w = frame.shape[:2]
        img = Image.fromarray(frame.astype(np.uint8))
        img = img.resize(size=[s // 2 for s in img.size])
        batch_boxes, probs = detector.detect(img, landmarks=False)
        faces = []
        scores = []
        if batch_boxes is None:
            continue
        for bbox, score in zip(batch_boxes, probs):
            if bbox is not None:
                xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
                w = xmax - xmin
                h = ymax - ymin
                p_h = h // 3
                p_w = w // 3
                crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
                faces.append(crop)
                scores.append(score)

        if len(faces) > 0 :
            x = np.zeros((1, input_size, input_size, 3), dtype=np.uint8)
            n = 0
            for face in faces:
                _,_,landmark = detector.detect(face, landmarks=True)
                resized_face = isotropically_resize_image(face, input_size)
                resized_face = put_to_center(resized_face, input_size)

                x[n] = resized_face
                n += 1
        if n>0:
            x = torch.tensor(x, device="cuda").float()
            x = x.permute((0,3,1,2))
            for i in range(len(x)):
                x[i] = normalize_transform(x[i]/255.)
            with torch.no_grad():
                preds = []
                for model in models:
                    if args.distill:
                        _, y_pred, _ = model(x[:n].half())
                    else:
                        y_pred = model(x[:n].half())
                    y_pred = torch.sigmoid(y_pred.squeeze())
                    bpred = y_pred.cpu().numpy()
                    preds.append(bpred)
                predictions.append(np.mean(preds))


    submission_df = pd.DataFrame({"filename": paths, "label": predictions})
    submission_df.to_csv(args.output, index=False)
    print("Elapsed:", time.time() - stime)
