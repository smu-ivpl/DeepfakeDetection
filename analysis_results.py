from torch import nn
import torch
import pandas as pd
import collections
import numpy as np
import cv2
from kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video_set
import os

test_path = '/home/yjheo/Deepfake/dataset/dfdc_facebook/test/labels.csv'
test_dir = '/home/yjheo/Deepfake/dataset/dfdc_facebook/test'
label = pd.read_csv(test_path)
predict = pd.read_csv('eye.csv')

label_v = torch.from_numpy(label['label'].values).unsqueeze(dim=1).float()
predict_v = torch.from_numpy(predict['label'].values).unsqueeze(dim=1).float()

file_name = predict['filename'].values

real = 0
real_v = 0
fake = 0
fake_v = 0
dict = collections.defaultdict(float)
for i in range(len(label_v)):
    if abs(label_v[i] - predict_v[i]) >= 0.5:
        if label_v[i] == 0 :
            real +=1
            real_v += abs(label_v[i] - predict_v[i])
        else:
            fake +=1
            fake_v += abs(label_v[i] - predict_v[i])
        dict[file_name[i]] = abs(label_v[i] - predict_v[i])
print(real, fake)
print(real_v/real, fake_v/fake)
dict = sorted(dict.items(), key = lambda x: x[1], reverse=True)
print(len(dict))
for i in range(len(dict)):
    print(dict[i])

def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized

def put_to_center(img, input_size):
    img = img[:input_size, :input_size]
    image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    start_w = (input_size - img.shape[1]) // 2
    start_h = (input_size - img.shape[0]) // 2
    image[start_h:start_h + img.shape[0], start_w: start_w + img.shape[1], :] = img
    return image
'''
video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=12)
face_extractor = FaceExtractor(video_read_fn)

for i in range(50):
    video_path = os.path.join(test_dir, dict[i][0])
    faces = face_extractor.process_video(video_path)
    directory = os.path.join('analysis', dict[i][0].split('.')[0])
    if not os.path.exists(directory):
        os.makedirs(directory)
    if len(faces) > 0:
        for j, frame_data in enumerate(faces):
            for k, face in enumerate(frame_data["faces"]):
                resized_face = isotropically_resize_image(face, 384)
                resized_face = put_to_center(resized_face, 384)
                cv2.imwrite(os.path.join('analysis', dict[i][0].split('.')[0], str(j)+'_'+str(k)+'.png'), cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR))

'''