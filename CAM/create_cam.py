import os
import argparse

import cv2
import torch
import numpy as np
from torch.nn import functional as F
import torchvision.transforms as transforms
from training.zoo.classifiers import DeepFakeClassifier_Distill, DeepFakeClassifier
from facenet_pytorch.models.mtcnn import MTCNN
import utils
import math
import model
import re
from torchvision.transforms import Normalize

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)
def create_cam(config):
    if not os.path.exists(config.result_path):
        os.mkdir(config.result_path)
    
    test_loader, num_class = utils.get_testloader(config.dataset,
                                        config.dataset_path,
                                        config.img_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cnn = DeepFakeClassifier_Distill(encoder="deit_distill_large_patch32_384").to("cuda")#DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to("cuda")
    #cnn = model.CNN(img_size=config.img_size, num_class=num_class).to(device)
    checkpoint = torch.load(os.path.join(config.model_path, config.model_name), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    cnn.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
    '''
    cnn.load_state_dict(
        torch.load(os.path.join(config.model_path, config.model_name))
    )
    '''
    finalconv_name = 'encoder'

    # hook
    feature_blobs = []
    def hook_feature(module, input, output):
        feature_blobs.append(output.cpu().data.numpy())


    cnn.encoder.blocks[-1].norm1.register_forward_hook(hook_feature)#cnn._modules.get(finalconv_name)._modules.get('conv_head').register_forward_hook(hook_feature)
    params = list(cnn.encoder.blocks[-1].norm1.parameters())
    #params = list(torch.nn.Sequential(*list(cnn.children())[:-3]).parameters())
    # get weight only from the last layer(linear)
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    def returnCAM(feature_conv, weight_softmax, class_idx):
        size_upsample = (config.img_size, config.img_size)
        nc, h, w = feature_conv.shape
        output_cam = []
        cam = weight_softmax.dot(feature_conv.reshape((w, h*nc))) # [class_idx]
        cam = cam.reshape(h, nc)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    detector = MTCNN(margin=0, thresholds=[0.7, 0.8, 0.8], device="cuda")
    for i, (image_tensor, label) in enumerate(test_loader):
        image_PIL = transforms.ToPILImage()(image_tensor[0])
        image_PIL.save(os.path.join(config.result_path, 'img%d.png' % (i + 1)))

        image_tensor = image_tensor.to(device)

        #eye
        '''
        try:
            _, _, landmark = detector.detect(image_PIL, landmarks=True)
            image = np.around(image_PIL).astype(np.int16)
            landmark = np.around(landmark[0]).astype(np.int16)
            (x1, y1), (x2, y2) = landmark[:2]
            w = dist((x1, y1), (x2, y2))
            dilation = int(w // 4)
            eye_image = image[y2 - dilation:y1 + dilation, x1 - dilation:x2 + dilation]
            eye_image = cv2.resize(eye_image, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
        except Exception as ex:
            eye_image = cv2.resize(image, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)

        eye_image = torch.tensor(eye_image, device="cuda").float()
        eye_image = eye_image.permute((2, 0, 1))
        eye_image = normalize_transform(eye_image / 255.)
        eye_image = eye_image.to(device)
        eye_image = eye_image.view(1, 3, 32, 32)
        '''
        #
        logit = cnn(image_tensor)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        print("True label : %d, Predicted label : %d, Probability : %.2f" % (label.item(), idx.item(), probs.item()))
        CAMs = returnCAM(feature_blobs[0], weight_softmax, [idx.item()])
        img = cv2.imread(os.path.join(config.result_path, 'img%d.png' % (i + 1)))
        height, width,_ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite(os.path.join(config.result_path, 'cam%d.png' % (i + 1)), result)
        #if i + 1 == config.num_result:
        #    break
        feature_blobs.clear()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='OWN', choices=['STL', 'CIFAR', 'OWN'])
    parser.add_argument('--dataset_path', type=str, default='./images')
    parser.add_argument('--model_path', type=str, default='/home/yjheo/Deepfake/dfdc_deepfake_challenge/weights/')
    parser.add_argument('--model_name', type=str, default='deit_d_555_eyeDeepFakeClassifier_Distill_deit_distill_large_patch32_384_0_best_dice')
    parser.add_argument('--result_path', type=str, default='./result')
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--num_result', type=int, default=1)

    config = parser.parse_args()
    print(config)

    create_cam(config)