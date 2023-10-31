import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from skimage.feature import local_binary_pattern
from torchvision import transforms


class MyResnet50(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        self.model = models.resnet50(weights='IMAGENET1K_V2')
        self.modules = list(self.model.children())[:-1]
        self.model = nn.Sequential(*self.modules)
        self.model = self.model.eval()
        self.model = self.model.to(device)
        self.shape = 2048

    def extract_features(self, image):
        transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        image = transform(image)

        with torch.no_grad():
            feature = self.model(image)
            feature = torch.flatten(feature, start_dim=1)

        return feature.cpu().detach().numpy()


class LBP():
    def __init__(self):
        self.shape = 26

    def extract_features(self, image):
        n_points = 24
        radius = 3

        image = image.cpu().numpy()
        features = []
        for img in image:
            img *= 255
            img = img.reshape(img.shape[1], img.shape[2], img.shape[0])

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lbp = local_binary_pattern(gray, n_points, radius, method="default")
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            hist = hist.astype("float32")
            hist /= (hist.sum() + 1e-7)

            features.append(hist)

        return np.array(features)


class SIFT2():
    def __init__(self):
        self.shape = 128

    def extract_features(self, image):
        image = image.cpu().numpy()
        features = []
        sift = cv2.SIFT_create()

        for img in image:
            img *= 255
            img = img.astype(np.uint8)
            img = img.transpose(1, 2, 0)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            keypoints, descriptors = sift.detectAndCompute(gray, None)

            if descriptors is not None:
                feature_vector = descriptors.flatten()
                feature_vector.resize(self.shape)
                features.append(feature_vector)
            else:
                features.append(np.zeros(self.shape))

        return np.array(features)

class SIFT():
    def __init__(self, nfeatures=0, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6):
        self.shape = 128
        self.nfeatures = nfeatures
        self.contrastThreshold = contrastThreshold
        self.edgeThreshold = edgeThreshold
        self.sigma = sigma

    def extract_features(self, image):
        image = image.cpu().numpy()
        features = []
        sift = cv2.SIFT_create(nfeatures=self.nfeatures, contrastThreshold=self.contrastThreshold, edgeThreshold=self.edgeThreshold)

        for img in image:
            img *= 255
            img = img.astype(np.uint8)
            img = img.transpose(1, 2, 0)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            keypoints = []
            descriptors = []
            for scale in [0.5, 1.0, 2.0]:
                scaled_img = cv2.resize(gray, None, fx=scale, fy=scale)
                kp, des = sift.detectAndCompute(scaled_img, None)
                if des is not None:
                    keypoints.extend(kp)
                    descriptors.extend(des)

            if descriptors:
                descriptors = np.array(descriptors)
                feature_vector = descriptors.flatten()
                feature_vector.resize(self.shape)
                features.append(feature_vector)
            else:
                features.append(np.zeros(self.shape))

        return np.array(features)