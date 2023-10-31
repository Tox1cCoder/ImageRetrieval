import os
import pathlib
import time
from argparse import ArgumentParser

import torch
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

from src.dataloader import get_transformation
from src.feature_extraction import MyResnet50, LBP, SIFT

ACCEPTED_IMAGE_EXTS = ['.jpg', '.png']

query_root = './dataset/groundtruth'
image_root = './dataset/paris'
feature_root = './dataset/feature'
evaluate_root = './dataset/evaluation'


def get_image_list(image_root):
    image_root = pathlib.Path(image_root)
    image_list = list()
    for image_path in os.listdir(image_root):
        image_list.append(image_path[:-4])
    image_list = sorted(image_list, key=lambda x: x)
    return image_list


def main():
    parser = ArgumentParser()
    parser.add_argument("--feature_extractor", required=True, type=str, default='Resnet50')
    parser.add_argument("--device", required=False, type=str, default='cuda:0')
    parser.add_argument("--top_k", required=False, type=int, default=11)
    parser.add_argument("--crop", required=False, type=bool, default=False)

    print('Ranking .......')
    start = time.time()

    args = parser.parse_args()
    device = torch.device(args.device)

    if args.feature_extractor == 'Resnet50':
        extractor = MyResnet50(device)
    elif args.feature_extractor == 'LBP':
        extractor = LBP()
    elif args.feature_extractor == 'SIFT':
        extractor = SIFT()
    else:
        print("No matching model found")
        return

    img_list = get_image_list(image_root)
    transform = get_transformation()

    for path_file in os.listdir(query_root):
        if path_file[-9:-4] == 'query':
            rank_list = []

            with open(query_root + '/' + path_file, "r") as file:
                img_query, left, top, right, bottom = file.read().split()

            test_image_path = pathlib.Path('./dataset/paris/' + img_query + '.jpg')
            pil_image = Image.open(test_image_path)
            pil_image = pil_image.convert('RGB')

            path_crop = 'original'
            if args.crop:
                pil_image = pil_image.crop((float(left), float(top), float(right), float(bottom)))
                path_crop = 'crop'

            image_tensor = transform(pil_image)
            image_tensor = image_tensor.unsqueeze(0).to(device)
            feat = extractor.extract_features(image_tensor)

            n_clusters = 256
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(feat)

            cluster_assignment = kmeans.predict(feat)
            cluster_indices = np.where(kmeans.labels_ == cluster_assignment)[0]
            distances = np.linalg.norm(feat[cluster_indices] - feat, axis=1)
            sorted_indices = np.argsort(distances)
            top_k_indices = cluster_indices[sorted_indices[:args.top_k]]

            for index in top_k_indices:
                rank_list.append(str(img_list[index]))

            with open(
                evaluate_root + '/' + path_crop + '/' + args.feature_extractor + '/' + path_file[:-10] + '.txt', "w") as file:
                file.write("\n".join(rank_list))

    end = time.time()
    print('Finish in ' + str(end - start) + ' seconds')


if __name__ == '__main__':
    main()
