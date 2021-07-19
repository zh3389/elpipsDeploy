import imageio
import requests
import numpy as np


def predict(img1, img2):
    payload = {"instances": [{'input_1': img1.tolist(), "input_2": img2.tolist()}]}
    r = requests.post('http://192.168.31.220:8501/v1/models/docker_test:predict', json=payload)
    score = r.json()["predictions"][0]
    return score


if __name__ == '__main__':
    image1 = imageio.imread("img/cat.png")[:, :, 0:3].astype(np.float32) / 255.0
    image2 = imageio.imread("img/dog.png")[:, :, 0:3].astype(np.float32) / 255.0
    score = predict(image1, image2)
    print(score)