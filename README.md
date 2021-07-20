## 基于E-LPIPS的感知图像相似性

**如果本项目对你部署一个比较两张图像相似的模型有帮助，欢迎Start...**

#### 环境安装

```bash
pip install -r requirements.txt
```

#### 部署已打包好的模型 需提前装好Docker

```bash
# 命令行执行
chmod +x cpuDeploy.sh
chmod +x gpuDeploy.sh
./cpuDeploy.sh  或者  ./gpuDeploy.sh
```

#### 如何进行两张图像相似度比较

```python
import imageio
import requests
import numpy as np


def predict(img1, img2):
    payload = {"instances": [{'input_1': img1.tolist(), "input_2": img2.tolist()}]}
    r = requests.post('http://localhost:8501/v1/models/docker_test:predict', json=payload)
    score = r.json()["predictions"][0]
    return score


if __name__ == '__main__':
    image1 = imageio.imread("img/cat.png")[:, :, 0:3].astype(np.float32) / 255.0
    image2 = imageio.imread("img/dog.png")[:, :, 0:3].astype(np.float32) / 255.0
    score = predict(image1, image2)
    print(score)
```

