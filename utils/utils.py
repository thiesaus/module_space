# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Some utils.
import os
import yaml
import torch
import random
import torch.distributed
import torch.backends.cudnn
import numpy as np


def is_distributed():
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return False
    return True


def distributed_rank():
    if not is_distributed():
        return 0
    else:
        return torch.distributed.get_rank()


def is_main_process():
    return distributed_rank() == 0


def distributed_world_size():
    if is_distributed():
        return torch.distributed.get_world_size()
    else:
        return 1


def set_seed(seed: int):
    seed = seed + distributed_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # If you don't want to wait until the universe is silent, do not use this below code :)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    return


def yaml_to_dict(path: str):
    with open(path) as f:
        return yaml.load(f.read(), yaml.FullLoader)


def labels_to_one_hot(labels: np.ndarray, class_num: int):
    return np.eye(N=class_num)[labels]


def inverse_sigmoid(x, eps=1e-5):
    """
    if      x = 1/(1+exp(-y))
    then    y = ln(x/(1-x))
    Args:
        x:
        eps:

    Returns:
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def convert_data(temp):
    output=[]
    for i,img in enumerate(temp["imgs"][0]):
        output_dict=temp["infos"][0][i]
        bboxes= output_dict["boxes"]
        output_dict["local_images"]=[img.crop(box.numpy()) for box in bboxes]
        output_dict["global_image"]=img 
        output_dict["sentences"]=temp["sentence"]
        output.append(output_dict)
    return output

def plotting(x,logit):
    count = len(x["sentences"][:10])

    plt.figure(figsize=(20, 20))
    plt.imshow(logit, vmin=-1, vmax=1)
    # plt.colorbar()
    plt.yticks(range(count), x["sentences"][:10], fontsize=10)
    plt.xticks([])
    # for i, image in enumerate(x["local_images"][:10]):
    #     plt.imshow(image.resize((400,400)), extent=(i + 0.5, i - 0.5, 1.6, 0.6), origin="upper")
    for x in range(len(logit)):
        for y in range(len(logit[x])):
            plt.text(x, y, f"{logit[y][x]:.2f}", ha="center", va="center", size=10)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])

    plt.title("Cosine similarity between text and image features", size=20)
