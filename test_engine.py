from model.model4 import Model4,build_model4
from model.utils import get_model, save_checkpoint, load_checkpoint
from PIL import Image
import yaml
import numpy as np
img = Image.open(r"D:\\Thesis\\DamnShit\\module_space\\testimg\\1.jpg") 
img=img.convert("RGB")
def yaml_to_dict(path: str):
    with open(path) as f:
        return yaml.load(f.read(), yaml.FullLoader)
config = yaml_to_dict(".\\configs\\train_mot17_coco.yaml")
config["DATA_ROOT"]="D:\\Thesis\\DamnShit\\Hello\\MeMOTR_IKUN\\DATA_DIR"
config["TRAIN_COCO"]="D:\\Thesis\\DamnShit\\Hello\\MeMOTR_IKUN\\outputs\\memotr_mot17_coco\\train\\mot17_train_coco_reforged.json"
config["NO_TRANSFORM"]=True
config["EPOCHS"]=10
config["RESUME"]="D:\\Thesis\\DamnShit\\module_space\\checkpoint_9.pth"

model = build_model4(config=config)
train_states = {
        "start_epoch": 0,
        "global_iters": 0
    }

# Resume
if config["RESUME"] is not None:
    load_checkpoint(model=model, path=config["RESUME"])
model.eval()
sentences=["woman in black pain"]
x=[{
    "local_images":[img],
    "sentences":sentences
}]
out=model(x)

print(out)