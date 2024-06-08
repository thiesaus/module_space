import yaml
# from model.filter_module import FilterModule
from train_engine import train


def yaml_to_dict(path: str):
    with open(path) as f:
        return yaml.load(f.read(), yaml.FullLoader)


if __name__ == "__main__":
    # filter_module = FilterModule()

    config = yaml_to_dict(".\\configs\\train_mot17_coco.yaml")
    config["DATA_ROOT"]="D:\\Thesis\\DamnShit\\Hello\\MeMOTR_IKUN\\DATA_DIR"
    config["TRAIN_COCO"]="D:\\Thesis\\DamnShit\\Hello\\MeMOTR_IKUN\\outputs\\memotr_mot17_coco\\train\\mot17_train_coco_reforged.json"
    config["NO_TRANSFORM"]=True
    config["EPOCHS"]=200
    config["EPOCHS_SPACE"]=20
    config["LR_SCHEDULER"] == "Cosine"
    # ik
    config["train_bs"]=10
    config["test_bs"]=1
    config["num_workers"]=4
    config["img_hw"]=[(224, 224), (448, 448), (672, 672)]
    config["random_crop_ratio"]=[0.8, 1.0]
    config["norm_mean"]=[0.48145466, 0.4578275, 0.40821073]
    config["norm_std"]=[0.26862954, 0.26130258, 0.27577711]
    config["rf_kitti_json"]="C:\\Users\\phamp\\Desktop\\module_space\\outputs\\Refer-KITTI_labels.json"
    config["rf_expression"]="D:\\Thesis\\DamnShit\\Hello\\MeMOTR_IKUN\\DATA_DIR\\Refer_Kitti\\expression"
    config["sample_frame_len"]=8
    config["sample_frame_num"]=2
    config["data_root"]="D:\\Thesis\\DamnShit\\Hello\\MeMOTR_IKUN\\DATA_DIR\\Refer_Kitti"
    config["sample_expression_num"]=1
    config["sample_frame_stride"]=4
    config["track_root"]=""
    config["NUM_LAYERS"]=[4,4,4,4]
    config["TEST_DIST"]=1
    config["WANDB"]=False
    # config["RESUME"]="C:\\Users\\phamp\\Desktop\\module_space\\checkpoint_99.pth"


    # config["RESUME"]="D:\\Thesis\\DamnShit\\module_space\\checkpoint_9.pth"
    # config["SUBSET_LENGTH"]=0.2
    # config["GET_DATA_SUBSET"]=True
    # dataset_train = build_dataset(config=config, split="train")
    # sampler_train = build_sampler(dataset=dataset_train, shuffle=True)
    # dataloader_train = build_dataloader(dataset=dataset_train, sampler=sampler_train,
    #                                     batch_size=config["BATCH_SIZE"], num_workers=config["NUM_WORKERS"])
    # for i, batch in enumerate(dataloader_train):
    #     if i>1: break
    #     check=convert_data(batch)
    #     logits= filter_module(check)
    #     plotting(check,logits)

    train(config=config)