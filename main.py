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
    config["EPOCHS"]=10
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