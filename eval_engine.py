# @Author       : Ruopeng Gao
# @Date         : 2022/11/21

import os
import yaml

from torch.utils import tensorboard as tb

from utils.utils import convert_data

from utils.train_visualize import Visualize
from model.criterion import ModuleCriterion

def eval_model(model: str,visualizer:Visualize, dataloader: str,epoch:int):
    print(f"===>  Running eval epoch '{epoch}'")

    loss= {
        "mse_loss":[],
        "mae_loss": [],
    }
    for i, batch in enumerate(dataloader):
        run=True
        datas=convert_data(batch)
        for data in datas:
            if len(data["local_images"])==0:
                run=False
                break
        if not run:
            continue
        model_outputs= model(datas)
        output=model_outputs["logits"]
        for out in output:
            mse_loss = ModuleCriterion.get_mse_loss(outputs=out)
            mae_loss = ModuleCriterion.get_mae_loss(outputs=out)
            loss["mse_loss"].append(mse_loss)
            loss["mae_loss"].append(mae_loss)

    avg_mse_loss = sum(loss["mse_loss"]) / len(loss["mse_loss"])
    avg_mae_loss = sum(loss["mae_loss"]) / len(loss["mae_loss"])

    print(f"===>  Eval epoch '{epoch}' finished, mse_loss: {avg_mse_loss}, mae_loss: {avg_mae_loss}")
    visualizer.add_loss({"mse_loss":avg_mse_loss,"mae_loss":avg_mae_loss})
    visualizer.plot_loss()
    
    return
    # print(f"===>  Running checkpoint '{model}'")

    # if threads > 1:
    #     os.system(f"python -m torch.distributed.run --nproc_per_node={str(threads)} --master_port={port} "
    #               f"main.py --mode submit --submit-dir {eval_dir} --submit-model {model} "
    #               f"--data-root {data_root} --submit-data-split {data_split} "
    #               f"--use-distributed --config-path {config_path}")
    # else:
    #     os.system(f"python main.py --mode submit --submit-dir {eval_dir} --submit-model {model} "
    #               f"--data-root {data_root} --submit-data-split {data_split} --config-path {config_path}")

    # # 将结果移动到对应的文件夹
    # tracker_dir = os.path.join(eval_dir, data_split, "tracker")
    # tracker_mv_dir = os.path.join(eval_dir, data_split, model.split(".")[0] + "_tracker")
    # os.system(f"mv {tracker_dir} {tracker_mv_dir}")

    # # 进行指标计算
    # data_dir = os.path.join(data_root, dataset_name)
    # if dataset_name == "DanceTrack" or dataset_name == "SportsMOT":
    #     gt_dir = os.path.join(data_dir, data_split)
    # elif "MOT17" in dataset_name:
    #     gt_dir = os.path.join(data_dir, "images", data_split)
    # else:
    #     raise NotImplementedError(f"Eval Engine DO NOT support dataset '{dataset_name}'")
    # if dataset_name == "DanceTrack" or dataset_name == "SportsMOT":
    #     os.system(f"python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {data_split}  "
    #               f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} "
    #               f"--SEQMAP_FILE {os.path.join(data_dir, f'{data_split}_seqmap.txt')} "
    #               f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
    #               f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
    #               f"--TRACKERS_FOLDER {tracker_mv_dir}")
    # elif "MOT17" in dataset_name:
    #     if "mot15" in data_split:
    #         os.system(f"python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {data_split}  "
    #                   f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} "
    #                   f"--SEQMAP_FILE {os.path.join(data_dir, f'{data_split}_seqmap.txt')} "
    #                   f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
    #                   f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
    #                   f"--TRACKERS_FOLDER {tracker_mv_dir} --BENCHMARK MOT15")
    #     else:
    #         os.system(f"python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {data_split}  "
    #                   f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} "
    #                   f"--SEQMAP_FILE {os.path.join(data_dir, f'{data_split}_seqmap.txt')} "
    #                   f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
    #                   f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
    #                   f"--TRACKERS_FOLDER {tracker_mv_dir} --BENCHMARK MOT17")
    # else:
    #     raise NotImplementedError(f"Do not support this Dataset name: {dataset_name}")

    # metric_path = os.path.join(tracker_mv_dir, "pedestrian_summary.txt")
    # with open(metric_path) as f:
    #     metric_names = f.readline()[:-1].split(" ")
    #     metric_values = f.readline()[:-1].split(" ")
    # metrics = {
    #     n: float(v) for n, v in zip(metric_names, metric_values)
    # }
    # return metrics


def metrics_to_tensorboard(writer: tb.SummaryWriter, metrics: dict, epoch: int):
    for k, v in metrics.items():
        writer.add_scalar(tag=k, scalar_value=v, global_step=epoch)
    return
