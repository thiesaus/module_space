# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Logger will log information.
import os
import json
import argparse
import yaml

from tqdm import tqdm
from typing import List, Any
from torch.utils import tensorboard as tb

from utils.utils import is_main_process

# @Author       : Ruopeng Gao
# @Date         : 2022/7/13
# @Description  :
import torch.distributed

from collections import deque, defaultdict
from utils.utils import is_distributed, distributed_world_size


class Value:
    def __init__(self, window_size: int = 100):
        self.value_deque = deque(maxlen=window_size)
        self.total_value = 0.0
        self.total_count = 0

        self.value_sync: None | torch.Tensor = None
        self.total_value_sync = None
        self.total_count_sync = None

    def update(self, value):
        self.value_deque.append(value)
        self.total_value += value
        self.total_count += 1

    def sync(self):
        if is_distributed():
            torch.distributed.barrier()
            value_list_gather = [None] * distributed_world_size()
            value_count_gather = [None] * distributed_world_size()
            torch.distributed.all_gather_object(value_list_gather, list(self.value_deque))
            torch.distributed.all_gather_object(value_count_gather, [self.total_value, self.total_count])
            value_list = [v for v_list in value_list_gather for v in v_list]
            self.value_sync = torch.as_tensor(value_list)
            self.total_value_sync = sum([_[0] for _ in value_count_gather])
            self.total_count_sync = int(sum([_[1] for _ in value_count_gather]))
        else:
            self.value_sync = torch.as_tensor(list(self.value_deque))
            self.total_value_sync = self.total_value
            self.total_count_sync = self.total_count
        return

    @property
    def avg(self):
        self.check_sync()
        return self.value_sync.mean().item()

    @property
    def global_avg(self):
        self.check_sync()
        return self.total_value_sync / self.total_count_sync

    def check_sync(self):
        if self.value_sync is None:
            raise RuntimeError(f"Be sure to use .sync() before metric statistic.")
        return


class MetricLog:
    def __init__(self):
        self.metrics = defaultdict(Value)

    def update(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.metrics[name].update(value)
        return

    def sync(self):
        for name, value in self.metrics.items():
            value.sync()
        return

    def get(self, name: str, mode: str):
        return self.metrics[name].__getattribute__(mode)

    def __str__(self):
        s = str()
        if "total_loss" in self.metrics:
            s += f"loss = {self.metrics['total_loss'].avg:.4f} ({self.metrics['total_loss'].global_avg:.4f}); "
        for name, value in self.metrics.items():
            if name == "time per iter":
                continue
            if name == "total_loss":
                continue
            s += f"{name} = {value.avg:.4f} ({value.global_avg:.4f}); "
        return s
    def total_loss(self):
        return f"loss = {self.metrics['total_loss'].avg:.4f} "
    

    def get_avg(self):
        return self.metrics["total_loss"].avg

    def get_global_avg(self):
        return self.metrics["total_loss"].global_avg

def merge_dicts(dicts: List[dict]) -> dict:
    merged = dict()
    for d in dicts:
        for k, v in d.items():
            if k not in merged.keys():
                merged[k] = list()
            merged[k] += v
    return merged


class ProgressLogger:
    def __init__(self, total_len: int, head: str = None, only_main: bool = True):
        self.only_main = only_main
        if (self.only_main and is_main_process()) or (self.only_main is False):
            self.total_len = total_len
            self.tqdm = tqdm(total=total_len)
            self.head = head
        else:
            self.total_len = None
            self.tqdm = None
            self.head = None

    def update(self, step_len: int, **kwargs: Any):
        if (self.only_main and is_main_process()) or (self.only_main is False):
            self.tqdm.set_description(self.head)
            self.tqdm.set_postfix(**kwargs)
            self.tqdm.update(step_len)
        else:
            return


class Logger:
    """
    Log information.
    """
    def __init__(self, logdir: str, only_main: bool = True):
        self.only_main = only_main
        if (self.only_main and is_main_process()) or (self.only_main is False):
            self.logdir = logdir
            os.makedirs(self.logdir, exist_ok=True)
            # os.makedirs(os.path.join(self.logdir, "tb_log"), exist_ok=True)
            self.tb_iters_logger: tb.SummaryWriter = tb.SummaryWriter(log_dir=os.path.join(self.logdir, "tb_iters_log"))
            self.tb_epochs_logger: tb.SummaryWriter = tb.SummaryWriter(log_dir=os.path.join(self.logdir, "tb_epochs_log"))
        else:
            self.logdir = None
            self.tb_iters_logger: tb.SummaryWriter | None = None
            self.tb_epochs_logger: tb.SummaryWriter | None = None
        return

    def show(self, head: str = "", log: MetricLog = ""):
        if (self.only_main and is_main_process()) or (self.only_main is False):
            string=log
            if isinstance(log, MetricLog):
                string = log.total_loss()
            print(f"{head} {string}")
        else:
            pass
        return

    def write(self, head: str = "", log: dict | str | MetricLog = "", filename: str = "log.txt", mode: str = "a"):
        """
        Logger write a log to a file.

        Args:
            head: Log head like self.show.
            log: A log.
            filename: Write file name.
            mode: Open file with this mode.
        """
        if (self.only_main and is_main_process()) or (self.only_main is False):
            if isinstance(log, dict):
                if head != "":
                    raise Warning("Log is a dict, Do not support 'head' attr.")
                if len(filename) > 5 and filename[-5:] == ".yaml":
                    self.write_dict_to_yaml(log, filename, mode)
                elif len(filename) > 5 and filename[-5:] == ".json":
                    self.write_dict_to_json(log, filename, mode)
                elif len(filename) > 4 and filename[-4:] == ".txt":
                    self.write_dict_to_json(log, filename, mode)
                else:
                    raise RuntimeError("Filename '%s' is not supported for dict log." % filename)
            elif isinstance(log, MetricLog):
                with open(os.path.join(self.logdir, filename), mode=mode) as f:
                    f.write(f"{head} {log}\n")
            elif isinstance(log, str):
                with open(os.path.join(self.logdir, filename), mode=mode) as f:
                    f.write(f"{head} {log}\n")
            else:
                raise RuntimeError("Log type '%s' is not supported." % type(log))
        else:
            pass
        return

    def write_dict_to_yaml(self, log: dict, filename: str, mode: str = "w"):
        """
        Logger writes a dict log to a .yaml file.

        Args:
            log: A dict log.
            filename: A yaml file's name.
            mode: Open with this mode.
        """
        with open(os.path.join(self.logdir, filename), mode=mode) as f:
            yaml.dump(log, f, allow_unicode=True)
        return

    def write_dict_to_json(self, log: dict, filename: str, mode: str = "w"):
        """
        Logger writes a dict log to a .json file.

        Args:
            log (dict): A dict log.
            filename (str): Log file's name.
            mode (str): File writing mode, "w" or "a".
        """
        with open(os.path.join(self.logdir, filename), mode=mode) as f:
            f.write(json.dumps(log, indent=4))
            f.write("\n")
        return

    def tb_add_scalar(self, tag: str, scalar_value: float, global_step: int, mode: str):
        if (self.only_main and is_main_process()) or (self.only_main is False):
            if mode == "iters":
                writer: tb.SummaryWriter = self.tb_iters_logger
            else:
                writer: tb.SummaryWriter = self.tb_epochs_logger
            writer.add_scalar(
                tag=tag,
                scalar_value=scalar_value,
                global_step=global_step
            )
        return

    def tb_add_metric_log(self, log: MetricLog, steps: int, mode: str):
        if (self.only_main and is_main_process()) or (self.only_main is False):
            log_keys = log.metrics.keys()
            cross_image_text_keys, cross_text_image_keys = [], []
            for k in log_keys:
                if "cross_image_text" in k:
                    cross_image_text_keys.append(k)  # like "frame0_box_l1_loss"
                elif "cross_text_image" in k:
                    cross_text_image_keys.append(k)
                else:
                    pass
            if mode == "iters":
                writer: tb.SummaryWriter = self.tb_iters_logger
            else:
                writer: tb.SummaryWriter = self.tb_epochs_logger
            writer.add_scalars(
                main_tag="cross_image_text",
                tag_scalar_dict={k.split("_")[0]: log.metrics[k].avg if mode == "iters" else log.metrics[k].global_avg
                                 for k in cross_image_text_keys},
                global_step=steps
            )
            writer.add_scalars(
                main_tag="cross_text_image",
                tag_scalar_dict={k.split("_")[0]: log.metrics[k].avg if mode == "iters" else log.metrics[k].global_avg
                                 for k in cross_text_image_keys},
                global_step=steps
            )
      

            if "total_loss" in log_keys:
                writer.add_scalar(
                    tag="loss",
                    scalar_value=log.metrics["total_loss"].avg
                    if mode == "iters" else log.metrics["total_loss"].global_avg,
                    global_step=steps
                )
        else:
            pass
        return

    def tb_add_git_version(self, git_version: str):
        if (self.only_main and is_main_process()) or (self.only_main is False):
            git_version = "null" if git_version is None else git_version
            self.tb_iters_logger.add_text(tag="git_version", text_string=git_version)
            self.tb_epochs_logger.add_text(tag="git_version", text_string=git_version)
        else:
            pass
        return


def parser_to_dict(log: argparse.ArgumentParser) -> dict:
    opts_dict = dict()
    for k, v in vars(log).items():
        if v:
            opts_dict[k] = v
    return opts_dict