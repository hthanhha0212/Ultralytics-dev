# This code is adapted from Issue [#147](https://github.com/VainF/Torch-Pruning/issues/147), implemented by @Hyunseok-Kim0.
import argparse
import json
import math
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Union, Optional

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from ultralytics import YOLO, __version__
from ultralytics.nn.modules import QC2f, Conv, QBottleneck, QCIB, QC2fCIB, QAttention, QPSA, Qv10Detect
from ultralytics.nn.tasks import torch_safe_load, load_checkpoint
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import YAML, LOGGER, RANK, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.torch_utils import initialize_weights, unwrap_model

import torch_pruning as tp



def clear_ptflops_cache(module):
    for m in module.modules():
        for attr in ("__flops__", "__params__", "__ptflops_backup_flops__", "__ptflops_backup_params__"):
            if hasattr(m, attr):
                delattr(m, attr)


def load_resume_state(path: Optional[Union[str, Path]]) -> Optional[dict]:
    if not path:
        return None
    resume_path = Path(path)
    if not resume_path.exists():
        return None
    with resume_path.open("r") as f:
        return json.load(f)


def save_resume_state(path: Optional[Union[str, Path]], state: dict) -> None:
    if not path:
        return
    resume_path = Path(path)
    resume_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_state = {**state}
    for key in ("macs_list", "nparams_list", "map_list", "pruned_map_list"):
        if key in serializable_state and serializable_state[key] is not None:
            serializable_state[key] = [float(x) for x in serializable_state[key]]
    for key in ("base_macs", "base_nparams", "init_map"):
        if key in serializable_state and serializable_state[key] is not None:
            serializable_state[key] = float(serializable_state[key])
    if serializable_state.get("last_model_path"):
        serializable_state["last_model_path"] = str(serializable_state["last_model_path"])

    with resume_path.open("w") as f:
        json.dump(serializable_state, f, indent=2)

def save_pruning_performance_graph(x, y1, y2, y3):
    """
    Draw performance change graph
    Parameters
    ----------
    x : List
        Parameter numbers of all pruning steps
    y1 : List
        mAPs after fine-tuning of all pruning steps
    y2 : List
        MACs of all pruning steps
    y3 : List
        mAPs after pruning (not fine-tuned) of all pruning steps

    Returns
    -------

    """
    try:
        plt.style.use("ggplot")
    except:
        pass

    x, y1, y2, y3 = np.array(x), np.array(y1), np.array(y2), np.array(y3)
    y2_ratio = y2 / y2[0]

    # create the figure and the axis object
    fig, ax = plt.subplots(figsize=(8, 6))

    # plot the pruned mAP and recovered mAP
    ax.set_xlabel('Pruning Ratio')
    ax.set_ylabel('mAP')
    ax.plot(x, y1, label='recovered mAP')
    ax.scatter(x, y1)
    ax.plot(x, y3, color='tab:gray', label='pruned mAP')
    ax.scatter(x, y3, color='tab:gray')

    # create a second axis that shares the same x-axis
    ax2 = ax.twinx()

    # plot the second set of data
    ax2.set_ylabel('MACs')
    ax2.plot(x, y2_ratio, color='tab:orange', label='MACs')
    ax2.scatter(x, y2_ratio, color='tab:orange')

    # add a legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    ax.set_xlim(105, -5)
    ax.set_ylim(0, max(y1) + 0.05)
    ax2.set_ylim(0.05, 1.05)

    # calculate the highest and lowest points for each set of data
    max_y1_idx = np.argmax(y1)
    min_y1_idx = np.argmin(y1)
    max_y2_idx = np.argmax(y2)
    min_y2_idx = np.argmin(y2)
    max_y1 = y1[max_y1_idx]
    min_y1 = y1[min_y1_idx]
    max_y2 = y2_ratio[max_y2_idx]
    min_y2 = y2_ratio[min_y2_idx]

    # add text for the highest and lowest values near the points
    ax.text(x[max_y1_idx], max_y1 - 0.05, f'max mAP = {max_y1:.2f}', fontsize=10)
    ax.text(x[min_y1_idx], min_y1 + 0.02, f'min mAP = {min_y1:.2f}', fontsize=10)
    ax2.text(x[max_y2_idx], max_y2 - 0.05, f'max MACs = {max_y2 * y2[0] / 1e9:.2f}G', fontsize=10)
    ax2.text(x[min_y2_idx], min_y2 + 0.02, f'min MACs = {min_y2 * y2[0] / 1e9:.2f}G', fontsize=10)

    plt.title('Comparison of mAP and MACs with Pruning Ratio')
    plt.savefig('pruning_perf_change.png')


def infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add

def infer_shortcut_cib(cib):
    c1 = cib.cv1[0].conv.in_channels
    c2 = cib.cv1[4].conv.in_channels
    return c1 == c2 and hasattr(cib, 'add') and cib.add 

class C2f_v2(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(QBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # y = list(self.cv1(x).chunk(2, 1))
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C2fCIB_v2(C2f_v2):
    """
    Quantized C2fCIB block used Quantized C2f and Quantized CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """
        Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use local key connection.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(QCIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))

def replace_c2fcib_with_c2fcib_v2(module):
    for name, child_module in module.named_children():
        if isinstance(child_module, QC2fCIB):
            # Replace C2f with C2f_v2 while preserving its parameters
            shortcut = infer_shortcut_cib(child_module.m[0])
            c2fcib_v2 = C2fCIB_v2(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=shortcut,
                            e=child_module.c / child_module.cv2.conv.out_channels)
            transfer_weights(child_module, c2fcib_v2)
            setattr(module, name, c2fcib_v2)
        else:
            replace_c2fcib_with_c2fcib_v2(child_module)

def transfer_weights(c2f, c2f_v2):
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    # Transfer cv1 weights from C2f to cv0 and cv1 in C2f_v2
    old_weight = state_dict['cv1.conv.weight']
    half_channels = old_weight.shape[0] // 2
    state_dict_v2['cv0.conv.weight'] = old_weight[:half_channels]
    state_dict_v2['cv1.conv.weight'] = old_weight[half_channels:]

    # Transfer cv1 batchnorm weights and buffers from C2f to cv0 and cv1 in C2f_v2
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'cv1.bn.{bn_key}']
        state_dict_v2[f'cv0.bn.{bn_key}'] = old_bn[:half_channels]
        state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn[half_channels:]

    # Transfer remaining weights and buffers
    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict_v2[key] = state_dict[key]

    # Transfer all non-method attributes
    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)


def replace_c2f_with_c2f_v2(module):
    for name, child_module in module.named_children():
        if isinstance(child_module, QC2f):
            if isinstance(child_module, QC2fCIB):
                continue
            # Replace C2f with C2f_v2 while preserving its parameters
            shortcut = infer_shortcut(child_module.m[0])
            c2f_v2 = C2f_v2(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=shortcut,
                            g=child_module.m[0].cv2.conv.groups,
                            e=child_module.c / child_module.cv2.conv.out_channels)
            transfer_weights(child_module, c2f_v2)
            setattr(module, name, c2f_v2)
        else:
            replace_c2f_with_c2f_v2(child_module)


def save_model_v2(self: BaseTrainer):
    """
    Disabled half precision saving. originated from ultralytics/yolo/engine/trainer.py
    """
    ckpt = {
        'epoch': self.epoch,
        'best_fitness': self.best_fitness,
        'model': deepcopy(unwrap_model(self.model)),
        'ema': deepcopy(self.ema.ema),
        'updates': self.ema.updates,
        'optimizer': self.optimizer.state_dict(),
        'train_args': vars(self.args),  # save as dict
        'date': datetime.now().isoformat(),
        'version': __version__}

    # Save last, best and delete
    torch.save(ckpt, self.last)
    if self.best_fitness == self.fitness:
        torch.save(ckpt, self.best)
    if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
        torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')
    del ckpt


def final_eval_v2(self: BaseTrainer):
    """
    originated from ultralytics/yolo/engine/trainer.py
    """
    for f in self.last, self.best:
        if f.exists():
            strip_optimizer_v2(f)  # strip optimizers
            if f is self.best:
                LOGGER.info(f'\nValidating {f}...')
                self.metrics = self.validator(model=f)
                self.metrics.pop('fitness', None)
                self.run_callbacks('on_fit_epoch_end')


def strip_optimizer_v2(f: Union[str, Path] = 'best.pt', s: str = '') -> None:
    """
    Disabled half precision saving. originated from ultralytics/yolo/utils/torch_utils.py
    """
    try:
        x = torch.load(f, map_location=torch.device('cpu'), weights_only=False)
    except TypeError:  # PyTorch < 2.6 does not support weights_only kwarg
        x = torch.load(f, map_location=torch.device('cpu'))
    args = {**DEFAULT_CFG_DICT, **x['train_args']}  # combine model args with default args, preferring model args
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'ema', 'updates':  # keys
        x[k] = None
    for p in x['model'].parameters():
        p.requires_grad = False
    x['train_args'] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # strip non-default keys
    # x['model'].args = x['train_args']
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")


def train_v2(self: YOLO, pruning=False, **kwargs):
    """
    Disabled loading new model when pruning flag is set. originated from ultralytics/yolo/engine/model.py
    """

    self._check_is_pytorch_model()
    if self.session:  # Ultralytics HUB session
        if any(kwargs):
            LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
        kwargs = self.session.train_args
    base_overrides = self.overrides.copy()
    cfg_overrides = {}
    if kwargs.get('cfg'):
        LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
        cfg_overrides = YAML.load(check_yaml(kwargs['cfg']))
    overrides = {**base_overrides, **cfg_overrides, **kwargs}
    overrides.pop('cfg', None)
    model_override = overrides.get('model') or base_overrides.get('model') or getattr(self, 'ckpt_path', None)
    if model_override:
        overrides['model'] = str(model_override)
    overrides['mode'] = 'train'
    if not overrides.get('data'):
        raise AttributeError("Dataset required but missing, i.e. pass 'data=coco8.yaml'")
    if overrides.get('resume'):
        overrides['resume'] = self.ckpt_path

    self.task = overrides.get('task') or self.task
    self.trainer = self._smart_load("trainer")(overrides=overrides, _callbacks=self.callbacks)

    if not pruning:
        if not overrides.get('resume'):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model

    else:
        # pruning mode
        self.trainer.pruning = True
        self.trainer.model = self.model

        # replace some functions to disable half precision saving
        self.trainer.save_model = save_model_v2.__get__(self.trainer)
        self.trainer.final_eval = final_eval_v2.__get__(self.trainer)

    self.trainer.hub_session = self.session  # attach optional HUB session
    self.trainer.train()
    # Update model and cfg after training
    if RANK in (-1, 0):
        ckpt_path = self.trainer.best if self.trainer.best.exists() else self.trainer.last
        self.model, self.ckpt = load_checkpoint(str(ckpt_path))
        self.overrides = self.model.args
        self.metrics = getattr(self.trainer.validator, 'metrics', None)


def prune(args):
    resume_state_path = Path(args.resume_state).expanduser() if getattr(args, "resume_state", None) else None
    resume_state = load_resume_state(resume_state_path)
    start_step = 0
    base_macs_value = None
    base_nparams_value = None
    init_map_value = None

    if resume_state:
        if resume_state.get("target_prune_rate") != args.target_prune_rate or \
                resume_state.get("iterative_steps") != args.iterative_steps:
            raise ValueError("Resume state does not match the requested pruning configuration. "
                             "Please adjust the resume file or start a fresh run.")
        start_step = int(resume_state.get("current_step", 0))
        base_macs_value = resume_state.get("base_macs")
        base_nparams_value = resume_state.get("base_nparams")
        init_map_value = resume_state.get("init_map")
        if base_macs_value is None or base_nparams_value is None or init_map_value is None:
            raise ValueError("Resume state is missing baseline statistics needed to continue pruning.")
        macs_list = [float(x) for x in resume_state.get("macs_list", [])]
        nparams_list = [float(x) for x in resume_state.get("nparams_list", [])]
        map_list = [float(x) for x in resume_state.get("map_list", [])]
        pruned_map_list = [float(x) for x in resume_state.get("pruned_map_list", [])]
        resume_weights_path = str(resume_state.get("last_model_path", args.model))
        print(f"Resuming pruning from step {start_step} using weights '{resume_weights_path}'.")
    else:
        resume_state = {
            "target_prune_rate": args.target_prune_rate,
            "iterative_steps": args.iterative_steps,
            "current_step": 0,
            "macs_list": [],
            "nparams_list": [],
            "map_list": [],
            "pruned_map_list": [],
            "last_model_path": str(Path(args.model).resolve())
        }
        macs_list, nparams_list, map_list, pruned_map_list = [], [], [], []
        resume_weights_path = str(args.model)

    model = YOLO(resume_weights_path)
    model.__setattr__("train_v2", train_v2.__get__(model))
    pruning_cfg = YAML.load(check_yaml(args.cfg))
    batch_size = pruning_cfg['batch']

    # use coco128 dataset for 10 epochs fine-tuning each pruning iteration step
    # this part is only for sample code, number of epochs should be included in config file
    pruning_cfg['data'] = "data_config.yaml"
    pruning_cfg['epochs'] = 10

    model.model.train()
    replace_c2f_with_c2f_v2(model.model)
    replace_c2fcib_with_c2fcib_v2(model.model)
    initialize_weights(model.model)  # set BN.eps, momentum, ReLU.inplace

    for name, param in model.model.named_parameters():
        param.requires_grad = True

    example_inputs = torch.randn(1, 3, pruning_cfg["imgsz"], pruning_cfg["imgsz"]).to(model.device)
    clear_ptflops_cache(model.model)

    if start_step == 0 and not macs_list:
        base_macs_value, base_nparams_value = tp.utils.count_ops_and_params(model.model, example_inputs)
        pruning_cfg['name'] = "baseline_val"
        pruning_cfg['batch'] = 1
        validation_model = deepcopy(model)
        metric = validation_model.val(**pruning_cfg)
        init_map_value = float(metric.box.map)
        base_macs_value = float(base_macs_value)
        base_nparams_value = float(base_nparams_value)
        macs_list.append(base_macs_value)
        nparams_list.append(100)
        map_list.append(init_map_value)
        pruned_map_list.append(init_map_value)
        resume_state.update({
            "base_macs": float(base_macs_value),
            "base_nparams": float(base_nparams_value),
            "init_map": float(init_map_value),
            "macs_list": macs_list,
            "nparams_list": nparams_list,
            "map_list": map_list,
            "pruned_map_list": pruned_map_list,
            "last_model_path": str(Path(resume_weights_path).resolve()),
            "current_step": 0
        })
        if resume_state_path:
            save_resume_state(resume_state_path, resume_state)
    else:
        validation_model = deepcopy(model)
        base_macs_value = float(base_macs_value)
        base_nparams_value = float(base_nparams_value)
        init_map_value = float(init_map_value)

    print(f"Before Pruning: MACs={base_macs_value / 1e9: .5f} G, #Params={base_nparams_value / 1e6: .5f} M, "
          f"mAP={init_map_value: .5f}")

    # prune same ratio of filter based on initial size
    pruning_ratio = 1 - math.pow((1 - args.target_prune_rate), 1 / args.iterative_steps)

    for i in range(start_step, args.iterative_steps):

        model.model.train()
        for name, param in model.model.named_parameters():
            param.requires_grad = True

        ignored_layers = []
        unwrapped_parameters = []
        for m in model.model.modules():
            if isinstance(m, (QAttention, Qv10Detect)):
                ignored_layers.append(m)

        example_inputs = example_inputs.to(model.device)
        pruner = tp.pruner.GroupNormPruner(
            model.model,
            example_inputs,
            importance=tp.importance.GroupMagnitudeImportance(),  # L2 norm pruning,
            iterative_steps=1,
            pruning_ratio=pruning_ratio,
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters
        )

        tp.utils.print_tool.before_pruning(model.model)
        pruner.step()
        tp.utils.print_tool.after_pruning(model.model, do_print=True)

        # pre fine-tuning validation
        pruning_cfg['name'] = f"step_{i}_pre_val"
        pruning_cfg['batch'] = 1
        validation_model.model = deepcopy(model.model)
        metric = validation_model.val(**pruning_cfg)
        pruned_map = float(metric.box.map)
        clear_ptflops_cache(model.model)
        pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(pruner.model, example_inputs.to(model.device))
        pruned_macs = float(pruned_macs)
        pruned_nparams = float(pruned_nparams)
        base_reference_macs = float(macs_list[0]) if macs_list else float(base_macs_value)
        current_speed_up = base_reference_macs / pruned_macs
        print(f"After pruning iter {i + 1}: MACs={pruned_macs / 1e9} G, #Params={pruned_nparams / 1e6} M, "
              f"mAP={pruned_map}, speed up={current_speed_up}")

        # fine-tuning
        for name, param in model.model.named_parameters():
            param.requires_grad = True
        pruning_cfg['name'] = f"step_{i}_finetune"
        pruning_cfg['batch'] = batch_size  # restore batch size
        model.train_v2(pruning=True, **pruning_cfg)

        # post fine-tuning validation
        pruning_cfg['name'] = f"step_{i}_post_val"
        pruning_cfg['batch'] = 1
        validation_model = YOLO(model.trainer.best)
        metric = validation_model.val(**pruning_cfg)
        current_map = float(metric.box.map)
        print(f"After fine tuning mAP={current_map}")

        macs_list.append(pruned_macs)
        nparams_list.append(pruned_nparams / base_nparams_value * 100)
        pruned_map_list.append(pruned_map)
        map_list.append(current_map)

        best_ckpt_path = getattr(model.trainer, "best", None)
        if best_ckpt_path is None or not best_ckpt_path.exists():
            best_ckpt_path = getattr(model.trainer, "last", None)
        if best_ckpt_path is None:
            raise FileNotFoundError("No checkpoint produced during fine-tuning. Cannot resume.")

        resume_state.update({
            "current_step": i + 1,
            "macs_list": macs_list,
            "nparams_list": nparams_list,
            "map_list": map_list,
            "pruned_map_list": pruned_map_list,
            "last_model_path": str(best_ckpt_path.resolve()),
            "base_macs": float(base_macs_value),
            "base_nparams": float(base_nparams_value),
            "init_map": float(init_map_value)
        })
        if resume_state_path:
            save_resume_state(resume_state_path, resume_state)

        # remove pruner after single iteration
        del pruner

        save_pruning_performance_graph(nparams_list, map_list, macs_list, pruned_map_list)

        if init_map_value - current_map > args.max_map_drop:
            print("Pruning early stop")
            resume_state["early_stop"] = True
            if resume_state_path:
                save_resume_state(resume_state_path, resume_state)
            break

    if resume_state.get("current_step", 0) >= args.iterative_steps:
        resume_state["completed"] = True
    if resume_state_path:
        save_resume_state(resume_state_path, resume_state)

    #model.export(format='onnx')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ultralytics/pretrained/weights/best.pt', help='Pretrained pruning target model file')
    parser.add_argument('--cfg', default='default.yaml',
                        help='Pruning config file.'
                             ' This file should have same format with ultralytics/yolo/cfg/default.yaml')
    parser.add_argument('--iterative-steps', default=16, type=int, help='Total pruning iteration step')
    parser.add_argument('--target-prune-rate', default=0.6, type=float, help='Target pruning rate')
    parser.add_argument('--max-map-drop', default=0.2, type=float, help='Allowed maximum map drop after fine-tuning')
    parser.add_argument('--resume-state', default=None, type=str,
                        help='Path to a JSON file used to store and resume iterative pruning progress.')

    args = parser.parse_args()

    prune(args)
