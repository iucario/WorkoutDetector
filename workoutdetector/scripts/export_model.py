from typing import Optional
from workoutdetector.trainer import LitModel
import torch
import onnx
import onnxruntime as ort
import mmcv
from mmcv.runner import load_checkpoint
from mmaction.models import build_model
from workoutdetector.settings import PROJ_ROOT


def _convert_batchnorm(module: torch.nn.Module) -> torch.nn.Module:
    """Convert the syncBNs into normal BN3ds."""

    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm3d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def export_lit_model(ckpt: str, onnx_path: Optional[str] = None) -> None:
    """Export a LitImageModel to ONNX format."""

    model = LitModel.load_from_checkpoint(ckpt)
    model.eval()
    model.cuda()
    if onnx_path is None:
        onnx_path = ckpt.replace('.ckpt', '.onnx')
    model.to_onnx(onnx_path,
                  input_sample=torch.randn(1, 8, 3, 224, 224),
                  export_params=True,
                  opset_version=11)
    print(f'Model exported to {onnx_path}')


def export_mmlab_model(ckpt: str, output: str, cfg_path: str) -> None:
    input_sample = torch.randn(1, 3, 8, 224, 224)
    cfg = mmcv.Config.fromfile(cfg_path)
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    model.cuda().eval()
    torch.onnx.export(model,
                      input_sample,
                      output,
                      export_params=True,
                      keep_initializers_as_inputs=True,
                      opset_version=11)
    onnx = onnx.load(output)
    assert onnx(input_sample.cuda()) == model(input_sample.cuda()), \
        'ONNX output does not match with the torch model output'
    print(f'Model exported to {output}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=False)
    parser.add_argument('--cfg', type=str, required=False)

    # args_mmlab = [
    #     f'--ckpt={PROJ_ROOT}/work_dirs/tsm_MultiActionRepCount_sthv2_20220625-224626/best_top1_acc_epoch_5.pth',
    #     '-o=tsm-pull_up-20220625.onnx',
    #     f'--cfg={PROJ_ROOT}/workoutdetector/configs/tsm_MultiActionRepCount_sthv2.py'
    # ]
    args_lit = [
        '--ckpt',
        "exp/repcount-12-tsm/checkpoints/best-val-acc=0.86-epoch=07-20220705-220720.ckpt"
    ]
    args = parser.parse_args(args_lit)
    export_lit_model(args.ckpt, args.output)
    # export_mmlab_model(args.ckpt, args.output, args.cfg)
