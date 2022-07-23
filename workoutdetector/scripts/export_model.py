from typing import Optional

import onnx
import onnxruntime as ort
import torch
from workoutdetector.trainer import LitModel


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


def export_lit_model(ckpt: str,
                     mode: str = 'onnx',
                     out_path: Optional[str] = None) -> None:
    """Export a LitImageModel to ONNX or torch script.
    
    Args:   
        ckpt (str): path to the checkpoint file
        mode (str): 'onnx' or 'torchscript'
        out_path (str): path to the output file
    """

    model = LitModel.load_from_checkpoint(ckpt)
    model.eval()
    model.cuda()
    ext = 'onnx' if mode == 'onnx' else 'pt'
    if out_path is None:
        out_path = ckpt.replace('.ckpt', f'.{ext}')
    if mode == 'onnx':
        model.to_onnx(out_path,
                      input_sample=torch.randn(1, 8, 3, 224, 224),
                      export_params=True,
                      opset_version=11)
    elif mode == 'torchscript':
        model.to_torchscript(
            out_path,
            example_inputs=torch.randn(1, 8, 3, 224, 224),
        )
    x = torch.randn(1, 8, 3, 224, 224).cuda()
    print(f'Model exported to {out_path}')
    if mode == 'onnx':
        onnx_model = ort.InferenceSession(out_path, providers=['CUDAExecutionProvider'])
        name = onnx_model.get_inputs()[0].name
        output = onnx_model.run(None, {name: x.cpu().numpy()})[0]
        assert output.shape == model(x).shape, f'{output.shape} != {model(x).shape}'
        assert torch.allclose(torch.Tensor(output), model(x).cpu()), \
            'ONNX output does not match with the torch model output'
    else:
        torch_model = torch.jit.load(out_path)
        assert torch.equal(torch_model(x), model(x)), \
            'Torchscript output does not match with the torch model output'


def export_mmlab_model(ckpt: str, output: str, cfg_path: str) -> None:
    import mmcv
    from mmaction.models import build_model
    from mmcv.runner import load_checkpoint
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
    parser.add_argument('--mode', type=str, default='onnx', choices=['onnx', 'torchscript'])

    # args_mmlab = [
    #     f'--ckpt={PROJ_ROOT}/work_dirs/tsm_MultiActionRepCount_sthv2_20220625-224626/best_top1_acc_epoch_5.pth',
    #     '-o=tsm-pull_up-20220625.onnx',
    #     f'--cfg={PROJ_ROOT}/workoutdetector/configs/tsm_MultiActionRepCount_sthv2.py'
    # ]
    args_lit = [
        '--cfg', 'workoutdetector/configs/defaults.yaml', '--ckpt',
        "checkpoints/repcount-12/best-val-acc=0.841-epoch=26-20220711-191616.ckpt",
        '-o', 'rural-river-23-repcount-12-20220711-191616.onnx',
        '--mode', 'onnx'
    ]
    args = parser.parse_args(args_lit)
    export_lit_model(args.ckpt,args.mode, args.output)
    # export_mmlab_model(args.ckpt, args.output, args.cfg)
