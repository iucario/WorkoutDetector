from typing import Optional
from workoutdetector.trainer import LitImageModel
import torch


def export_lit_model(ckpt: str, onnx_path: Optional[str] = None) -> None:
    model = LitImageModel.load_from_checkpoint(ckpt)
    model.eval()
    if onnx_path is None:
        onnx_path = ckpt.replace('.ckpt', '.onnx')
    model.to_onnx(onnx_path,
                  input_sample=torch.randn(1, 3, 224, 224),
                  export_params=True,
                  opset_version=11)
    print(f'Model exported to {onnx_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('-o', '--onnx_path', type=str, required=False)
    args = parser.parse_args()
    export_lit_model(args.ckpt, args.onnx_path)