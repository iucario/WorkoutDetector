import onnx
import onnxruntime as rt
import numpy as np
import torch

ckpt = '/home/umi/projects/WorkoutDetector/checkpoints/tsm_r50_1x1x16_50e_sthv2_20220521.onnx'

input_shape = [1,16,3,256,256]
input_tensor = torch.randn(input_shape)
onnx_model = onnx.load(ckpt)
onnx.checker.check_model(onnx_model)

input_all = [node.name for node in onnx_model.graph.input]
input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
net_feed_input = list(set(input_all) - set(input_initializer))
assert len(net_feed_input) == 1
sess = rt.InferenceSession(ckpt)
onnx_result = sess.run(
        None, {net_feed_input[0]: input_tensor.detach().numpy()})[0]

print(onnx_result)
