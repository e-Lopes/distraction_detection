"""Fail-fast CUDA/ONNX Runtime diagnostic used before GPU benchmarks."""
from __future__ import annotations

import json
import subprocess
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def main():
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise SystemExit(f"FALHA: CUDAExecutionProvider ausente. Providers: {providers}")

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    c = helper.make_tensor("c", TensorProto.FLOAT, [1, 4], [1.0, 1.0, 1.0, 1.0])
    model = helper.make_model(
        helper.make_graph([helper.make_node("Add", ["x", "c"], ["y"])], "cuda_probe", [x], [y], [c]),
        opset_imports=[helper.make_opsetid("", 13)], ir_version=8,
    )
    with tempfile.NamedTemporaryFile(suffix=".onnx") as file:
        onnx.save(model, file.name)
        session = ort.InferenceSession(file.name, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        actual = session.get_providers()
        output = session.run(None, {"x": np.zeros((1, 4), dtype=np.float32)})[0]
    if actual[0] != "CUDAExecutionProvider" or not np.allclose(output, 1.0):
        raise SystemExit(f"FALHA: sessao nao executou prioritariamente em CUDA: {actual}")
    print(json.dumps({"status": "ok", "nvidia_smi": smi,
                      "onnxruntime": ort.__version__, "providers": actual}, indent=2))


if __name__ == "__main__":
    main()
