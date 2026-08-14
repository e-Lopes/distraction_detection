import pytest
import torch

from fase_2.src.models.temporal import build_temporal_model, trainable_parameters


@pytest.mark.parametrize(
    ("name", "parameters"),
    [
        (
            "lstm",
            {
                "hidden_dim": 16,
                "num_layers": 1,
                "bidirectional": False,
                "dropout": 0.1,
                "pooling": "last",
            },
        ),
        (
            "tcn",
            {"channels": [8, 16], "kernel_size": 3, "dropout": 0.1, "pooling": "mean"},
        ),
        (
            "transformer",
            {
                "model_dim": 16,
                "num_heads": 4,
                "num_layers": 1,
                "feedforward_dim": 32,
                "dropout": 0.1,
                "pooling": "mean",
                "max_length": 150,
            },
        ),
    ],
)
def test_temporal_model_output_shape(name, parameters):
    model = build_temporal_model(name, input_dim=8, num_classes=3, parameters=parameters)
    output = model(torch.randn(4, 60, 8))
    assert output.shape == (4, 3)
    assert trainable_parameters(model) > 0


def test_temporal_model_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="desconhecido"):
        build_temporal_model("unknown", input_dim=8, num_classes=3, parameters={})
