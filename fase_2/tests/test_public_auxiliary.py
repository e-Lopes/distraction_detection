import pytest
import torch
from fase_2.src.models.public_auxiliary import AuxiliaryVisualEncoder


@pytest.mark.parametrize("task", ["eye_state", "yawn"])
def test_auxiliary_encoder_shape_and_probability(task):
    model = AuxiliaryVisualEncoder(task=task)
    images = torch.rand(3, 3, 32, 48)
    assert model(images).shape == (3, 2)
    probabilities = model.positive_probability(images)
    assert probabilities.shape == (3,)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))


def test_auxiliary_encoder_rejects_unknown_task():
    with pytest.raises(ValueError):
        AuxiliaryVisualEncoder(task="fatigue")
