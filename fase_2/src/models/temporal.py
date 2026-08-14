"""Modelos temporais com interface comum para classificação multiclasse."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        hidden_dim: int = 64,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.2,
        pooling: str = "last",
    ) -> None:
        super().__init__()
        if pooling not in {"last", "mean"}:
            raise ValueError(f"Pooling LSTM não suportado: {pooling}")
        self.pooling = pooling
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(output_dim, num_classes))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence, hidden = self.lstm(values)
        if self.pooling == "mean":
            representation = sequence.mean(dim=1)
        elif self.lstm.bidirectional:
            representation = torch.cat((hidden[0][-2], hidden[0][-1]), dim=1)
        else:
            representation = hidden[0][-1]
        return self.head(representation)


class Chomp1d(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values[:, :, : -self.size].contiguous() if self.size else values


class TemporalResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.network = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                output_channels,
                output_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Conv1d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(self.network(values) + self.residual(values))


class TCNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        channels: Sequence[int] = (32, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.2,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("TCN exige ao menos um bloco")
        if pooling not in {"last", "mean"}:
            raise ValueError(f"Pooling TCN não suportado: {pooling}")
        self.pooling = pooling
        blocks: list[nn.Module] = []
        previous = input_dim
        for index, channel in enumerate(channels):
            blocks.append(
                TemporalResidualBlock(
                    previous,
                    int(channel),
                    kernel_size=kernel_size,
                    dilation=2**index,
                    dropout=dropout,
                )
            )
            previous = int(channel)
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(previous, num_classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence = self.network(values.transpose(1, 2))
        representation = sequence.mean(dim=2) if self.pooling == "mean" else sequence[:, :, -1]
        return self.head(representation)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dimension: int, max_length: int) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / dimension)
        )
        encoding = torch.zeros(max_length, dimension)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] > self.encoding.shape[1]:
            raise ValueError("Sequencia excede max_length do positional encoding")
        return values + self.encoding[:, : values.shape[1]]


class TransformerClassifier(nn.Module):
    """Transformer pequeno para a comparacao controlada do plano integrado."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        model_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 64,
        dropout: float = 0.2,
        pooling: str = "mean",
        max_length: int = 150,
    ) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim deve ser divisivel por num_heads")
        if pooling not in {"mean", "last"}:
            raise ValueError(f"Pooling Transformer nao suportado: {pooling}")
        self.pooling = pooling
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.position = SinusoidalPositionalEncoding(model_dim, max_length)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, num_classes))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence = self.encoder(self.position(self.input_projection(values)))
        representation = sequence.mean(dim=1) if self.pooling == "mean" else sequence[:, -1]
        return self.head(representation)


def build_temporal_model(
    name: str,
    *,
    input_dim: int,
    num_classes: int,
    parameters: Mapping[str, object],
) -> nn.Module:
    if name == "lstm":
        return LSTMClassifier(input_dim, num_classes, **parameters)
    if name == "tcn":
        return TCNClassifier(input_dim, num_classes, **parameters)
    if name == "transformer":
        return TransformerClassifier(input_dim, num_classes, **parameters)
    raise ValueError(f"Modelo temporal desconhecido: {name}")


def trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
