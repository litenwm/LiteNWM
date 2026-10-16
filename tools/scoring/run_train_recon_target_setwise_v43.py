#!/usr/bin/env python3
"""Run v42 protocol with a true setwise self-attention candidate trunk."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("train_recon_target_full_listwise_v42.py")
    source = target.read_text(encoding="utf-8")
    start_marker = "@dataclass(frozen=True)\nclass Config:"
    end_marker = "\ndef train_model("
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    replacement = '''@dataclass(frozen=True)
class Config:
    input_dim: int
    hidden_dim: int
    dropout: float


class Ranker(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=4,
            dim_feedforward=config.hidden_dim * 3,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 4),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.input(value))
        context = encoded.mean(dim=1, keepdim=True).expand_as(encoded)
        return self.head(torch.cat((encoded, context), dim=-1))

'''
    source = source[:start] + replacement + source[end + 1:]
    source = source.replace(
        "recon_target_full_listwise_v42_member",
        "recon_target_setwise_v43_member",
    ).replace(
        "recon_target_full_listwise_v42",
        "recon_target_setwise_v43",
    ).replace(
        "eleven_bank_full_feature_listwise_lobo_development_screen_complete",
        "eleven_bank_true_setwise_lobo_development_screen_complete",
    ).replace(
        "candidate static + raw LTWM + candidate-incumbent raw delta + ",
        "true 16-candidate self-attention over candidate static + raw LTWM + candidate-incumbent raw delta + ",
    )
    exec(
        compile(source, str(target), "exec"),
        {"__name__": "__main__", "__file__": str(target)},
    )


if __name__ == "__main__":
    main()
