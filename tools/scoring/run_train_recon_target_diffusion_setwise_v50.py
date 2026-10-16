#!/usr/bin/env python3
"""Twenty-bank true-setwise screen with NoMaD diffusion diagnostics."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("train_recon_target_full_listwise_v42.py")
    source = target.read_text(encoding="utf-8")

    parser_old = '    parser.add_argument("--bank-ijk-root", type=Path, required=True)\n'
    parser_new = parser_old + (
        '    parser.add_argument("--bank-lp-root", type=Path, required=True)\n'
        '    parser.add_argument("--bank-qt-root", type=Path, required=True)\n'
    )
    if source.count(parser_old) != 1:
        raise RuntimeError("v50 parser source contract changed")
    source = source.replace(parser_old, parser_new, 1)

    feature_old = '    x = np.concatenate((compact, rollout), axis=2).astype(np.float32)\n'
    feature_new = (
        '    with np.load(source.parent / "diffusion_v48b.npz", '
        'allow_pickle=False) as payload:\n'
        '        diffusion = payload["candidate_diffusion_features"].astype(np.float32)\n'
        '    if diffusion.shape != (100, 16, 90):\n'
        '        raise RuntimeError(f"diffusion feature contract changed for {name}")\n'
        '    x = np.concatenate((compact, rollout, diffusion), axis=2).astype(np.float32)\n'
    )
    if source.count(feature_old) != 1:
        raise RuntimeError("v50 feature source contract changed")
    source = source.replace(feature_old, feature_new, 1)

    return_old = '    return banks\n\n\n@dataclass(frozen=True)\nclass Config:'
    return_new = '''    for letter in "lmnop":
        root = args.bank_lp_root / f"bank_{letter}"
        banks.append(load_bank(
            f"bank_{letter}",
            root / "features" / f"recon_target_bank_{letter}_v44_features.npz",
            root / "universal_scores.npz",
            root / "source_labeled.npz",
            root / "spatial_rollout_v17.npz",
        ))
    for letter in "qrst":
        root = args.bank_qt_root / f"bank_{letter}"
        banks.append(load_bank(
            f"bank_{letter}",
            root / "features" / f"recon_target_bank_{letter}_v46_features.npz",
            root / "universal_scores.npz",
            root / "source_labeled.npz",
            root / "spatial_rollout_v17.npz",
        ))
    return banks


@dataclass(frozen=True)
class Config:'''
    if source.count(return_old) != 1:
        raise RuntimeError("v50 bank loader source contract changed")
    source = source.replace(return_old, return_new, 1)

    model_start = source.index("@dataclass(frozen=True)\nclass Config:")
    model_end = source.index("\ndef train_model(", model_start)
    model = '''@dataclass(frozen=True)
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
            d_model=config.hidden_dim, nhead=4,
            dim_feedforward=config.hidden_dim * 3,
            dropout=config.dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 4),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.input(value))
        context = encoded.mean(dim=1, keepdim=True).expand_as(encoded)
        return self.head(torch.cat((encoded, context), dim=-1))

'''
    source = source[:model_start] + model + source[model_end + 1:]
    source = source.replace(
        "recon_target_full_listwise_v42_member",
        "recon_target_diffusion_setwise_v50_member",
    ).replace(
        "recon_target_full_listwise_v42",
        "recon_target_diffusion_setwise_v50",
    ).replace(
        "eleven_bank_full_feature_listwise_lobo_development_screen_complete",
        "twenty_bank_diffusion_true_setwise_lobo_development_screen_complete",
    ).replace(
        "candidate static + raw LTWM + candidate-incumbent raw delta + ",
        "true setwise candidate static + raw LTWM + diffusion diagnostics + candidate-incumbent raw delta + ",
    )
    exec(
        compile(source, str(target), "exec"),
        {"__name__": "__main__", "__file__": str(target)},
    )


if __name__ == "__main__":
    main()
