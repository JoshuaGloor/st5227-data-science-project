from pathlib import Path
import matplotlib.pyplot as plt


def savefig(
    name: str,
    reports_dir: Path | None = None,
    dpi: int = 300,
) -> None:
    if reports_dir is None:
        # Anchor to project root via this file's location: src/plot_helpers.py
        reports_dir = Path(__file__).resolve().parents[1] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(reports_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
