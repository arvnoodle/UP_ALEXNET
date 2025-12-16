"""
Evaluate Conformal Prediction Coverage
- Overall coverage
- Per-class coverage
- Wilson 95% confidence intervals
- Average conformal set size per class
"""

import argparse
import math
import torch

from src.model import AlexNet
from src.conformal import ConformalPredictor
from src.data_loader import get_data_loaders
from config_utils import load_config, setup_device


def wilson_ci(k, n, z=1.96):
    """
    Wilson score interval for binomial proportion.
    Returns (low, high).
    """
    if n == 0:
        return float("nan"), float("nan")

    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) / n) + (z**2) / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


@torch.no_grad()
def evaluate_conformal(model, test_loader, cp, device):
    model.eval()

    # infer number of classes
    x0, _ = next(iter(test_loader))
    x0 = x0.to(device)
    K = model(x0).shape[1]

    class_counts = torch.zeros(K, dtype=torch.long, device=device)
    class_covered = torch.zeros(K, dtype=torch.long, device=device)
    class_set_size_sum = torch.zeros(K, dtype=torch.double, device=device)


    total = 0
    covered_total = 0
    set_size_total = 0.0

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        pred_sets, _ = cp.predict(model, x)      # [B, K]
        in_set = pred_sets.gather(1, y.view(-1, 1)).squeeze(1)
        set_sizes = pred_sets.sum(dim=1)

        B = y.size(0)

        total += B
        covered_total += in_set.sum().item()
        set_size_total += set_sizes.double().sum().item()

        class_counts.scatter_add_(0, y, torch.ones_like(y))
        class_covered.scatter_add_(0, y, in_set.long())
        class_set_size_sum.scatter_add_(0, y, set_sizes.double())

    overall_coverage = covered_total / total
    overall_avg_size = set_size_total / total

    return {
        "total": total,
        "overall_coverage": overall_coverage,
        "overall_avg_size": overall_avg_size,
        "class_counts": class_counts.cpu(),
        "class_covered": class_covered.cpu(),
        "class_avg_set_size": (class_set_size_sum / torch.clamp(class_counts, min=1)).cpu(),
    }



def main():
    parser = argparse.ArgumentParser(description="Evaluate conformal coverage")
    parser.add_argument("--config", default="config_utils/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--conformal-path", default="alexnet_data_out/models/conformal.pt")

    args = parser.parse_args()

    # Load config and device
    config = load_config(args.config)
    device = setup_device(config)

    # Load test loader
    _, _, test_loader = get_data_loaders(config)
    class_names = test_loader.dataset.classes

    # Load model
    model = AlexNet(num_classes=config["model"]["num_classes"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load conformal predictor
    try:
        conf = torch.load(args.conformal_path, map_location="cpu", weights_only=True)
    except TypeError:
        conf = torch.load(args.conformal_path, map_location="cpu")

    cp = ConformalPredictor(alpha=conf["alpha"], device=device)
    cp.qhat = conf["qhat"]

    # Evaluate
    stats = evaluate_conformal(model, test_loader, cp, device)

    # Overall CI
    overall_lo, overall_hi = wilson_ci(
        int(stats["class_covered"].sum()),
        int(stats["class_counts"].sum())
    )

    print("\n" + "=" * 80)
    print("CONFORMAL COVERAGE EVALUATION")
    print("=" * 80)
    print(f"alpha (target miscoverage): {cp.alpha}")
    print(f"target coverage:           {1 - cp.alpha:.3f}")
    print(f"qhat:                      {cp.qhat:.6f}")
    print(f"num test samples:          {stats['total']}")
    print(f"overall coverage:          {stats['overall_coverage']:.3f}")
    print(f"overall coverage CI95:     [{overall_lo:.3f}, {overall_hi:.3f}]")
    print(f"overall avg set size:      {stats['overall_avg_size']:.3f}")
    print("-" * 80)
    print(f"{'Class':<18} {'N':>6} {'Coverage':>10} {'CI95 Low':>10} {'CI95 High':>10} {'AvgSetSize':>12}")
    print("-" * 80)

    for k, name in enumerate(class_names):
        n = int(stats["class_counts"][k].item())
        covered = int(stats["class_covered"][k].item())

        if n > 0:
            cov = covered / n
            lo, hi = wilson_ci(covered, n)
            avg_size = float(stats["class_avg_set_size"][k].item())
            print(f"{name:<18} {n:>6} {cov:>10.3f} {lo:>10.3f} {hi:>10.3f} {avg_size:>12.3f}")
        else:
            print(f"{name:<18} {n:>6} {'nan':>10} {'nan':>10} {'nan':>10} {'nan':>12}")

    print("=" * 80)


if __name__ == "__main__":
    main()
