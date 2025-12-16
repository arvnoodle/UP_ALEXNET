import torch
import math


class ConformalPredictor:
    """
    Split Conformal Predictor for multiclass classification.
    """

    def __init__(self, alpha=0.1, device="cpu"):
        self.alpha = alpha
        self.device = device
        self.qhat = None

    @staticmethod
    def wilson_ci(k, n, z=1.96):
        """
        Wilson score interval for a binomial proportion.
        Returns (low, high). If n == 0, returns (nan, nan).
        """
        if n == 0:
            return float("nan"), float("nan")

        p = k / n
        denom = 1.0 + (z**2) / n
        center = (p + (z**2) / (2.0 * n)) / denom
        half = (z * math.sqrt((p * (1.0 - p) / n) + (z**2) / (4.0 * n * n))) / denom
        return max(0.0, center - half), min(1.0, center + half)

    def calibrate(self, model, calib_loader):
        """
        Compute and store conformal threshold q̂ using calibration data.
        """
        model.eval()
        scores = []

        with torch.no_grad():
            for x, y in calib_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = model(x)
                probs = torch.softmax(logits, dim=1)

                true_probs = probs.gather(1, y.view(-1, 1)).squeeze(1)
                nonconformity = 1.0 - true_probs
                scores.append(nonconformity.cpu())

        scores = torch.cat(scores)
        n = scores.numel()
        k = math.ceil((n + 1) * (1 - self.alpha))
        self.qhat = torch.kthvalue(scores, k).values.item()

        return self.qhat

    def predict(self, model, x):
        """
        Generate conformal prediction sets using stored q̂.
        """
        if self.qhat is None:
            raise RuntimeError("ConformalPredictor must be calibrated before prediction.")

        model.eval()
        x = x.to(self.device)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
            threshold = 1.0 - self.qhat
            pred_sets = probs >= threshold

        return pred_sets, probs