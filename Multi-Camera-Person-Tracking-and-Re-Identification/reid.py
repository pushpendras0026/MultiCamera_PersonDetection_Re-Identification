from torchreid.data.transforms import build_transforms
from PIL import Image
import torchreid
import torch
from torchreid import metrics
import cv2
import numpy as np


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
BLUR_THRESHOLD  = 100     # Laplacian variance below this → blurry frame
EMA_ALPHA       = 0.9     # Weight on the running embedding (vs new frame)
COS_THRESH_HQ   = 0.40    # Max cosine distance for both-sides HQ match
COS_THRESH_LQ   = 0.60    # Lenient threshold when one side is blurry


def compute_blur_score(crop_bgr: np.ndarray) -> float:
    """Return Laplacian variance of a BGR crop (higher = sharper)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def preprocess_crop(crop_bgr: np.ndarray, target_h: int = 256, target_w: int = 128) -> np.ndarray:
    """CLAHE lighting normalisation + letterbox resize → RGB uint8 (H, W, 3)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return np.full((target_h, target_w, 3), 128, dtype=np.uint8)

    # 1. CLAHE on L-channel (LAB)
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_eq  = clahe.apply(l_ch)
    bgr_eq = cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # 2. Letterbox resize with gray padding
    h, w   = bgr_eq.shape[:2]
    scale  = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr_eq, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    canvas[(target_h - new_h) // 2:(target_h - new_h) // 2 + new_h,
           (target_w - new_w) // 2:(target_w - new_w) // 2 + new_w] = resized

    # 3. BGR → RGB
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


class REID:
    def __init__(self):
        self.use_gpu = torch.cuda.is_available()
        self.model = torchreid.models.build_model(
            name='resnet50',
            num_classes=1,
            loss='softmax',
            pretrained=True,
            use_gpu=self.use_gpu
        )
        torchreid.utils.load_pretrained_weights(self.model, 'model_data/models/model.pth')
        if self.use_gpu:
            self.model = self.model.cuda()
        _, self.transform_te = build_transforms(
            height=256, width=128,
            random_erase=False,
            color_jitter=False,
            color_aug=False
        )
        # Use cosine distance for cross-camera matching
        self.dist_metric = 'cosine'
        self.model.eval()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _extract_features(self, tensor):
        self.model.eval()
        return self.model(tensor)

    def _embed_crop(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """Preprocess one BGR crop and return a (1, D) embedding on CPU."""
        img_rgb = preprocess_crop(crop_bgr)
        tensor  = self.transform_te(Image.fromarray(img_rgb))
        tensor  = torch.unsqueeze(tensor, 0)
        if self.use_gpu:
            tensor = tensor.cuda()
        return self._extract_features(tensor).data.cpu()

    # ── Public API ───────────────────────────────────────────────────────────

    def _features(self, imgs):
        """Fallback: extract & average features with no quality info."""
        f = []
        for img in imgs:
            f.append(self._embed_crop(img))
        return torch.cat(f, 0)

    def _features_with_quality(
        self,
        imgs,
        blur_scores,
        blur_threshold: float = BLUR_THRESHOLD,
        ema_alpha: float      = EMA_ALPHA,
    ) -> torch.Tensor:
        """Extract a single EMA-smoothed embedding for one track.

        Algorithm per frame:
          • If blur_score >= blur_threshold → update EMA normally.
          • If blur_score <  blur_threshold AND we already have an EMA → skip
            (reuse last good embedding; don't let the blurry frame corrupt the
            gallery representation).
          • If ALL frames are blurry → fall back to the least-blurry crop.

        Returns a (1, D) tensor.
        """
        ema_emb: torch.Tensor | None = None

        for crop, score in zip(imgs, blur_scores):
            if score < blur_threshold and ema_emb is not None:
                # Blurry frame – keep existing EMA, don't update
                continue
            curr_emb = self._embed_crop(crop)
            if ema_emb is None:
                ema_emb = curr_emb
            else:
                ema_emb = ema_alpha * ema_emb + (1.0 - ema_alpha) * curr_emb

        if ema_emb is None:
            # Every single frame was blurry – use the least-blurry one
            best_idx = int(np.argmax(blur_scores))
            ema_emb  = self._embed_crop(imgs[best_idx])

        return ema_emb   # shape (1, D)

    def compute_distance(self, qf, gf):
        return metrics.compute_distance_matrix(qf, gf, self.dist_metric).numpy()


if __name__ == '__main__':
    reid = REID()
