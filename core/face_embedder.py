"""
Face Embedding Extractor using ArcFace

Extracts 512-dimensional identity-discriminative embeddings from face images
using a pre-trained ArcFace model. These embeddings are the primary identity
signal for face authentication, replacing MASt3R descriptors which lack
identity discrimination.

Supports two backends:
  - insightface (preferred): buffalo_l model bundle with SCRFD + ArcFace R100
  - facenet-pytorch (fallback): InceptionResnetV1 with VGGFace2 pretraining

Usage:
    from core.face_embedder import FaceEmbedder

    embedder = FaceEmbedder(config)
    embedder.load_model()

    # Single image
    embedding = embedder.extract_embedding(face_crop_bgr)  # (512,)

    # Multi-frame aggregation (enrollment)
    embedding = embedder.extract_multi_frame(face_crops)    # (512,)
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Backend availability flags
_INSIGHTFACE_AVAILABLE = False
_FACENET_AVAILABLE = False

try:
    from insightface.app import FaceAnalysis
    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    pass

try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    import torch
    _FACENET_AVAILABLE = True
except ImportError:
    pass


class FaceEmbedder:
    """
    Extract identity embeddings from face images using ArcFace.

    The embedder handles model loading, face detection/alignment (internal
    to the recognition model), and embedding extraction. It produces
    L2-normalized 512-dim vectors suitable for cosine similarity comparison.

    Args:
        config: Dictionary with keys:
            - model: Model name ("buffalo_l", "buffalo_sc", or "vggface2")
            - embedding_dim: Expected embedding dimension (default 512)
            - device: "cuda" or "cpu"
            - backend: "insightface" or "facenet" (auto-detected if omitted)
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = {}

        self.model_name = config.get("model", "buffalo_l")
        self.embedding_dim = config.get("embedding_dim", 512)
        self.device = config.get("device", "cuda")

        # Auto-detect backend
        requested_backend = config.get("backend", "auto")
        if requested_backend == "auto":
            if _INSIGHTFACE_AVAILABLE:
                self.backend = "insightface"
            elif _FACENET_AVAILABLE:
                self.backend = "facenet"
            else:
                raise ImportError(
                    "No face embedding backend available. "
                    "Install insightface: pip install insightface onnxruntime-gpu\n"
                    "Or facenet-pytorch: pip install facenet-pytorch"
                )
        else:
            self.backend = requested_backend

        self._model = None
        self._detector = None  # For facenet backend
        self.is_loaded = False

    def load_model(self) -> None:
        """Load the face recognition model. Call this before extract_embedding."""
        if self.is_loaded:
            return

        if self.backend == "insightface":
            self._load_insightface()
        elif self.backend == "facenet":
            self._load_facenet()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        self.is_loaded = True
        logger.info(f"FaceEmbedder loaded (backend={self.backend}, model={self.model_name})")

    def _load_insightface(self) -> None:
        """Load insightface model bundle."""
        if not _INSIGHTFACE_AVAILABLE:
            raise ImportError("insightface not installed. Run: pip install insightface onnxruntime-gpu")

        # Determine providers based on device
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self._model = FaceAnalysis(
            name=self.model_name,
            providers=providers,
        )
        # det_size controls the internal face detection input size
        self._model.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))

    def _load_facenet(self) -> None:
        """Load facenet-pytorch model."""
        if not _FACENET_AVAILABLE:
            raise ImportError("facenet-pytorch not installed. Run: pip install facenet-pytorch")

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        self._detector = MTCNN(
            image_size=160,
            margin=20,
            device=device,
            select_largest=True,
        )
        self._model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a 512-dim identity embedding from a face image.

        Args:
            face_image: Face crop in BGR format (H, W, 3), uint8.
                        Can be any size — internal alignment handles resizing.

        Returns:
            L2-normalized embedding as float32 ndarray of shape (512,),
            or None if no face is detected in the image.
        """
        if not self.is_loaded:
            self.load_model()

        if self.backend == "insightface":
            return self._extract_insightface(face_image)
        else:
            return self._extract_facenet(face_image)

    def _extract_insightface(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding using insightface."""
        # insightface expects BGR input (same as OpenCV)
        faces = self._model.get(face_image)

        if not faces:
            logger.warning("insightface detected no face, trying with padded input")
            # 0.5 padding makes the face ~25% of padded image area,
            # comparable to the 0.3 face_padding used in the capture pipeline
            # where SCRFD reliably detects faces
            padded = self._pad_image(face_image, ratio=0.5)
            faces = self._model.get(padded)

        if faces:
            # Select the face with highest detection score
            best_face = max(faces, key=lambda f: f.det_score)
            embedding = best_face.normed_embedding  # Already L2-normalized, (512,)
            return embedding.astype(np.float32)

        # Final fallback: bypass SCRFD detection entirely and run ArcFace
        # directly on the image. Valid when input is already a face crop.
        logger.warning("Detection failed, using direct ArcFace recognition bypass")
        return self._direct_arcface_embed(face_image)

    def _extract_facenet(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding using facenet-pytorch."""
        import cv2

        # facenet-pytorch expects RGB PIL-style input
        rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Detect and align face
        face_tensor = self._detector(rgb)
        if face_tensor is None:
            logger.warning("MTCNN detected no face, trying center crop fallback")
            face_tensor = self._center_crop_tensor(rgb)

        if face_tensor is None:
            return None

        # Ensure batch dimension
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)

        device = next(self._model.parameters()).device
        face_tensor = face_tensor.to(device)

        with torch.no_grad():
            embedding = self._model(face_tensor).cpu().numpy().flatten()

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def _center_crop_tensor(self, rgb_image: np.ndarray) -> Optional["torch.Tensor"]:
        """Fallback: center-crop and resize to 160x160 for facenet-pytorch."""
        h, w = rgb_image.shape[:2]
        size = min(h, w)
        y_start = (h - size) // 2
        x_start = (w - size) // 2
        cropped = rgb_image[y_start:y_start + size, x_start:x_start + size]

        import cv2
        resized = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_AREA)

        # Convert to tensor in [-1, 1] range (facenet-pytorch convention)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float()
        tensor = (tensor - 127.5) / 128.0
        return tensor

    def _direct_arcface_embed(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Run ArcFace recognition directly on a pre-cropped face image,
        bypassing SCRFD face detection entirely.

        This is a fallback for images where the face fills the frame and
        SCRFD cannot locate it. The image is resized to 112x112 (ArcFace
        standard input) and fed directly to the ONNX recognition model.
        """
        import cv2

        # Find the recognition model within FaceAnalysis
        rec_model = None
        for model in self._model.models:
            if hasattr(model, "taskname") and model.taskname == "recognition":
                rec_model = model
                break

        if rec_model is None:
            logger.error("No recognition model found in FaceAnalysis")
            return None

        # Resize to 112x112 (ArcFace standard input size)
        aligned = cv2.resize(face_image, (112, 112), interpolation=cv2.INTER_AREA)

        # Prepare blob: (1, 3, 112, 112), normalized to [-1, 1]
        blob = cv2.dnn.blobFromImage(
            aligned, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
        )

        # Run ONNX inference directly
        input_name = rec_model.session.get_inputs()[0].name
        output_name = rec_model.session.get_outputs()[0].name
        pred = rec_model.session.run([output_name], {input_name: blob})[0]

        embedding = pred.flatten()
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    @staticmethod
    def _pad_image(image: np.ndarray, ratio: float = 0.2) -> np.ndarray:
        """Add padding around the image to help face detection on tight crops."""
        h, w = image.shape[:2]
        pad_h = int(h * ratio)
        pad_w = int(w * ratio)

        # Use mean color for padding (matches MASt3R convention)
        mean_color = image.mean(axis=(0, 1)).astype(np.uint8)
        padded = np.full(
            (h + 2 * pad_h, w + 2 * pad_w, 3),
            mean_color,
            dtype=np.uint8,
        )
        padded[pad_h:pad_h + h, pad_w:pad_w + w] = image
        return padded

    def extract_multi_frame(
        self,
        face_images: List[np.ndarray],
        quality_scores: Optional[List[float]] = None,
    ) -> Optional[np.ndarray]:
        """
        Extract embeddings from multiple frames and aggregate.

        Uses quality-weighted mean aggregation: frontal/sharp frames
        contribute more to the final template embedding.

        Args:
            face_images: List of face crops in BGR format.
            quality_scores: Optional per-frame quality weights.
                           Higher = better quality. If None, uniform weights.

        Returns:
            L2-normalized aggregated embedding (512,), or None if no faces detected.
        """
        embeddings = []
        weights = []

        for i, img in enumerate(face_images):
            emb = self.extract_embedding(img)
            if emb is not None:
                embeddings.append(emb)
                w = quality_scores[i] if quality_scores is not None else 1.0
                weights.append(w)

        if not embeddings:
            logger.error("No face detected in any of the input frames")
            return None

        embeddings = np.array(embeddings)  # (K, 512)
        weights = np.array(weights, dtype=np.float32)

        # Normalize weights
        weights = weights / weights.sum()

        # Weighted mean
        aggregated = (embeddings * weights[:, np.newaxis]).sum(axis=0)

        # Re-normalize to unit length
        norm = np.linalg.norm(aggregated)
        if norm > 1e-8:
            aggregated = aggregated / norm

        logger.info(
            f"Aggregated {len(embeddings)}/{len(face_images)} frame embeddings "
            f"(norm={np.linalg.norm(aggregated):.4f})"
        )
        return aggregated.astype(np.float32)

    def extract_per_frame(
        self, face_images: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Extract embeddings from each frame individually.

        Args:
            face_images: List of face crops in BGR format.

        Returns:
            Tuple of:
              - embeddings: (K, 512) float32 array of successful extractions
              - valid_indices: list of indices into face_images that succeeded
        """
        embeddings = []
        valid_indices = []

        for i, img in enumerate(face_images):
            emb = self.extract_embedding(img)
            if emb is not None:
                embeddings.append(emb)
                valid_indices.append(i)

        if not embeddings:
            return np.empty((0, self.embedding_dim), dtype=np.float32), []

        return np.array(embeddings, dtype=np.float32), valid_indices
