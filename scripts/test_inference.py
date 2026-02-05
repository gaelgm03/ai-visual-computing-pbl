#!/usr/bin/env python3
"""
============================================================
MASt3R Inference Smoke Test
============================================================
This script verifies that MASt3R is properly installed and working.
It runs a basic inference test with dummy images to ensure:
  1. The model can be loaded
  2. GPU memory is sufficient
  3. Output shapes are correct

WHAT IS A SMOKE TEST?
  A smoke test is a quick preliminary test to check if the basic
  functionality works. Like checking if smoke comes out of a machine
  when you turn it on - if there's smoke, something's wrong!

USAGE:
  python scripts/test_inference.py

  Optional arguments:
    --use-real-images    Use sample face images instead of random noise
    --cpu                Force CPU inference (slow but works without GPU)
    --verbose            Show detailed output

EXPECTED OUTPUT:
  If successful, you'll see:
    - GPU information
    - Model loading confirmation
    - Output tensor shapes
    - "All tests passed!" message

Author: CS-1
============================================================
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path

# Add project paths for imports
# This allows us to import MASt3R even if it's not installed system-wide
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
MAST3R_PATH = PROJECT_ROOT / "third_party" / "mast3r"
DUST3R_PATH = MAST3R_PATH / "dust3r"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MAST3R_PATH))
sys.path.insert(0, str(DUST3R_PATH))


def print_header(text: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def print_status(text: str, status: str = "INFO"):
    """Print a status message with color coding."""
    colors = {
        "INFO": "\033[92m",   # Green
        "WARN": "\033[93m",   # Yellow
        "ERROR": "\033[91m",  # Red
        "OK": "\033[94m",     # Blue
    }
    reset = "\033[0m"
    color = colors.get(status, "")
    print(f"{color}[{status}]{reset} {text}")


def check_gpu_availability():
    """Check if CUDA GPU is available and print info."""
    print_header("GPU Check")

    try:
        import torch
    except ImportError:
        print_status("PyTorch not installed. Run: pip install torch", "ERROR")
        return False, None

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print_status(f"GPU detected: {gpu_name}", "OK")
        print_status(f"GPU memory: {gpu_mem:.1f} GB", "INFO")

        # Warning for low VRAM
        if gpu_mem < 8:
            print_status(
                f"Low VRAM ({gpu_mem:.1f}GB). May need to reduce batch size.",
                "WARN"
            )

        return True, torch.device("cuda")
    else:
        print_status("No GPU detected. Using CPU (will be very slow).", "WARN")
        return False, torch.device("cpu")


def check_mast3r_import():
    """Check if MASt3R modules can be imported."""
    print_header("Import Check")

    required_modules = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow (PIL)"),
    ]

    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print_status(f"{display_name} imported successfully", "OK")
        except ImportError as e:
            print_status(f"Failed to import {display_name}: {e}", "ERROR")
            return False

    # Check MASt3R-specific imports
    try:
        from mast3r.model import AsymmetricMASt3R
        print_status("MASt3R model module imported successfully", "OK")
    except ImportError as e:
        print_status(f"Failed to import MASt3R: {e}", "ERROR")
        print_status("Make sure you ran the setup script first!", "WARN")
        return False

    try:
        from dust3r.inference import inference
        from dust3r.utils.image import load_images
        print_status("DUSt3R utilities imported successfully", "OK")
    except ImportError as e:
        print_status(f"Failed to import DUSt3R utilities: {e}", "ERROR")
        return False

    return True


def load_mast3r_model(device):
    """Load the MASt3R model."""
    print_header("Model Loading")

    import torch
    from mast3r.model import AsymmetricMASt3R

    print_status("Loading MASt3R model (this may take 30-60 seconds)...", "INFO")

    try:
        # Load pre-trained model from Hugging Face Hub
        # This automatically downloads the weights if not cached
        model = AsymmetricMASt3R.from_pretrained(
            "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        )
        model = model.to(device)
        model.eval()  # Set to evaluation mode (important for inference)

        print_status("Model loaded successfully!", "OK")

        # Print model info
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print_status(f"Model parameters: {n_params:.1f}M", "INFO")

        return model

    except Exception as e:
        print_status(f"Failed to load model: {e}", "ERROR")
        return None


def create_test_images(use_real_images: bool = False):
    """Create or load test images for inference."""
    print_header("Test Image Preparation")

    import numpy as np
    from PIL import Image

    # Create a temporary directory for test images
    temp_dir = tempfile.mkdtemp()

    if use_real_images:
        # Check if sample images exist
        samples_dir = PROJECT_ROOT / "tests" / "fixtures"
        if samples_dir.exists():
            sample_files = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.png"))
            if len(sample_files) >= 2:
                print_status(f"Using real sample images from {samples_dir}", "OK")
                return [str(f) for f in sample_files[:2]]

        print_status("No sample images found, falling back to synthetic images", "WARN")

    # Create synthetic test images (random noise + gradient for some structure)
    print_status("Creating synthetic test images...", "INFO")

    for i in range(2):
        # Create an image with some structure (not pure noise)
        # This helps test that the model can find correspondences
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Add a gradient background
        for x in range(640):
            img[:, x, :] = int(255 * x / 640)

        # Add some random colored rectangles (simulated "features")
        np.random.seed(42 + i)  # Reproducible but different per image
        for _ in range(10):
            x1, y1 = np.random.randint(0, 600), np.random.randint(0, 440)
            x2, y2 = x1 + np.random.randint(20, 60), y1 + np.random.randint(20, 60)
            color = np.random.randint(0, 255, 3).tolist()
            img[y1:y2, x1:x2] = color

        # Save the image
        img_path = os.path.join(temp_dir, f"test_image_{i+1}.jpg")
        Image.fromarray(img).save(img_path)
        print_status(f"Created test image: {img_path}", "OK")

    return [
        os.path.join(temp_dir, "test_image_1.jpg"),
        os.path.join(temp_dir, "test_image_2.jpg")
    ]


def run_inference_test(model, image_paths, device, verbose: bool = False):
    """Run inference on a pair of test images."""
    print_header("Inference Test")

    import torch
    from dust3r.inference import inference
    from dust3r.utils.image import load_images

    print_status("Loading images...", "INFO")

    # Load images using DUSt3R's utility function
    # The 'size' parameter resizes images to fit within 512x512
    images = load_images(image_paths, size=512)
    print_status(f"Loaded {len(images)} images", "OK")

    print_status("Running inference (this may take 10-30 seconds)...", "INFO")

    try:
        # Enable mixed precision for memory efficiency
        # FP16 (half precision) uses less GPU memory
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            with torch.no_grad():  # Disable gradient computation (saves memory)
                # Run inference on the image pair
                # inference() returns predictions for both views
                output = inference([tuple(images)], model, device, batch_size=1)

        print_status("Inference completed successfully!", "OK")

        # Extract and verify outputs
        view1, view2 = output["view1"], output["view2"]

        # MASt3R outputs explained:
        # - pts3d: 3D point positions for each pixel (the "pointmap")
        # - pts3d_in_other_view: 3D points from view2 expressed in view1's coordinate frame
        # - conf: Confidence values (how sure the model is about each point)
        # - desc: Dense local descriptors (feature vectors for matching)

        print_header("Output Shapes")

        outputs_to_check = [
            ("Pointmap 1 (3D coords)", view1["pts3d"], 4, 3),
            ("Pointmap 2 (in view1 frame)", view2["pts3d_in_other_view"], 4, 3),
            ("Confidence 1", view1["conf"], 3, None),
            ("Confidence 2", view2["conf"], 3, None),
            ("Descriptors 1", view1["desc"], 4, None),
            ("Descriptors 2", view2["desc"], 4, None),
        ]

        all_passed = True
        for name, tensor, expected_dims, last_dim in outputs_to_check:
            shape = tensor.shape
            print_status(f"{name}: {shape}", "INFO")

            # Verify dimensions
            if tensor.dim() != expected_dims:
                print_status(
                    f"  Expected {expected_dims} dimensions, got {tensor.dim()}",
                    "ERROR"
                )
                all_passed = False

            if last_dim is not None and shape[-1] != last_dim:
                print_status(
                    f"  Expected last dim={last_dim}, got {shape[-1]}",
                    "ERROR"
                )
                all_passed = False

        if verbose:
            # Print more detailed statistics
            print_header("Output Statistics")
            pts3d = view1["pts3d"].cpu().numpy()
            print_status(f"Point cloud range: X[{pts3d[...,0].min():.2f}, {pts3d[...,0].max():.2f}]", "INFO")
            print_status(f"                   Y[{pts3d[...,1].min():.2f}, {pts3d[...,1].max():.2f}]", "INFO")
            print_status(f"                   Z[{pts3d[...,2].min():.2f}, {pts3d[...,2].max():.2f}]", "INFO")

            conf = view1["conf"].cpu().numpy()
            print_status(f"Confidence range: [{conf.min():.3f}, {conf.max():.3f}]", "INFO")
            print_status(f"Confidence mean: {conf.mean():.3f}", "INFO")

            desc = view1["desc"]
            print_status(f"Descriptor dimension: {desc.shape[-1]}", "INFO")

        return all_passed

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print_status("GPU out of memory!", "ERROR")
            print_status("Try one of the following:", "WARN")
            print_status("  1. Close other GPU applications", "WARN")
            print_status("  2. Use smaller images (reduce image_size in config)", "WARN")
            print_status("  3. Run with --cpu flag (slow but works)", "WARN")
        else:
            print_status(f"Inference failed: {e}", "ERROR")
        return False


def cleanup_test_images(image_paths):
    """Remove temporary test images."""
    for path in image_paths:
        try:
            os.remove(path)
        except:
            pass

    # Try to remove the temp directory
    try:
        temp_dir = os.path.dirname(image_paths[0])
        os.rmdir(temp_dir)
    except:
        pass


def main():
    """Main function to run all smoke tests."""
    parser = argparse.ArgumentParser(
        description="MASt3R Inference Smoke Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--use-real-images",
        action="store_true",
        help="Use sample face images instead of synthetic images"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference (very slow, but works without GPU)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output statistics"
    )
    args = parser.parse_args()

    print_header("MASt3R Smoke Test")
    print("This test verifies that MASt3R is properly installed.")
    print("Expected duration: 1-3 minutes on GPU, 5-10 minutes on CPU")

    # Step 1: Check GPU
    has_gpu, device = check_gpu_availability()
    if args.cpu:
        import torch
        device = torch.device("cpu")
        print_status("Forced CPU mode (--cpu flag)", "WARN")

    # Step 2: Check imports
    if not check_mast3r_import():
        print_header("TEST FAILED")
        print_status("Import check failed. Please run the setup script.", "ERROR")
        print("  Bash:       bash scripts/setup_mast3r.sh")
        print("  PowerShell: .\\scripts\\setup_mast3r.ps1")
        return 1

    # Step 3: Load model
    model = load_mast3r_model(device)
    if model is None:
        print_header("TEST FAILED")
        print_status("Model loading failed.", "ERROR")
        return 1

    # Step 4: Prepare test images
    image_paths = create_test_images(args.use_real_images)

    # Step 5: Run inference
    try:
        success = run_inference_test(model, image_paths, device, args.verbose)
    finally:
        cleanup_test_images(image_paths)

    # Final result
    print_header("TEST RESULT")
    if success:
        print_status("All tests passed!", "OK")
        print()
        print("MASt3R is properly installed and working.")
        print("You can now proceed with the face authentication pipeline.")
        print()
        print("Next steps:")
        print("  1. Capture test face images")
        print("  2. Run the enrollment pipeline")
        print("  3. Test authentication")
        return 0
    else:
        print_status("Some tests failed. Check the errors above.", "ERROR")
        return 1


if __name__ == "__main__":
    sys.exit(main())
