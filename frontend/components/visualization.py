"""
Visualization component for MASt3R Face Authentication System.
CS-2 Primary Ownership.

Handles 3D point cloud visualization and match overlay display.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PointCloudData:
    """Container for point cloud visualization data."""
    points: np.ndarray  # (N, 3) xyz coordinates
    colors: Optional[np.ndarray] = None  # (N, 3) RGB values 0-255
    confidence: Optional[np.ndarray] = None  # (N,) confidence scores


class PointCloudVisualizer:
    """
    Visualizes 3D point clouds in the browser using Plotly.
    
    Responsibilities:
    - Render enrollment point clouds
    - Render probe vs template comparison
    - Highlight matched points
    - Provide interactive 3D rotation/zoom
    """
    
    def __init__(self, max_points: int = 10000):
        """
        Args:
            max_points: Maximum points to display (subsampled for performance).
        """
        self.max_points = max_points
    
    def subsample(self, data: PointCloudData) -> PointCloudData:
        """Subsample point cloud for visualization performance."""
        n_points = len(data.points)
        if n_points <= self.max_points:
            return data
        
        idx = np.random.choice(n_points, self.max_points, replace=False)
        
        return PointCloudData(
            points=data.points[idx],
            colors=data.colors[idx] if data.colors is not None else None,
            confidence=data.confidence[idx] if data.confidence is not None else None,
        )
    
    def create_plotly_figure(
        self,
        data: PointCloudData,
        title: str = "3D Face Point Cloud",
        point_size: float = 1.5,
    ) -> Dict:
        """
        Create a Plotly figure dict for the point cloud.
        
        Args:
            data: Point cloud data to visualize.
            title: Figure title.
            point_size: Size of points in the scatter plot.
        
        Returns:
            Plotly figure dict suitable for gr.Plot.
        """
        data = self.subsample(data)
        pts = data.points
        
        # Generate colors
        if data.colors is not None:
            colors = [
                f'rgb({r},{g},{b})' 
                for r, g, b in data.colors.astype(int)
            ]
        else:
            # Default: color by depth (z-axis)
            z_norm = (pts[:, 2] - pts[:, 2].min()) / (pts[:, 2].ptp() + 1e-6)
            colors = [
                f'rgb({int(255*z)},{int(100*(1-z))},{int(255*(1-z))})' 
                for z in z_norm
            ]
        
        figure = {
            "data": [{
                "type": "scatter3d",
                "x": pts[:, 0].tolist(),
                "y": pts[:, 1].tolist(),
                "z": pts[:, 2].tolist(),
                "mode": "markers",
                "marker": {
                    "size": point_size,
                    "color": colors,
                },
                "hoverinfo": "skip",
            }],
            "layout": {
                "title": title,
                "scene": {
                    "aspectmode": "data",
                    "xaxis": {"title": "X"},
                    "yaxis": {"title": "Y"},
                    "zaxis": {"title": "Z"},
                },
                "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
                "width": 500,
                "height": 400,
            }
        }
        
        return figure
    
    def create_comparison_figure(
        self,
        template: PointCloudData,
        probe: PointCloudData,
        matched_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict:
        """
        Create a side-by-side or overlaid comparison of template and probe.
        
        Args:
            template: Enrolled template point cloud.
            probe: Authentication probe point cloud.
            matched_pairs: List of (template_idx, probe_idx) matched point pairs.
        
        Returns:
            Plotly figure dict.
        """
        template = self.subsample(template)
        probe = self.subsample(probe)
        
        # Offset probe for side-by-side view
        offset = template.points[:, 0].max() - probe.points[:, 0].min() + 0.1
        probe_shifted = probe.points.copy()
        probe_shifted[:, 0] += offset
        
        traces = [
            {
                "type": "scatter3d",
                "x": template.points[:, 0].tolist(),
                "y": template.points[:, 1].tolist(),
                "z": template.points[:, 2].tolist(),
                "mode": "markers",
                "marker": {"size": 1.5, "color": "blue"},
                "name": "Template",
            },
            {
                "type": "scatter3d",
                "x": probe_shifted[:, 0].tolist(),
                "y": probe_shifted[:, 1].tolist(),
                "z": probe_shifted[:, 2].tolist(),
                "mode": "markers",
                "marker": {"size": 1.5, "color": "red"},
                "name": "Probe",
            },
        ]
        
        return {
            "data": traces,
            "layout": {
                "title": "Template vs Probe Comparison",
                "scene": {"aspectmode": "data"},
                "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
                "width": 700,
                "height": 400,
            }
        }
    
    @staticmethod
    def create_empty_figure(message: str = "No point cloud data") -> Dict:
        """Create an empty placeholder figure."""
        return {
            "data": [],
            "layout": {
                "title": message,
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "annotations": [{
                    "text": message,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 16, "color": "gray"},
                }],
                "width": 500,
                "height": 400,
            }
        }


def load_npz_point_cloud(filepath: str) -> Optional[PointCloudData]:
    """
    Load point cloud data from an .npz file.
    
    Args:
        filepath: Path to the .npz file.
    
    Returns:
        PointCloudData or None if loading fails.
    """
    try:
        data = np.load(filepath, allow_pickle=True)
        return PointCloudData(
            points=data["point_cloud"],
            colors=data.get("colors"),
            confidence=data.get("confidence"),
        )
    except Exception as e:
        print(f"[Visualization] Failed to load {filepath}: {e}")
        return None
