"""
Matching Module for Face Authentication

This package contains the matching algorithms for comparing face templates.
The interfaces are defined by CS-1, and the implementations are provided
by the DS team.

Components:
    - interfaces: Abstract base classes for matchers (CS-1 defines)
    - geometric_matcher: 3D shape comparison using ICP (DS-1 implements)
    - descriptor_matcher: Feature vector comparison (DS-1 implements)
    - score_fusion: Combining match scores (DS-1 implements)

Usage:
    from core.matching import ICPGeometricMatcher, NNDescriptorMatcher, WeightedFusion
    # or use stubs for early testing:
    from core.matching import StubGeometricMatcher, StubDescriptorMatcher, StubScoreFusion
"""

from core.matching.interfaces import (
    MatchResult,
    GeometricMatcher,
    DescriptorMatcher,
    ScoreFusion,
    StubGeometricMatcher,
    StubDescriptorMatcher,
    StubScoreFusion,
)

# DS-1 implementations (import with fallback for environments where
# dependencies like open3d may not be installed)
try:
    from core.matching.geometric_matcher import ICPGeometricMatcher
except ImportError:
    ICPGeometricMatcher = None

try:
    from core.matching.descriptor_matcher import NNDescriptorMatcher
except ImportError:
    NNDescriptorMatcher = None

try:
    from core.matching.score_fusion import WeightedFusion
except ImportError:
    WeightedFusion = None

__all__ = [
    # Data classes
    "MatchResult",
    # Abstract interfaces
    "GeometricMatcher",
    "DescriptorMatcher",
    "ScoreFusion",
    # Stub implementations (for testing before DS team delivers)
    "StubGeometricMatcher",
    "StubDescriptorMatcher",
    "StubScoreFusion",
    # DS-1 concrete implementations
    "ICPGeometricMatcher",
    "NNDescriptorMatcher",
    "WeightedFusion",
]
