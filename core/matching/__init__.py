"""
Matching Module for Face Authentication

This package contains the matching algorithms for comparing face templates.
The interfaces are defined by CS-1, and the implementations are provided
by the DS team.

Components:
    - interfaces: Abstract base classes for matchers (CS-1 defines)
    - geometric_matcher: 3D shape comparison using ICP (DS team implements)
    - descriptor_matcher: Feature vector comparison (DS team implements)
    - score_fusion: Combining match scores (DS team implements)

Usage:
    from core.matching.interfaces import MatchResult
    from core.matching.geometric_matcher import ICPGeometricMatcher
    from core.matching.descriptor_matcher import NNDescriptorMatcher
    from core.matching.score_fusion import WeightedFusion
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
]
