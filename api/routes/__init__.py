"""
API Routes Package

This package contains route handlers organized by feature:
- enrollment.py: WebSocket and REST endpoints for user enrollment
- authentication.py: REST endpoints for authentication (TODO)
- management.py: REST endpoints for user management (TODO)

Author: CS-1
"""

from api.routes.enrollment import router as enrollment_router

__all__ = ["enrollment_router"]
