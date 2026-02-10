"""
API Routes Package

This package contains route handlers organized by feature:
- enrollment.py: WebSocket and REST endpoints for user enrollment
- authentication.py: REST endpoints for authentication
- management.py: REST endpoints for user management

Author: CS-1
"""

from api.routes.enrollment import router as enrollment_router
from api.routes.enrollment import rest_router as enrollment_rest_router
from api.routes.authentication import router as authentication_router
from api.routes.management import router as management_router

__all__ = [
    "enrollment_router",
    "enrollment_rest_router",
    "authentication_router",
    "management_router",
]
