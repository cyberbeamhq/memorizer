"""
API Module
GraphQL and REST API endpoints for the Memorizer framework.
"""

from .framework_api import app as api_app
from .graphql_api import MemorizerGraphQLAPI, create_graphql_schema

__all__ = [
    "api_app",
    "MemorizerGraphQLAPI",
    "create_graphql_schema",
]
