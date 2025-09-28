"""
json_schema_validator.py
JSON schema validation for compression agent outputs.
Ensures reliable and consistent response formats from LLM providers.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised for JSON schema validation errors."""
    pass


class CompressionSchemaType(Enum):
    """Types of compression schemas."""
    MID_TERM = "mid_term"
    LONG_TERM = "long_term"
    GENERAL = "general"


@dataclass
class ValidationResult:
    """Result of JSON schema validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_data: Optional[Dict[str, Any]] = None


class JSONSchemaValidator:
    """Validates JSON responses against defined schemas."""
    
    def __init__(self):
        """Initialize the schema validator."""
        self.schemas = self._define_schemas()
        logger.info("JSON schema validator initialized")
    
    def _define_schemas(self) -> Dict[CompressionSchemaType, Dict[str, Any]]:
        """Define JSON schemas for different compression types."""
        return {
            CompressionSchemaType.MID_TERM: {
                "type": "object",
                "required": ["summary", "key_points", "user_preferences", "metadata"],
                "properties": {
                    "summary": {
                        "type": "string",
                        "minLength": 10,
                        "maxLength": 2000
                    },
                    "key_points": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "minLength": 5
                        },
                        "minItems": 1,
                        "maxItems": 10
                    },
                    "user_preferences": {
                        "type": "object",
                        "additionalProperties": True
                    },
                    "metadata": {
                        "type": "object",
                        "required": ["compression_type", "original_length"],
                        "properties": {
                            "compression_type": {
                                "type": "string",
                                "enum": ["mid_term"]
                            },
                            "original_length": {
                                "type": "integer",
                                "minimum": 0
                            },
                            "compression_ratio": {
                                "type": "string"
                            }
                        }
                    }
                }
            },
            
            CompressionSchemaType.LONG_TERM: {
                "type": "object",
                "required": ["brief", "patterns", "sentiment", "preferences", "metadata"],
                "properties": {
                    "brief": {
                        "type": "string",
                        "minLength": 10,
                        "maxLength": 1000
                    },
                    "patterns": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "minLength": 5
                        },
                        "minItems": 1,
                        "maxItems": 15
                    },
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral", "mixed"]
                    },
                    "preferences": {
                        "type": "object",
                        "additionalProperties": True
                    },
                    "metadata": {
                        "type": "object",
                        "required": ["compression_type", "original_length"],
                        "properties": {
                            "compression_type": {
                                "type": "string",
                                "enum": ["long_term"]
                            },
                            "original_length": {
                                "type": "integer",
                                "minimum": 0
                            },
                            "insights_count": {
                                "type": "string"
                            }
                        }
                    }
                }
            },
            
            CompressionSchemaType.GENERAL: {
                "type": "object",
                "required": ["summary", "metadata"],
                "properties": {
                    "summary": {
                        "type": "string",
                        "minLength": 10,
                        "maxLength": 2000
                    },
                    "metadata": {
                        "type": "object",
                        "required": ["compression_type", "original_length"],
                        "properties": {
                            "compression_type": {
                                "type": "string",
                                "enum": ["general"]
                            },
                            "original_length": {
                                "type": "integer",
                                "minimum": 0
                            }
                        }
                    }
                }
            }
        }
    
    def validate_compression_response(
        self, 
        data: Dict[str, Any], 
        schema_type: CompressionSchemaType
    ) -> ValidationResult:
        """
        Validate compression response against schema.
        
        Args:
            data: JSON data to validate
            schema_type: Type of compression schema to use
            
        Returns:
            ValidationResult with validation status and details
        """
        if not isinstance(data, dict):
            return ValidationResult(
                is_valid=False,
                errors=[f"Expected dict, got {type(data).__name__}"],
                warnings=[]
            )
        
        schema = self.schemas.get(schema_type)
        if not schema:
            return ValidationResult(
                is_valid=False,
                errors=[f"Unknown schema type: {schema_type}"],
                warnings=[]
            )
        
        errors = []
        warnings = []
        
        # Validate required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings
            )
        
        # Validate each field
        properties = schema.get("properties", {})
        sanitized_data = {}
        
        for field, field_schema in properties.items():
            if field in data:
                field_result = self._validate_field(
                    field, data[field], field_schema
                )
                if field_result["errors"]:
                    errors.extend(field_result["errors"])
                if field_result["warnings"]:
                    warnings.extend(field_result["warnings"])
                
                # Add sanitized field value
                sanitized_data[field] = field_result["value"]
            else:
                # Field not present, use default if available
                if "default" in field_schema:
                    sanitized_data[field] = field_schema["default"]
        
        # Add any additional fields that weren't in schema
        for field, value in data.items():
            if field not in properties:
                sanitized_data[field] = value
                warnings.append(f"Unexpected field: {field}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized_data if len(errors) == 0 else None
        )
    
    def _validate_field(
        self, 
        field_name: str, 
        value: Any, 
        field_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a single field against its schema."""
        errors = []
        warnings = []
        sanitized_value = value
        
        # Type validation
        expected_type = field_schema.get("type")
        if expected_type:
            if not self._check_type(value, expected_type):
                errors.append(
                    f"Field '{field_name}': expected {expected_type}, got {type(value).__name__}"
                )
                return {"errors": errors, "warnings": warnings, "value": value}
        
        # String validations
        if expected_type == "string" and isinstance(value, str):
            min_length = field_schema.get("minLength")
            max_length = field_schema.get("maxLength")
            
            if min_length is not None and len(value) < min_length:
                errors.append(
                    f"Field '{field_name}': string too short (min {min_length} chars)"
                )
            elif max_length is not None and len(value) > max_length:
                # Truncate if too long
                sanitized_value = value[:max_length]
                warnings.append(
                    f"Field '{field_name}': string truncated to {max_length} chars"
                )
        
        # Array validations
        elif expected_type == "array" and isinstance(value, list):
            items_schema = field_schema.get("items", {})
            min_items = field_schema.get("minItems")
            max_items = field_schema.get("maxItems")
            
            if min_items is not None and len(value) < min_items:
                errors.append(
                    f"Field '{field_name}': array too short (min {min_items} items)"
                )
            elif max_items is not None and len(value) > max_items:
                # Truncate if too long
                sanitized_value = value[:max_items]
                warnings.append(
                    f"Field '{field_name}': array truncated to {max_items} items"
                )
            
            # Validate array items
            if items_schema:
                sanitized_items = []
                for i, item in enumerate(sanitized_value):
                    item_result = self._validate_field(
                        f"{field_name}[{i}]", item, items_schema
                    )
                    if item_result["errors"]:
                        errors.extend(item_result["errors"])
                    if item_result["warnings"]:
                        warnings.extend(item_result["warnings"])
                    sanitized_items.append(item_result["value"])
                sanitized_value = sanitized_items
        
        # Object validations
        elif expected_type == "object" and isinstance(value, dict):
            object_schema = field_schema.get("properties", {})
            required_fields = field_schema.get("required", [])
            
            # Check required fields
            for req_field in required_fields:
                if req_field not in value:
                    errors.append(
                        f"Field '{field_name}.{req_field}': required field missing"
                    )
            
            # Validate object properties
            sanitized_obj = {}
            for prop_name, prop_value in value.items():
                if prop_name in object_schema:
                    prop_result = self._validate_field(
                        f"{field_name}.{prop_name}", prop_value, object_schema[prop_name]
                    )
                    if prop_result["errors"]:
                        errors.extend(prop_result["errors"])
                    if prop_result["warnings"]:
                        warnings.extend(prop_result["warnings"])
                    sanitized_obj[prop_name] = prop_result["value"]
                else:
                    sanitized_obj[prop_name] = prop_value
                    warnings.append(f"Field '{field_name}.{prop_name}': unexpected field")
            
            sanitized_value = sanitized_obj
        
        # Enum validation
        enum_values = field_schema.get("enum")
        if enum_values and value not in enum_values:
            errors.append(
                f"Field '{field_name}': value must be one of {enum_values}, got '{value}'"
            )
        
        # Integer validations
        elif expected_type == "integer" and isinstance(value, (int, float)):
            if not isinstance(value, int):
                sanitized_value = int(value)
                warnings.append(f"Field '{field_name}': converted to integer")
            
            minimum = field_schema.get("minimum")
            maximum = field_schema.get("maximum")
            
            if minimum is not None and sanitized_value < minimum:
                errors.append(
                    f"Field '{field_name}': value too small (min {minimum})"
                )
            elif maximum is not None and sanitized_value > maximum:
                errors.append(
                    f"Field '{field_name}': value too large (max {maximum})"
                )
        
        return {
            "errors": errors,
            "warnings": warnings,
            "value": sanitized_value
        }
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if not expected_python_type:
            return True  # Unknown type, skip validation
        
        if expected_type == "number":
            return isinstance(value, expected_python_type)
        else:
            return isinstance(value, expected_python_type)
    
    def validate_and_sanitize(
        self, 
        data: Union[str, Dict[str, Any]], 
        schema_type: CompressionSchemaType
    ) -> Tuple[bool, Optional[Dict[str, Any]], List[str]]:
        """
        Validate and sanitize compression response.
        
        Args:
            data: JSON string or dict to validate
            schema_type: Type of compression schema
            
        Returns:
            Tuple of (is_valid, sanitized_data, error_messages)
        """
        # Parse JSON if string
        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
            except json.JSONDecodeError as e:
                return False, None, [f"Invalid JSON: {e}"]
        else:
            parsed_data = data
        
        # Validate against schema
        result = self.validate_compression_response(parsed_data, schema_type)
        
        if result.is_valid:
            return True, result.sanitized_data, result.warnings
        else:
            return False, None, result.errors


# Global schema validator instance
_schema_validator = None


def get_schema_validator() -> JSONSchemaValidator:
    """Get the global schema validator instance."""
    global _schema_validator
    if _schema_validator is None:
        _schema_validator = JSONSchemaValidator()
    return _schema_validator


def validate_compression_output(
    data: Union[str, Dict[str, Any]], 
    compression_type: str
) -> Tuple[bool, Optional[Dict[str, Any]], List[str]]:
    """Convenience function to validate compression output."""
    validator = get_schema_validator()
    
    try:
        schema_type = CompressionSchemaType(compression_type)
    except ValueError:
        return False, None, [f"Unknown compression type: {compression_type}"]
    
    return validator.validate_and_sanitize(data, schema_type)
