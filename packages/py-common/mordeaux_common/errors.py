"""Custom exceptions for Mordeaux services."""


class MordeauxError(Exception):
    """Base exception for Mordeaux services."""
    
    def __init__(self, message: str, code: str = "INTERNAL_ERROR", context: dict = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.context = context or {}


class ValidationError(MordeauxError):
    """Validation error."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message, "VALIDATION_ERROR", context)


class ServiceError(MordeauxError):
    """Service error."""
    
    def __init__(self, message: str, code: str = "SERVICE_ERROR", context: dict = None):
        super().__init__(message, code, context)


class AuthenticationError(MordeauxError):
    """Authentication error."""
    
    def __init__(self, message: str = "Authentication required", context: dict = None):
        super().__init__(message, "AUTHENTICATION_ERROR", context)


class AuthorizationError(MordeauxError):
    """Authorization error."""
    
    def __init__(self, message: str = "Insufficient permissions", context: dict = None):
        super().__init__(message, "AUTHORIZATION_ERROR", context)


class NotFoundError(MordeauxError):
    """Resource not found error."""
    
    def __init__(self, message: str = "Resource not found", context: dict = None):
        super().__init__(message, "NOT_FOUND", context)
