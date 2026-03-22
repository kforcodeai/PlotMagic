from __future__ import annotations


class ProviderError(RuntimeError):
    """Base class for provider integration failures."""


class ProviderUnavailable(ProviderError):
    """Raised when provider SDK/runtime is unavailable."""


class ProviderTimeout(ProviderError):
    """Raised when provider call exceeds timeout budget."""


class ProviderAuthError(ProviderError):
    """Raised when provider credentials are missing/invalid."""


class ProviderConfigurationError(ProviderError):
    """Raised when provider configuration is invalid."""
