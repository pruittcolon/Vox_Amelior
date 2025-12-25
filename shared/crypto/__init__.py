"""
Cryptographic Module Package.

Provides enterprise-grade cryptographic operations:
- HSM integration for key storage
- FIPS 140-2 compliant algorithms
- Automated key rotation
- Cryptographic agility for algorithm migration

Part of the 6-Month Security Hardening Plan - Month 2.
"""

__version__ = "2.0.0"

# Lazy imports to avoid circular dependencies and missing module errors during initial creation
def get_hsm_provider():
    """Get the configured HSM provider."""
    from shared.crypto.hsm_provider import get_hsm_provider as _get
    return _get()

def get_key_rotation_manager():
    """Get the key rotation manager."""
    from shared.crypto.key_rotation import get_key_rotation_manager as _get
    return _get()

def get_fips_crypto():
    """Get the FIPS-compliant crypto wrapper."""
    from shared.crypto.fips_wrapper import FIPSCrypto
    return FIPSCrypto()

__all__ = [
    "get_hsm_provider",
    "get_key_rotation_manager", 
    "get_fips_crypto",
]
