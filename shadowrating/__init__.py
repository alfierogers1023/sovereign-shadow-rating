"""Sovereign credit shadow-rating model."""
import os

import certifi

# python.org's macOS build doesn't wire stdlib urllib up to a CA bundle by
# default, which breaks fredapi (it calls urllib directly, unlike the other
# loaders which go through requests). Point it at certifi's bundle before any
# loader has a chance to make a request.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

__version__ = "0.1.0"
