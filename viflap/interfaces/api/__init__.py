"""The HTTP interface.

``create_app`` is an application factory taking an assembled container, so the
API can be tested against in-memory adapters without a database, a model file or
a network.

The transport layer enforces two safety properties that would otherwise depend
on each client getting them right: no response carries a likelihood ratio
without its prior, and no response renders a strength band without its
direction.
"""

from viflap.interfaces.api.app import create_app
from viflap.interfaces.api.dependencies import (
    ApplicationContainer,
    HeaderPrincipalResolver,
)

__all__ = ["ApplicationContainer", "HeaderPrincipalResolver", "create_app"]
