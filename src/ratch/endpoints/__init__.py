"""Endpoints — ratch's thin clients to the ONLINE model services.

The one boundary between offline compute and online models. Everything here is a
handle to a model service in ``runners/<name>/`` (a Ray Serve deployment
at merge). Each endpoint is a ``Protocol`` with two impls behind one factory:

  * ``Remote*`` — HTTP to the service's URL (Ray Serve at merge). Selected when
    the service's ``MEDIA_<NAME>_URL`` env is set.
  * ``Local*``  — runs the service's own compute in ITS sealed env
    (``uv run --project runners/<name>``), no server needed. The
    pre-merge default.

ratch stages name an endpoint (a Stage's ``client=``) and never import a model
library. Import from the specific module — this package has no barrel.
"""
