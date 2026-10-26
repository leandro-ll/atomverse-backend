"""Pure chemistry domain adapters shared by the backend services.

Modules in this package read the generated catalog binding and contain no
FastAPI, provider, environment, clock, or random dependency, so the same
chemistry can be exercised directly in tests.
"""
