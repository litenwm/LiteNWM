"""Generic CEM and trajectory-constraint utilities for the scorer mainline."""

# Keep the package root deliberately empty. ``scoring.features`` imports
# ``planning.costs``; eager imports here would create a planning/scoring cycle.
__all__: list[str] = []
