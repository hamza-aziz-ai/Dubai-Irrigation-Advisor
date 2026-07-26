"""Real observational data sources.

Everything in this package reads from files committed to the repository. The
single function that touches the network (`nasa_power.download`) is a
maintenance tool, run deliberately to refresh the cache - never called from
the library, the tests, the demo, or the dashboard.

That separation is the point. A pipeline that silently fetches at runtime
cannot be reproduced, cannot be reviewed offline, and fails in a way that
looks like a modelling bug.
"""
