"""MLflow models-from-code entry script for the Matching Agent.

Logged via ``mlflow.pyfunc.log_model(python_model="<this file>")`` so MLflow
loads the model definition by *executing this script fresh* at load time,
rather than pickling an already-imported (and potentially stale) module from
``code_paths``. This guarantees the served model runs the current
``MatchingAgentModel.predict`` implementation.

The repo's ``src`` package is shipped alongside via ``code_paths`` so the
``from src.agent.matching_agent import ...`` below resolves at load time.
"""

import mlflow

from src.agent.matching_agent import make_pyfunc_model

# set_model registers the model instance MLflow should serve.
mlflow.models.set_model(make_pyfunc_model())
