"""FutureBench task utils for the Hindsight VEP-OPSD eval.

Thin re-export: the protocol (option parsing, hop-wise aggregation) lives in the
vendored lmms-eval task module so this repo cannot drift from it.
"""

from lmms_eval.tasks.futurebench.utils import (  # noqa: F401
    future_pred_aggregate_hop1,
    future_pred_aggregate_hop2,
    future_pred_aggregate_hop3,
    future_pred_aggregate_hop5,
    future_pred_aggregate_results,
    future_pred_doc_to_target,
    future_pred_doc_to_text,
    future_pred_doc_to_visual,
    future_pred_process_reasoning_results,
)
