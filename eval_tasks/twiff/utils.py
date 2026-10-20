"""TwiFF-Bench task utils for the Hindsight VEP-OPSD eval.

Thin re-export: the protocol (prompt, frame handling, judge prompt) lives in the
vendored lmms-eval task module so this repo cannot drift from it.
"""

from lmms_eval.tasks.twiffbench.utils import (  # noqa: F401
    twiffbench_aggregate_results,
    twiffbench_doc_to_text,
    twiffbench_doc_to_visual,
    twiffbench_process_results,
)
