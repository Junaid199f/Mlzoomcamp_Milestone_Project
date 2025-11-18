"""Script that documents the business problem and how ML helps solve it."""

from __future__ import annotations

import textwrap

from config import DATA_PATH

PROBLEM_STATEMENT = """
We want to identify fraudulent transactions on an e-commerce platform before
they are approved. Historical payment data in {data_path} contains device-,
behavior-, and geography-based signals about every purchase. The column
``is_fraud`` tells us which transactions were flagged after the fact, making
the task a supervised binary classification problem.

Machine learning helps because real-time rule systems cannot catch the subtle
interactions between dozens of numerical and categorical features: the channel
used, geographic differences between card origin and shipping destination,
transaction size relative to the customer's past behavior, and various
authorization steps such as AVS, CVV, or 3-D Secure results. A trained model
can learn a non-linear decision boundary that summarizes these signals into a
fraud probability so operations teams can auto-block, step-up, or fast-track
transactions based on risk.
"""


def describe_problem() -> str:
    """Return a clean paragraph that explains the project."""

    return textwrap.dedent(PROBLEM_STATEMENT.format(data_path=DATA_PATH.name)).strip()


if __name__ == "__main__":
    print(describe_problem())
