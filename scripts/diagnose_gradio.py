"""
Inference smoke-test for the Gradio demo backend (``src/gradio_app.py``).

For a handful of sample users + cart scenarios, this script calls
``make_predictions`` directly (no HTTP server needed) and asserts the
sanity property:

    "All Top-10 products recommended by NCF / XGBoost / Sequential
     Transformer must be products the user has bought in their prior
     history."

Run it from the project root::

    .venv\\Scripts\\python.exe scripts\\diagnose_gradio.py

Originally written to diagnose a bug where the Gradio Transformer column
was returning products the user had never bought (and NaN scores when the
cart was empty). The fix lives in ``gradio_app.build_transformer_input``
and the candidate-restricted ranking inside ``make_predictions``. Keep
this script around as a regression check for future Gradio changes.
"""

import os
import sys
import re
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config
from src import gradio_app as ga

TAG = "[gradio-smoke]"


def user_prior_product_ids(user_id):
    """Return the set of product_ids this user has historically bought."""
    user_order_ids = set(ga.orders_df[ga.orders_df['user_id'] == user_id]['order_id'])
    user_prior = ga.prior_df[ga.prior_df['order_id'].isin(user_order_ids)]
    return set(int(pid) for pid in user_prior['product_id'].tolist())


_NAME_SCORE_RE = re.compile(
    r'<div style="font-weight: 600.*?>([^<]+)</div>\s*'
    r'<div style="color: #0070f3.*?>([^<:]+):\s*([0-9.]+)</div>',
    re.DOTALL,
)


def parse_html_cards(html):
    """Parse the cards rendered by ``make_html_cards``."""
    rows = []
    for m in _NAME_SCORE_RE.finditer(html):
        rows.append((m.group(1).strip(), m.group(2).strip(), float(m.group(3))))
    return rows


def diagnose_user(user_id, cart_items):
    print(f"\n{TAG} ============================================")
    print(f"{TAG} User {user_id}, cart={cart_items}")
    print(f"{TAG} ============================================")

    prior_pids = user_prior_product_ids(user_id)
    print(f"{TAG} User has bought {len(prior_pids)} distinct products historically.")

    tf_html, xgb_html, mf_html = ga.make_predictions(user_id, cart_items)

    failures = []
    for label, html in [("Transformer", tf_html), ("XGBoost", xgb_html), ("NCF", mf_html)]:
        rows = parse_html_cards(html)
        print(f"\n{TAG} -- {label} Top-{len(rows)} --")
        if not rows:
            failures.append(f"{label}: 0 rows parsed")
            print("    (no recommendations parsed)")
            continue
        seen_count = 0
        for rank, (name, score_text, val) in enumerate(rows, 1):
            pid = ga.prod_name_to_id.get(name)
            ever_bought = pid in prior_pids if pid is not None else False
            seen_count += int(ever_bought)
            tag = "[OWNED]" if ever_bought else "[NEVER-BOUGHT]"
            print(f"    {rank:2d}. {tag:15s} score={val:.4f}  pid={pid}  {name}")
        print(f"{TAG} {label}: {seen_count}/{len(rows)} in user prior history.")
        if seen_count < len(rows):
            failures.append(f"{label}: {len(rows) - seen_count} unseen products in Top-10")
    return failures


if __name__ == "__main__":
    user_id_a = int(ga.sample_user_ids[0])
    user_id_b = int(ga.sample_user_ids[3])

    scenarios = [
        ("A: empty cart (default flow)", user_id_a, []),
        # Mixed cart formats: legacy product-name string + new pid integer.
        ("B: cart with one common item", user_id_a, ["Banana"]),
        ("C: another user, empty cart", user_id_b, []),
    ]

    all_failures = []
    for title, uid, cart in scenarios:
        print(f"\n{TAG} === Scenario {title} ===")
        all_failures.extend(diagnose_user(uid, cart))

    print()
    if all_failures:
        print(f"{TAG} FAIL: {len(all_failures)} issue(s) found:")
        for f in all_failures:
            print(f"        - {f}")
        sys.exit(2)
    print(f"{TAG} PASS: all Top-10 recommendations sit inside the user's prior history.")
