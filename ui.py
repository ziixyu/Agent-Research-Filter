#!/usr/bin/env python3
"""
ui.py — Interactive Streamlit dashboard for the Epistemic Filtering Agent.

Run:
    streamlit run ui.py

This is a READ + LIVE-RECOMPUTE surface over a run agent.py already
produced (run_output.json / sample_run_output.json) plus the persistent
knowledge graph (epistemic_memory.db). It never re-calls PubMed. The
Methodology/Velocity weight sliders and the HITL override panel recompute
posterior scores locally, from already-extracted telemetry, using the exact
same `agent.score_one()` math the CLI pipeline uses — no duplicated
formula to drift out of sync. The one place this dashboard DOES call
Gemini live is the counterfactual arbiter console, which reuses agent.py's
retry-wrapped client.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pandas as pd
import streamlit as st

import agent
import graph_memory

st.set_page_config(page_title="Epistemic Filtering Agent", layout="wide", page_icon="🧬")

DB_PATH = graph_memory.DB_PATH_DEFAULT


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


@st.cache_data
def load_run(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def available_run_files() -> list[str]:
    candidates = ["run_output.json", "sample_run_output.json"]
    found = [c for c in candidates if Path(c).exists()]
    return found or ["sample_run_output.json"]


def get_db_connection():
    return graph_memory.init_db(DB_PATH)


# --------------------------------------------------------------------------
# Sidebar: run selection + live weight sliders
# --------------------------------------------------------------------------

st.title("🧬 Epistemic Filtering Agent — Live Dashboard")
st.caption(
    "Read-only over a completed agent.py run, with live client-side recomputation of "
    "posterior scores and a Human-in-the-Loop override panel backed by epistemic_memory.db."
)

with st.sidebar:
    st.header("Run data")
    run_path = st.selectbox("Run output file", options=available_run_files())
    if st.button("Reload from disk"):
        st.cache_data.clear()
    st.divider()

    st.header("Reasoning Step 2 weights")
    w_prior = st.slider(
        "Methodology / Prior weight (w_M)", 0.0, 1.0, agent.W_PRIOR, 0.05,
        help="Weight on [P(E) x L(Absurdity)]. Velocity weight is set to 1 - w_M so the two "
        "always sum to 1.0, matching the deterministic formula's design.",
    )
    w_velocity = round(1.0 - w_prior, 4)
    st.caption(f"Velocity weight (w_V) = **{w_velocity:.2f}** (auto-set: w_M + w_V = 1.0)")

try:
    data = load_run(run_path)
except FileNotFoundError:
    st.error(f"Could not find {run_path}. Run `python agent.py` first to generate it.")
    st.stop()


# --------------------------------------------------------------------------
# Live posterior recompute (pure function reuse — see agent.score_one)
# --------------------------------------------------------------------------


def recompute(paper: dict, w_prior: float, w_velocity: float) -> dict:
    tele = paper["telemetry"]
    comp = agent.score_one(
        paper["prior_credence"],
        tele["claim_hyperbole"],
        paper["sample_power_weight"],
        paper["velocity_norm"],
        w_prior=w_prior,
        w_velocity=w_velocity,
    )
    merged = dict(paper)
    merged.update(comp)
    merged["posterior_score_live"] = comp["posterior_score"]
    return merged


live_ranking = [recompute(p, w_prior, w_velocity) for p in data["full_ranking"]]
live_ranking.sort(key=lambda p: p["posterior_score_live"], reverse=True)


# --------------------------------------------------------------------------
# Dynamic Ranking Matrix
# --------------------------------------------------------------------------

st.header("📊 Dynamic Ranking Matrix")

rows = []
for i, p in enumerate(live_ranking, start=1):
    rows.append(
        {
            "Rank": i,
            "PMID": p["metadata"]["pmid"],
            "Title": p["metadata"]["title"][:70] + ("…" if len(p["metadata"]["title"]) > 70 else ""),
            "Design": p["telemetry"]["study_design"],
            "N": p["telemetry"]["sample_size"],
            "Hype": p["telemetry"]["claim_hyperbole"],
            "P(E)": round(p["prior_credence"], 3),
            "D": round(p["discrepancy_index"], 3),
            "L(Absurdity)": round(p["likelihood_penalty"], 3),
            "V_norm": round(p["velocity_norm"], 3),
            "S_posterior (live)": round(p["posterior_score_live"], 4),
            "Audit": p.get("audit_status", "PASSED"),
        }
    )
df = pd.DataFrame(rows)


def _highlight_audit(row):
    color = {"FLAGGED": "background-color: #5a1a1a", "OVERRIDDEN": "background-color: #5a4a1a"}.get(
        row["Audit"], ""
    )
    return [color] * len(row)


st.dataframe(df.style.apply(_highlight_audit, axis=1), width='stretch', hide_index=True)
st.caption(
    "S_posterior (live) recomputes instantly as you move the sliders — it uses the SAME "
    "prior_credence, likelihood_penalty and velocity_norm the CLI already extracted, just "
    "re-blended with your chosen weights. It does not re-run extraction or re-call Gemini."
)


# --------------------------------------------------------------------------
# Physics-based interactive graph (pyvis)
# --------------------------------------------------------------------------

st.header("🕸 Knowledge Graph")

conn = get_db_connection()
nx_graph = graph_memory.load_graph(conn)

# If the DB doesn't have this run's papers yet (e.g. dashboard opened before
# any agent.py run persisted them), seed the live view from the loaded JSON
# so the graph isn't empty on first use.
if nx_graph.number_of_nodes() == 0:
    for p in data["full_ranking"]:
        nx_graph.add_node(
            p["metadata"]["pmid"],
            title=p["metadata"]["title"],
            study_design=p["telemetry"]["study_design"],
            sample_size=p["telemetry"]["sample_size"],
            prior_credence=p["prior_credence"],
            discrepancy_index=p["discrepancy_index"],
            likelihood_penalty=p["likelihood_penalty"],
            posterior_score=p["posterior_score"],
            audit_status=p.get("audit_status", "PASSED"),
        )


def score_to_color(score: float) -> str:
    """Green (high credence) -> red (low/penalized). HSL hue 0=red,
    120=green; score is already in [0,1] by construction (see agent.py
    clip01)."""
    score = max(0.0, min(1.0, score))
    hue = int(score * 120)
    return f"hsl({hue}, 70%, 45%)"


def build_pyvis_html(g) -> str:
    from pyvis.network import Network

    # cdn_resources="in_line" is required here: pyvis's default ("local")
    # emits <script src="lib/bindings/utils.js"> with a path relative to
    # wherever the HTML is served FROM. That's fine for a standalone file,
    # but this HTML is embedded via st.components.v1.html() inside an
    # iframe whose base URL is Streamlit's own dev server — the relative
    # path resolved to Streamlit's index page instead of pyvis's JS, which
    # the browser then tried (and failed) to parse as a script. Inlining
    # everything sidesteps the base-URL mismatch entirely.
    net = Network(
        height="600px", width="100%", directed=True,
        bgcolor="#0e1117", font_color="#f0f0f0", cdn_resources="in_line",
    )
    for node, attrs in g.nodes(data=True):
        n = attrs.get("sample_size") or 0
        size = 12 + 9 * math.log10(n + 1)
        score = attrs.get("posterior_score", 0.5) or 0.5
        color = score_to_color(score)
        title = (
            f"PMID {node}<br>{attrs.get('title', '')}<br>"
            f"Design: {attrs.get('study_design')}<br>N: {n}<br>"
            f"P(E): {attrs.get('prior_credence', 0):.2f}<br>"
            f"D: {attrs.get('discrepancy_index', 0):.2f}<br>"
            f"L(Absurdity): {attrs.get('likelihood_penalty', 0):.2f}<br>"
            f"S_posterior: {score:.3f}<br>"
            f"Audit: {attrs.get('audit_status', 'PASSED')}"
        )
        net.add_node(node, label=str(node), size=size, color=color, title=title)
    for src, dst, attrs in g.edges(data=True):
        if attrs.get("edge_type") == "contradiction":
            net.add_edge(src, dst, color="#e74c3c", width=3, dashes=True, title=attrs.get("detail", "contradiction"))
        else:
            net.add_edge(src, dst, color="#7f8c8d", title="citation")
    net.set_options(
        """
        { "physics": { "barnesHut": { "gravitationalConstant": -12000, "springLength": 150 },
                       "minVelocity": 0.75, "stabilization": {"iterations": 150} } }
        """
    )
    return net.generate_html(notebook=False)


col_legend, col_graph = st.columns([1, 4])
with col_legend:
    st.markdown(
        "**Legend**\n\n"
        "- Node size ∝ log10(N)\n"
        "- Node color: 🟢 high S_posterior → 🔴 low/penalized\n"
        "- Gray solid edge = citation (real PubMed link data)\n"
        "- Red dashed edge = candidate contradiction (keyword heuristic)\n"
    )
    st.metric("Nodes", nx_graph.number_of_nodes())
    st.metric("Edges", nx_graph.number_of_edges())

with col_graph:
    if nx_graph.number_of_nodes() == 0:
        st.info("No graph data yet — run `python agent.py` at least once to populate epistemic_memory.db.")
    else:
        html = build_pyvis_html(nx_graph)
        st.components.v1.html(html, height=620, scrolling=True)


# --------------------------------------------------------------------------
# HITL Override Panel
# --------------------------------------------------------------------------

st.header("🚨 HITL Override Panel")

flagged = [p for p in data["full_ranking"] if p.get("audit_status") == "FLAGGED"]
if not flagged:
    st.success("No papers currently flagged for audit in this run.")
else:
    st.warning(f"{len(flagged)} paper(s) flagged for epistemic audit.")
    for p in flagged:
        meta, tele = p["metadata"], p["telemetry"]
        with st.expander(f"⚠ PMID {meta['pmid']} — {meta['title'][:80]}", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Design:** {tele['study_design']}  **N:** {tele['sample_size']}")
                st.write(f"**Claim hyperbole:** {tele['claim_hyperbole']}/5")
                st.write(f"**Discrepancy Index D:** {p['discrepancy_index']:.2f}")
            with c2:
                st.write(f"**Current P(E):** {p['prior_credence']:.2f}")
                st.write(f"**L(Absurdity):** {p['likelihood_penalty']:.2f}")
                st.write(f"**S_posterior:** {p['posterior_score']:.3f}")

            new_p = st.slider(
                "Manually clamp Prior P(E)", 0.0, 1.0, float(p["prior_credence"]), 0.01,
                key=f"prior_slider_{meta['pmid']}",
            )
            comp = agent.score_one(new_p, tele["claim_hyperbole"], p["sample_power_weight"], p["velocity_norm"])
            st.caption(f"Preview: overriding P(E) -> {new_p:.2f} would change S_posterior to **{comp['posterior_score']:.3f}** (was {p['posterior_score']:.3f})")

            b1, b2 = st.columns(2)
            with b1:
                if st.button(f"Apply override for {meta['pmid']}", key=f"apply_{meta['pmid']}"):
                    tier = graph_memory.beta_tier_for(tele["study_design"], tele["sample_size"])
                    graph_memory.record_feedback(conn, tier=tier, action="override", manual_p=new_p, pmid=meta["pmid"])
                    graph_memory.upsert_node(
                        conn, pmid=meta["pmid"], title=meta["title"], study_design=tele["study_design"],
                        sample_size=tele["sample_size"], prior_credence=new_p,
                        discrepancy_index=comp["discrepancy_index"], likelihood_penalty=comp["likelihood_penalty"],
                        posterior_score=comp["posterior_score"], audit_status="OVERRIDDEN",
                    )
                    st.success(f"Recorded. Beta hyperparameters for tier '{tier}' updated — next `python agent.py` run will load this.")
                    st.rerun()
            with b2:
                if st.button(f"Quarantine {meta['pmid']}", key=f"quarantine_{meta['pmid']}"):
                    tier = graph_memory.beta_tier_for(tele["study_design"], tele["sample_size"])
                    graph_memory.record_feedback(conn, tier=tier, action="reject", pmid=meta["pmid"])
                    graph_memory.set_audit_status(conn, meta["pmid"], "OVERRIDDEN")
                    st.success(f"PMID {meta['pmid']} recorded as rejected. Beta tier '{tier}' updated.")
                    st.rerun()

st.subheader("Current learned priors (epistemic_memory.db)")
prior_rows = [
    {"Tier": tier, "alpha": round(a, 2), "beta": round(b, 2), "Mean P(E)": round(a / (a + b), 4)}
    for tier, (a, b) in graph_memory.get_current_prior_hyperparams(conn).items()
]
st.dataframe(pd.DataFrame(prior_rows), width='stretch', hide_index=True)


# --------------------------------------------------------------------------
# LLM Arbiter Synthesis & Counterfactual Console
# --------------------------------------------------------------------------

st.header("⚖️ Arbiter Synthesis & Counterfactual Console")

verdict = data.get("arbiter_verdict")
if verdict:
    for j in verdict["justifications"]:
        st.markdown(f"**Rank {j['rank']} — PMID {j['pmid']}**")
        st.write(j["justification"])
    st.info(f"**Overall defense:** {verdict['overall_defense']}")
else:
    st.warning("No arbiter verdict present in this run's output.")

st.subheader("Ask a counterfactual")
scenario = st.text_input(
    "e.g. \"Re-evaluate the ranking if liver stiffness endpoints are excluded\"",
    key="counterfactual_input",
)
if st.button("Ask Gemini", type="primary"):
    if not scenario.strip():
        st.warning("Enter a scenario first.")
    else:
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            st.error("GEMINI_API_KEY is not set in .env — cannot call the live arbiter.")
        else:
            from google import genai

            client = genai.Client(api_key=gemini_key)
            model = data.get("model", "gemini-flash-lite-latest")
            with st.spinner("Asking Gemini for a counterfactual defense..."):
                try:
                    answer = agent.counterfactual_arbiter(client, model, data["top3"], scenario)
                    st.markdown(answer)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Counterfactual call failed: {e}")
