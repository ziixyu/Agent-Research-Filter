#!/usr/bin/env python3
"""
ui.py — Interactive Streamlit dashboard for the Epistemic Filtering Agent.

Run:
    streamlit run ui.py

This is a READ + LIVE-RECOMPUTE surface over a run agent.py already
produced (run_output.json / sample_run_output.json) plus the persistent
knowledge graph (epistemic_memory.db). It never re-calls PubMed. The
Methodology/Velocity weight sliders recompute posterior scores locally,
from already-extracted telemetry, using the exact same `agent.score_one()`
math the CLI pipeline uses — no duplicated formula to drift out of sync.

Two tabs:
    1. "Epistemic Matrix & Knowledge Graph" — live ranking, physics graph
       with signed citation topology, and the Async Audit Queue / Surrogate
       Inspector for agent.py's non-blocking fail-safe.
    2. "Empirical Prior Convergence & Calibration" — a per-tier convergence
       chart over calibration_history, plus a console that runs
       backtest_calibration.py's adversarial calibration loop live.

Gemini is called live in exactly two places: the counterfactual arbiter
console (Tab 1) and the calibration console (Tab 2, via backtest_calibration).
Both reuse agent.py's retry-wrapped client — nothing here re-implements
the Gemini calling convention.
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


def pubmed_url(pmid: str, meta: dict | None = None) -> str:
    if meta and meta.get("url"):
        return meta["url"]
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


# --------------------------------------------------------------------------
# Sidebar: run selection + live weight sliders (shared across both tabs)
# --------------------------------------------------------------------------

st.title("🧬 Epistemic Filtering Agent — Live Dashboard")
st.caption(
    "Read-only over a completed agent.py run, with live client-side recomputation of "
    "posterior scores and an ML surrogate + async quarantine queue backed by epistemic_memory.db."
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
        help="Weight on [P(E) x L(Absurdity) x PrecisionPenalty]. Velocity weight is set to "
        "1 - w_M so the two always sum to 1.0. Re-sorts the ranking table entirely in "
        "client-side memory — no re-extraction, no API calls.",
    )
    w_velocity = round(1.0 - w_prior, 4)
    st.caption(f"Velocity weight (w_V) = **{w_velocity:.2f}** (auto-set: w_M + w_V = 1.0)")

try:
    data = load_run(run_path)
except FileNotFoundError:
    st.error(f"Could not find {run_path}. Run `python agent.py` first to generate it.")
    st.stop()

conn = get_db_connection()


# --------------------------------------------------------------------------
# Live posterior recompute (pure function reuse — see agent.score_one)
# --------------------------------------------------------------------------


def recompute(paper: dict, w_prior: float, w_velocity: float) -> dict:
    tele = paper["telemetry"]
    # NOTE: pass base_prior_credence (the raw, pre-bonus tier prior), not
    # prior_credence (which is already the boosted effective value) — score_one()
    # applies the preregistration bonus itself, so passing the boosted value
    # back in would double-apply it.
    comp = agent.score_one(
        paper.get("base_prior_credence", paper["prior_credence"]),
        tele["claim_hyperbole"],
        paper["sample_power_weight"],
        paper["velocity_norm"],
        ci_lower=tele.get("ci_lower"),
        ci_upper=tele.get("ci_upper"),
        is_preregistered=tele.get("is_preregistered", False),
        w_prior=w_prior,
        w_velocity=w_velocity,
    )
    merged = dict(paper)
    merged.update(comp)
    merged["prior_credence"] = comp["effective_prior_credence"]
    merged["posterior_score_live"] = comp["posterior_score"]
    return merged


live_ranking = [recompute(p, w_prior, w_velocity) for p in data["full_ranking"]]
live_ranking.sort(key=lambda p: p["posterior_score_live"], reverse=True)

tab1, tab2 = st.tabs(["📊 Epistemic Matrix & Knowledge Graph", "📈 Empirical Prior Convergence & Calibration"])


# ============================================================================
# TAB 1 — Epistemic Matrix & Knowledge Graph
# ============================================================================

with tab1:
    # ------------------------------------------------------------------
    # Dynamic Ranking Matrix
    # ------------------------------------------------------------------

    st.header("📊 Dynamic Ranking Matrix")

    rows = []
    for i, p in enumerate(live_ranking, start=1):
        tele, meta = p["telemetry"], p["metadata"]
        tier = graph_memory.beta_tier_for(tele["study_design"], tele["sample_size"])
        rows.append(
            {
                "Rank": i,
                "PMID": pubmed_url(meta["pmid"], meta),
                "Title": meta["title"][:70] + ("…" if len(meta["title"]) > 70 else ""),
                "DOI": f"https://doi.org/{meta['doi']}" if meta.get("doi") else None,
                "Design": tele["study_design"] + (" 🆕" if graph_memory.is_ood_tier(tier) else ""),
                "N": tele["sample_size"],
                "Hype": tele["claim_hyperbole"],
                "Prereg": "✅" if tele.get("is_preregistered") else "",
                "P(E)": round(p["prior_credence"], 3),
                "D": round(p["discrepancy_index"], 3),
                "L(Absurdity)": round(p["likelihood_penalty"], 3),
                "SE": round(p["standard_error"], 3) if p.get("standard_error") is not None else None,
                "Precision": round(p["precision_penalty"], 3),
                "V_norm": round(p["velocity_norm"], 3),
                "S_posterior (live)": round(p["posterior_score_live"], 4),
                "Audit": p.get("audit_status", "PASSED"),
            }
        )
    df = pd.DataFrame(rows)

    _AUDIT_ROW_COLORS = {
        "FLAGGED": "background-color: #5a1a1a",
        "OVERRIDDEN": "background-color: #5a4a1a",
        "AUTO_RESOLVED_BY_SURROGATE": "background-color: #123a4a",
        "ASYNC_QUARANTINED": "background-color: #3a1a4a",
    }

    def _highlight_audit(row):
        return [_AUDIT_ROW_COLORS.get(row["Audit"], "")] * len(row)

    st.dataframe(
        df.style.apply(_highlight_audit, axis=1),
        width="stretch",
        hide_index=True,
        column_config={
            "PMID": st.column_config.LinkColumn(
                "PMID", display_text=r"https://pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/"
            ),
            "DOI": st.column_config.LinkColumn("DOI", display_text=r"https://doi\.org/(.+)"),
        },
    )
    st.caption(
        "S_posterior (live) recomputes instantly as the sidebar sliders move — pure client-side "
        "re-blend of already-extracted prior_credence/likelihood_penalty/velocity_norm, no API "
        "calls. 🆕 = out-of-distribution study design (Jeffreys-prior tier). PMID/DOI cells are "
        "clickable links."
    )

    # ------------------------------------------------------------------
    # Paper Inspector
    # ------------------------------------------------------------------

    st.header("🔍 Paper Inspector")

    _by_pmid = {p["metadata"]["pmid"]: p for p in live_ranking}
    inspected_pmid = st.selectbox(
        "Select a paper", options=list(_by_pmid.keys()),
        format_func=lambda pmid: f"{pmid} — {_by_pmid[pmid]['metadata']['title'][:60]}",
    )
    if inspected_pmid:
        ip = _by_pmid[inspected_pmid]
        imeta, itele = ip["metadata"], ip["telemetry"]
        itier = graph_memory.beta_tier_for(itele["study_design"], itele["sample_size"])

        badge_bits = []
        if graph_memory.is_ood_tier(itier):
            badge_bits.append(":violet-badge[OOD design]")
        if itele.get("is_preregistered"):
            badge_bits.append(":green-badge[Preregistered]")
        status = ip.get("audit_status")
        if status == "FLAGGED":
            badge_bits.append(":red-badge[FLAGGED]")
        elif status == "OVERRIDDEN":
            badge_bits.append(":orange-badge[OVERRIDDEN]")
        elif status == "AUTO_RESOLVED_BY_SURROGATE":
            badge_bits.append(f":blue-badge[SURROGATE: {ip.get('surrogate_action')}]")
        elif status == "ASYNC_QUARANTINED":
            badge_bits.append(":violet-badge[ASYNC QUARANTINED]")
        if badge_bits:
            st.markdown(" ".join(badge_bits))

        st.markdown(f"### {imeta['title']}")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.link_button("View on PubMed", pubmed_url(inspected_pmid, imeta))
        with c2:
            if imeta.get("doi"):
                st.link_button("View DOI", f"https://doi.org/{imeta['doi']}")

        st.write(f"**Design:** {itele['study_design']} (tier: `{itier}`)  **N:** {itele['sample_size']}  **Year:** {imeta.get('publication_year')}")
        st.write(f"**Claim hyperbole:** {itele['claim_hyperbole']}/5  **Preregistered:** {itele.get('is_preregistered', False)}")
        if itele.get("ci_lower") is not None and itele.get("ci_upper") is not None:
            st.write(f"**95% CI:** [{itele['ci_lower']}, {itele['ci_upper']}]  **p-value:** {itele.get('p_value')}")
        st.write(
            f"**base P(E):** {ip.get('base_prior_credence', ip['prior_credence']):.3f}  "
            f"**+ prereg bonus:** {ip.get('preregistration_bonus', 0.0):.3f}  "
            f"**= effective P(E):** {ip['prior_credence']:.3f}"
        )
        st.write(
            f"**L(Absurdity):** {ip['likelihood_penalty']:.3f}  "
            f"**Precision penalty:** {ip.get('precision_penalty', 1.0):.3f}  "
            f"**S_posterior:** {ip['posterior_score_live']:.4f}"
        )
        with st.expander("Full abstract"):
            st.write(imeta.get("abstract", "(not available)"))

    # ------------------------------------------------------------------
    # Physics-stabilized signed knowledge graph (pyvis)
    # ------------------------------------------------------------------

    st.header("🕸 Knowledge Graph")

    nx_graph = graph_memory.load_graph(conn)

    # If the DB doesn't have this run's papers yet (e.g. dashboard opened
    # before any agent.py run persisted them), seed the live view from the
    # loaded JSON so the graph isn't empty on first use.
    if nx_graph.number_of_nodes() == 0:
        for p in data["full_ranking"]:
            meta, tele = p["metadata"], p["telemetry"]
            tier = graph_memory.beta_tier_for(tele["study_design"], tele["sample_size"])
            nx_graph.add_node(
                meta["pmid"],
                title=meta["title"],
                study_design=tele["study_design"],
                sample_size=tele["sample_size"],
                prior_credence=p["prior_credence"],
                discrepancy_index=p["discrepancy_index"],
                likelihood_penalty=p["likelihood_penalty"],
                posterior_score=p["posterior_score"],
                audit_status=p.get("audit_status", "PASSED"),
                url=pubmed_url(meta["pmid"], meta),
                doi=meta.get("doi"),
                is_ood_design=graph_memory.is_ood_tier(tier),
                is_preregistered=tele.get("is_preregistered", False),
            )

    def score_to_color(score: float) -> str:
        """Green (high credence) -> red (low/penalized). HSL hue 0=red,
        120=green; score is already in [0,1] by construction (see agent.py
        clip01)."""
        score = max(0.0, min(1.0, score))
        hue = int(score * 120)
        return f"hsl({hue}, 70%, 45%)"

    # Signed edge styling — exact spec: SUPPORTING solid green width=2,
    # MENTION slate gray width=1, REFUTING bold red dashed width=3 (a
    # detected contradiction is unconditionally REFUTING too).
    EDGE_STYLE = {
        "SUPPORTING": {"color": "#2ecc71", "width": 2, "dashes": False},
        "MENTION": {"color": "#95a5a6", "width": 1, "dashes": False},
        "REFUTING": {"color": "#e74c3c", "width": 3, "dashes": True},
    }
    NODE_SIZE_MIN, NODE_SIZE_MAX = 10, 55

    def build_pyvis_html(g, prereg_lookup: dict) -> str:
        from pyvis.network import Network

        # cdn_resources="in_line" is required here: pyvis's default
        # ("local") emits <script src="lib/bindings/utils.js"> with a path
        # relative to wherever the HTML is served FROM. That's fine for a
        # standalone file, but this HTML is embedded via
        # st.components.v1.html() inside an iframe whose base URL is
        # Streamlit's own dev server — the relative path resolved to
        # Streamlit's index page instead of pyvis's JS, which the browser
        # then tried (and failed) to parse as a script. Inlining
        # everything sidesteps the base-URL mismatch entirely.
        net = Network(
            height="640px", width="100%", directed=True,
            bgcolor="#0e1117", font_color="#f0f0f0", cdn_resources="in_line",
        )
        for node, attrs in g.nodes(data=True):
            n = attrs.get("sample_size") or 0
            # log10(N+1) scaling, clamped to [NODE_SIZE_MIN, NODE_SIZE_MAX]
            # so a single N=100000 outlier can't swamp a 30-50 node batch
            # visually — explicit min/max bounds, not just a scale factor.
            size = max(NODE_SIZE_MIN, min(NODE_SIZE_MAX, 10 + 8 * math.log10(n + 1)))
            score = attrs.get("posterior_score", 0.5) or 0.5
            color = score_to_color(score)
            url = attrs.get("url") or f"https://pubmed.ncbi.nlm.nih.gov/{node}/"
            # graph_memory.load_graph() computes is_ood_design live from the
            # stored study_design/sample_size; is_preregistered is stored
            # directly on the node (added this pass) but fall back to the
            # loaded run JSON for older DBs that predate that column.
            is_ood = attrs.get("is_ood_design", False)
            is_prereg = attrs.get("is_preregistered") or prereg_lookup.get(node, False)
            badges = ("🆕 OOD  " if is_ood else "") + ("✅ Preregistered" if is_prereg else "")
            # pyvis renders `title` as HTML in the hover tooltip by default,
            # so a real clickable <a> works — "hovering a node exposes the
            # direct PubMed link" per spec (click-to-navigate on the node
            # itself would need custom vis-network click-event JS).
            title = (
                f"PMID {node}<br>{attrs.get('title', '')}<br>"
                f"<a href='{url}' target='_blank'>{url}</a><br>"
                f"Design: {attrs.get('study_design')}<br>N: {n}<br>"
                f"P(E): {attrs.get('prior_credence', 0):.2f}<br>"
                f"D: {attrs.get('discrepancy_index', 0):.2f}<br>"
                f"L(Absurdity): {attrs.get('likelihood_penalty', 0):.2f}<br>"
                f"S_posterior: {score:.3f}<br>"
                f"Audit: {attrs.get('audit_status', 'PASSED')}"
                + (f"<br>{badges}" if badges else "")
            )
            border_width = 4 if (is_ood or is_prereg) else 1
            border_color = "#9b59b6" if is_ood else ("#2ecc71" if is_prereg else color)
            net.add_node(node, label=str(node), size=size, title=title, borderWidth=border_width, shape="dot")
            # pyvis's add_node doesn't cleanly take a separate border color
            # kwarg across versions — set the split fill/border color
            # directly on the node dict it just created instead.
            net.get_node(node)["color"] = {"background": color, "border": border_color}

        for src, dst, attrs in g.edges(data=True):
            sentiment = "REFUTING" if attrs.get("edge_type") == "contradiction" else (attrs.get("sentiment") or "MENTION")
            style = EDGE_STYLE[sentiment]
            detail = attrs.get("detail") or f"citation ({sentiment})"
            net.add_edge(src, dst, color=style["color"], width=style["width"], dashes=style["dashes"], title=detail)

        # forceAtlas2Based with central gravity damping renders 30-50 node
        # batches without the dense-cluster overlap barnesHut alone tends
        # to produce at that scale; springLength/springConstant tuned down
        # so edges don't fight the central pull.
        net.set_options(
            """
            {
              "physics": {
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                  "gravitationalConstant": -80,
                  "centralGravity": 0.010,
                  "springLength": 120,
                  "springConstant": 0.06,
                  "damping": 0.45,
                  "avoidOverlap": 0.6
                },
                "minVelocity": 0.75,
                "stabilization": { "iterations": 200 }
              }
            }
            """
        )
        return net.generate_html(notebook=False)

    prereg_lookup = {
        p["metadata"]["pmid"]: p["telemetry"].get("is_preregistered", False) for p in data["full_ranking"]
    }

    col_legend, col_graph = st.columns([1, 4])
    with col_legend:
        st.markdown(
            "**Legend**\n\n"
            "- Node size ∝ log10(N+1), clamped to a readable range\n"
            "- Node fill: 🟢 high S_posterior → 🔴 low/penalized\n"
            "- Violet border = out-of-distribution design (Jeffreys prior)\n"
            "- Green border = preregistered trial\n"
            "- 🟢 solid edge (width 2) = citation, SUPPORTING\n"
            "- ⬜ slate solid edge (width 1) = citation, MENTION\n"
            "- 🔴 dashed edge (width 3) = REFUTING citation or contradiction\n"
            "- **Hover any node** for its clickable PubMed link + full telemetry\n"
        )
        st.metric("Nodes", nx_graph.number_of_nodes())
        st.metric("Edges", nx_graph.number_of_edges())

    with col_graph:
        if nx_graph.number_of_nodes() == 0:
            st.info("No graph data yet — run `python agent.py` at least once to populate epistemic_memory.db.")
        else:
            html = build_pyvis_html(nx_graph, prereg_lookup)
            st.components.v1.html(html, height=660, scrolling=True)

    # ------------------------------------------------------------------
    # Async Audit Queue & Surrogate Inspector
    # ------------------------------------------------------------------

    st.header("🤖 Async Audit Queue & Surrogate Inspector")

    auto_resolved = [p for p in data["full_ranking"] if p.get("audit_status") == "AUTO_RESOLVED_BY_SURROGATE"]
    if auto_resolved:
        st.subheader(f"Auto-resolved by the ML surrogate ({len(auto_resolved)})")
        for p in auto_resolved:
            meta, tele = p["metadata"], p["telemetry"]
            conf = p.get("surrogate_confidence") or 0.0
            action = p.get("surrogate_action", "?")
            action_color = {"PASS": "green", "CLAMP": "orange", "REJECT": "red"}.get(action, "blue")
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**PMID {meta['pmid']}** — {meta['title'][:80]}")
                    st.caption(f"{tele['study_design']}, N={tele['sample_size']}, hype={tele['claim_hyperbole']}/5")
                with c2:
                    st.markdown(f":{action_color}-badge[{action}]")
                    st.caption(f"confidence {conf:.2f}")
                st.link_button("View on PubMed", pubmed_url(meta["pmid"], meta), key=f"auto_{meta['pmid']}")
    else:
        st.caption("No papers auto-resolved by the surrogate in this run.")

    st.subheader("Asynchronous Quarantine Queue")
    unresolved = graph_memory.get_unresolved_audits(conn)
    if not unresolved:
        st.success("Queue is empty — nothing waiting on manual review.")
    else:
        with st.expander(f"{len(unresolved)} paper(s) awaiting manual review", expanded=True):
            for item in unresolved:
                try:
                    telemetry = json.loads(item["telemetry_json"])
                except (json.JSONDecodeError, TypeError):
                    telemetry = {}
                with st.container(border=True):
                    st.markdown(f"**PMID {item['pmid']}**  ·  queued {item['timestamp']}")
                    st.caption(f"Reason: {item['reason']}")
                    if telemetry:
                        st.caption(
                            f"study_design={telemetry.get('study_design')}, "
                            f"N={telemetry.get('sample_size')}, "
                            f"claim_hyperbole={telemetry.get('claim_hyperbole')}/5"
                        )
                    st.link_button("View on PubMed", pubmed_url(item["pmid"]), key=f"queue_link_{item['id']}")
                    if st.button("Resolve (confirm & release)", key=f"resolve_{item['id']}"):
                        tier = graph_memory.beta_tier_for(
                            telemetry.get("study_design", "Retrospective/Observational"),
                            telemetry.get("sample_size", 0),
                        )
                        graph_memory.record_feedback(
                            conn, tier=tier, action="confirm", pmid=item["pmid"], trigger_source="human"
                        )
                        graph_memory.set_audit_status(conn, item["pmid"], "OVERRIDDEN")
                        graph_memory.delete_unresolved_audit(conn, item["id"])
                        st.success(f"PMID {item['pmid']} resolved and released from the queue.")
                        st.rerun()

    st.subheader("Current learned priors (epistemic_memory.db)")
    prior_rows = [
        {"Tier": tier, "alpha": round(a, 2), "beta": round(b, 2), "Mean P(E)": round(a / (a + b), 4)}
        for tier, (a, b) in graph_memory.get_current_prior_hyperparams(conn).items()
    ]
    st.dataframe(pd.DataFrame(prior_rows), width="stretch", hide_index=True)

    # ------------------------------------------------------------------
    # LLM Arbiter Synthesis & Counterfactual Console
    # ------------------------------------------------------------------

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


# ============================================================================
# TAB 2 — Empirical Prior Convergence & Calibration
# ============================================================================

with tab2:
    st.header("📈 Empirical Prior Convergence")
    st.caption(
        "Every feedback event (human, ML surrogate, or the adversarial calibration engine) "
        "snapshots E[P(E)] = alpha/(alpha+beta) into calibration_history. This tracks how each "
        "tier's credence has moved over time."
    )

    history_df = graph_memory.get_calibration_history_df(conn)
    if history_df.empty:
        st.info(
            "No calibration history yet — calibration_history fills in as soon as any feedback "
            "is recorded (a HITL decision, a surrogate auto-resolution, or a calibration run "
            "below)."
        )
    else:
        wide = history_df.pivot_table(
            index="iteration", columns="tier", values="expected_credence", aggfunc="last"
        )
        # Forward-fill so a tier with fewer updates than another still draws
        # a flat continuation line instead of stopping abruptly at its last
        # real point, which reads as "diverged to zero" on a shared x-axis.
        wide = wide.ffill()
        st.line_chart(wide, height=420)
        st.caption(
            f"{len(history_df)} snapshot(s) across {history_df['tier'].nunique()} tier(s). "
            "X-axis is each tier's own update count (iteration), not wall-clock time — "
            "tiers that have received more feedback have more points."
        )
        with st.expander("Raw calibration_history"):
            st.dataframe(history_df, width="stretch", hide_index=True)

    st.divider()
    st.header("🧪 Live Calibration Console")
    st.caption(
        "Runs backtest_calibration.py's adversarial LLM-as-a-judge critic directly against "
        f"`{run_path}`'s full_ranking, live. Each iteration re-judges every paper and applies "
        "ROBUST/VULNERABLE feedback — this is a self-optimizing loop, not idempotent; repeated "
        "runs keep shifting the same tiers (see backtest_calibration.py's own docstring)."
    )

    calib_iterations = st.slider("Number of iterations", min_value=1, max_value=10, value=1)
    run_calibration = st.button("Execute Autonomous Adversarial Calibration", type="primary")

    if run_calibration:
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            st.error("GEMINI_API_KEY is not set in .env — cannot run the live adversarial judge.")
        else:
            import backtest_calibration as bc
            from google import genai

            papers_to_judge = data.get("full_ranking", [])
            if not papers_to_judge:
                st.warning(f"{run_path} has no full_ranking to calibrate against.")
            else:
                client = genai.Client(api_key=gemini_key)
                model = data.get("model", "gemini-flash-lite-latest")
                progress = st.progress(0.0, text="Starting calibration...")
                log_area = st.empty()
                log_lines: list[str] = []
                for i in range(1, calib_iterations + 1):
                    progress.progress(i / calib_iterations, text=f"Iteration {i}/{calib_iterations}...")
                    pass_result = bc.run_calibration_pass(client, model, conn, papers_to_judge)
                    for r in pass_result["results"]:
                        log_lines.append(f"iter {i} — PMID {r['pmid']} ({r['tier']}): {r['verdict']['verdict']}")
                    log_area.code("\n".join(log_lines[-25:]), language=None)
                progress.progress(1.0, text="Calibration complete.")
                st.success(
                    f"Ran {calib_iterations} iteration(s) over {len(papers_to_judge)} paper(s). "
                    "Priors updated and calibration_history snapshotted — redrawing convergence curves."
                )
                st.rerun()
