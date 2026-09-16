"""Throwaway internal-iteration UI. Not the graded deliverable — see the build order:
this gets replaced by the React/TS app in the extended-scope phase. Calls the agent
graph in-process (no need to run the FastAPI server separately for local iteration).

Run with:
    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.agent.graph import run_query  # noqa: E402
from app.db import get_session  # noqa: E402
from app.models import UserRecord  # noqa: E402
from app.personalization.feedback import record_feedback  # noqa: E402
from app.personalization.temporal_patterns import current_context_boost, detect_recurring_patterns  # noqa: E402
from app.personalization.profile_builder import build_user_profile  # noqa: E402
from sqlmodel import select  # noqa: E402

st.set_page_config(page_title="Agentic File Retrieval — dev UI", layout="wide")


@st.cache_data(ttl=5)
def _load_users():
    with get_session() as session:
        users = session.exec(select(UserRecord)).all()
    return [{"id": u.id, "name": u.name, "persona_key": u.persona_key} for u in users]


users = _load_users()
user_by_id = {u["id"]: u for u in users}

st.title("Agentic File Recommendation and Retrieval — dev UI")
st.caption("Internal iteration prototype. Not the graded deliverable (see README build order).")

col_query, col_user = st.columns([3, 1])
with col_user:
    selected_user_id = st.selectbox(
        "Simulated user", options=[u["id"] for u in users],
        format_func=lambda uid: f"{user_by_id[uid]['name']} ({user_by_id[uid]['persona_key']})",
    )
with col_query:
    query = st.text_input(
        "Query", value="find that thing I was working on with my advisor about audio deepfakes a few weeks ago"
    )

run = st.button("Search", type="primary")

st.divider()

with st.sidebar:
    st.subheader("Personalization insights")
    now = datetime.now()
    profile = build_user_profile(selected_user_id, now=now)
    patterns = detect_recurring_patterns(selected_user_id)
    active_boost = current_context_boost(selected_user_id, now=now)

    st.write("**Preferred file types:**", ", ".join(profile.preferred_file_types[:5]) or "none yet")

    st.write("**Recurring patterns detected:**")
    if patterns:
        for p in patterns[:3]:
            st.write(f"- {p.weekday_name} {p.hour}:00 → {len(p.file_ids)} files (confidence {p.confidence:.2f})")
    else:
        st.write("_none detected_")

    if active_boost:
        st.success(f"It's {now.strftime('%A')} {now.hour}:00 — {len(active_boost)} files from your usual routine are boosted right now.")
    else:
        st.write("_no active recurring-pattern boost right now_")

if run and query.strip():
    with st.spinner("running agent graph..."):
        out = run_query(query, selected_user_id, now=now)

    trace = out["routing_trace"]

    st.subheader(f"Routed to: `{trace['tier'].upper()}`")
    st.write(trace["rationale"])
    st.write(f"LLM calls: **{trace['llm_call_count']}** · Total latency: **{trace['total_duration_ms']:.1f} ms**")

    st.write("**Pipeline trace:**")
    trace_cols = st.columns(len(trace["stages"]) or 1)
    for col, stage in zip(trace_cols, trace["stages"]):
        with col:
            if stage["skipped"]:
                st.markdown(f"⚪ ~~{stage['name']}~~")
                if stage["detail"]:
                    st.caption(stage["detail"])
            else:
                st.markdown(f"✅ **{stage['name']}**")
                st.caption(f"{stage['duration_ms']:.1f} ms")

    st.divider()
    st.subheader(f"Results ({len(out['results'])})")
    for r in out["results"]:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{r['filename']}**  ·  `{r['file_type']}`  ·  {r['topic_cluster']}")
                st.caption(r["explanation"])
            with c2:
                st.metric("score", f"{r['score']:.3f}")
                fb1, fb2 = st.columns(2)
                if fb1.button("👍", key=f"up-{r['file_id']}"):
                    record_feedback(selected_user_id, r["file_id"], query, "thumbs_up")
                    st.toast(f"Recorded 👍 on {r['filename']}")
                if fb2.button("👎", key=f"down-{r['file_id']}"):
                    record_feedback(selected_user_id, r["file_id"], query, "thumbs_down")
                    st.toast(f"Recorded 👎 on {r['filename']}")
