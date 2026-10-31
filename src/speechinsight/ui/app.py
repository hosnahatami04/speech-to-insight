"""Streamlit meeting viewer.

Launch with:
    uv run streamlit run src/speechinsight/ui/app.py

The viewer reads the processing API over HTTP, lists utterances colored by
speaker, and lets the reader click a decision or action to highlight the
utterances it was taken from. The implementation here is intentionally thin —
most heavy lifting (color assignment, evidence lookup) is pure functions
exposed in ``ui_data`` so they can be tested without Streamlit installed.
"""

from __future__ import annotations

import os
import time
from typing import Any, cast

import httpx

API_URL = os.environ.get("SPEECHINSIGHT_API", "http://localhost:8000")


def _poll_job(client: httpx.Client, job_id: str, timeout_s: float = 600.0) -> dict[str, Any]:
    """Poll /jobs/{id} until the job is terminal or the timeout fires."""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        resp = client.get(f"{API_URL}/jobs/{job_id}")
        resp.raise_for_status()
        payload = cast("dict[str, Any]", resp.json())
        if payload["status"] in {"done", "error", "extractor_unavailable"}:
            return payload
        time.sleep(1.5)
    raise TimeoutError(f"job {job_id} did not finish within {timeout_s:.0f}s")


def _render() -> None:
    import streamlit as st

    st.set_page_config(page_title="speech-to-insight", layout="wide")
    st.title("Meeting viewer")

    uploaded = st.file_uploader("Audio file (wav/flac/mp3)", type=["wav", "flac", "mp3"])
    if not uploaded:
        st.info("Upload a meeting recording to begin.")
        return

    with httpx.Client(timeout=30.0) as client:
        with st.spinner("Uploading..."):
            resp = client.post(
                f"{API_URL}/process",
                files={"file": (uploaded.name, uploaded.getvalue())},
            )
            resp.raise_for_status()
            job_id = resp.json()["job_id"]
        with st.spinner("Processing (this takes a while on CPU)..."):
            result = _poll_job(client, job_id)

    if result["status"] == "error":
        st.error(result.get("error", "unknown error"))
        return

    _show_result(result)


def _show_result(result: dict[str, Any]) -> None:
    import streamlit as st

    from speechinsight.ui.ui_data import color_for_speaker, utterances_for_evidence

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Transcript")
        for utt in result["transcript"]:
            color = color_for_speaker(utt["speaker"])
            st.markdown(
                f"<span style='background:{color};padding:0 .3em;border-radius:3px'>"
                f"<b>{utt['speaker']}</b></span> "
                f"<small>[{utt['id']}]</small> {utt['text']}",
                unsafe_allow_html=True,
            )
    with right:
        for kind in ("decisions", "actions"):
            st.subheader(kind.title())
            for item in result.get(kind, []):
                flag = "" if item["supported"] else " (unsupported)"
                owner = f" — {item['owner']}" if item.get("owner") else ""
                due = f" (due {item['due']})" if item.get("due") else ""
                with st.expander(f"{item['text']}{owner}{due}{flag}"):
                    for utt in utterances_for_evidence(item["evidence"], result["transcript"]):
                        st.markdown(f"**[{utt['id']}]** {utt['speaker']}: {utt['text']}")
                    for reason in item.get("reasons", []):
                        st.caption(f"⚠ {reason}")


if __name__ == "__main__":
    _render()
