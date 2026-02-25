# pyre-ignore-all-errors
import csv
import io
import json
from typing import Any, Dict, List

import requests
import streamlit as st

API_BASE_DEFAULT = "http://localhost:8000"
RUN_PATH_DEFAULT = "/api/v1/run"
REQUEST_TIMEOUT_SECONDS = 180
SEPARATOR = "----"


def inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: "Space Grotesk", sans-serif;
}

.app-hero {
    border: 1px solid #dbe6f5;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    background: linear-gradient(135deg, #f8fbff 0%, #edf4ff 55%, #e7f7f5 100%);
    margin-bottom: 1rem;
}

.app-hero h1 {
    margin: 0;
    font-size: 1.55rem;
    line-height: 1.2;
    color: #132a43;
}

.app-hero p {
    margin: 0.35rem 0 0 0;
    color: #3f5873;
    font-size: 0.95rem;
}

.status-chip {
    display: inline-block;
    border-radius: 999px;
    padding: 0.2rem 0.65rem;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.01em;
}

.status-shortlist { background: #e8f7ee; color: #137333; border: 1px solid #b7e1c2; }
.status-reject { background: #fdecec; color: #b42318; border: 1px solid #f6c4c4; }
.status-pending { background: #fff6df; color: #9a6700; border: 1px solid #f2d48a; }
.status-unknown { background: #edf1f7; color: #364152; border: 1px solid #d0d7e2; }

div[data-testid="stExpander"] {
    border-radius: 10px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
}

.stButton > button {
    border-radius: 10px;
    font-weight: 600;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _build_url(base: str, path: str) -> str:
    base_clean = (base or "").strip().rstrip("/")
    path_clean = (path or "").strip()
    if not path_clean.startswith("/"):
        path_clean = "/" + path_clean
    return f"{base_clean}{path_clean}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _status_class(status: str) -> str:
    s = (status or "unknown").strip().lower()
    if s == "shortlist":
        return "status-shortlist"
    if s == "reject":
        return "status-reject"
    if s == "pending":
        return "status-pending"
    return "status-unknown"


def _status_chip(status: str) -> str:
    label = status or "unknown"
    return f"<span class='status-chip {_status_class(label)}'>{label}</span>"


def _read_pdf(buf: io.BytesIO) -> str:
    import pdfplumber  # pyre-ignore[21]

    with pdfplumber.open(buf) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def _read_docx(buf: io.BytesIO) -> str:
    from docx import Document  # pyre-ignore[21]

    doc = Document(buf)
    return "\n".join(p.text for p in doc.paragraphs)


def _read_txt(buf: io.BytesIO) -> str:
    return buf.read().decode("utf-8", errors="replace")


def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    buf = io.BytesIO(uploaded_file.read())
    if name.endswith(".pdf"):
        return _read_pdf(buf)
    if name.endswith(".docx"):
        return _read_docx(buf)
    return _read_txt(buf)


def parse_resumes(raw: str, sep: str = SEPARATOR) -> List[str]:
    return [blk.strip() for blk in (raw or "").split(sep) if blk.strip()]


def _auth_headers(bearer_token: str | None) -> Dict[str, str]:
    token = (bearer_token or "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def safe_post(
    url: str,
    payload: Dict[str, Any],
    timeout: int = 180,
    bearer_token: str | None = None,
) -> Dict[str, Any]:
    r = requests.post(url, json=payload, timeout=timeout, headers=_auth_headers(bearer_token))
    r.raise_for_status()
    return r.json()


def safe_get(
    url: str,
    timeout: int = 60,
    params: Dict[str, Any] | None = None,
    bearer_token: str | None = None,
) -> Dict[str, Any]:
    r = requests.get(url, params=params, timeout=timeout, headers=_auth_headers(bearer_token))
    r.raise_for_status()
    return r.json()


def normalize_history_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(candidate)

    if out.get("score") is None:
        out["score"] = _safe_float(out.get("semantic_score", 0.0))

    if not isinstance(out.get("skills_match"), dict):
        out["skills_match"] = {
            "found": out.get("skills_found") or [],
            "missing": out.get("skills_missing") or [],
            "match_rate": _safe_float(out.get("skills_match_rate", 0.0)),
        }

    if not isinstance(out.get("experience"), dict):
        out["experience"] = {"years": _safe_float(out.get("experience_years", 0.0))}

    if not isinstance(out.get("contacts"), dict):
        email = out.get("email")
        phone = out.get("phone")
        out["contacts"] = {
            "email": email,
            "phone": phone,
            "emails": [email] if email else [],
            "phones": [phone] if phone else [],
            "urls": [],
        }

    if not isinstance(out.get("meta"), dict):
        out["meta"] = {}

    if not isinstance(out.get("breakdown"), dict):
        out["breakdown"] = {}

    return out


def normalize_history_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [normalize_history_candidate(c) for c in candidates]


def results_to_csv(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""

    buf = io.StringIO()
    fields = [
        "rank",
        "id",
        "candidate_name",
        "email",
        "phone",
        "final_score",
        "semantic_score",
        "skills_score",
        "experience_score",
        "status",
        "summary",
    ]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()

    for i, c in enumerate(results, 1):
        bd = c.get("breakdown") or {}
        contacts = c.get("contacts") or {}
        writer.writerow(
            {
                "rank": i,
                "id": c.get("id", ""),
                "candidate_name": c.get("candidate_name", ""),
                "email": contacts.get("email") or c.get("email", ""),
                "phone": contacts.get("phone") or c.get("phone", ""),
                "final_score": round(_safe_float(c.get("final_score", 0)), 2),
                "semantic_score": round(_safe_float(bd.get("semantic", 0)), 2),
                "skills_score": round(_safe_float(bd.get("skills", 0)), 2),
                "experience_score": round(_safe_float(bd.get("experience", 0)), 2),
                "status": c.get("status", ""),
                "summary": (c.get("summary") or "")[:220],
            }
        )
    return buf.getvalue()


def render_candidate_card(candidate: Dict[str, Any], rank: int, key_prefix: str = "") -> bool:
    cid = candidate.get("id") or str(rank)
    name = candidate.get("candidate_name")
    display_title = f"{name} ({cid})" if name else str(cid)
    final_score = _safe_float(candidate.get("final_score", 0.0))
    status = str(candidate.get("status") or "unknown")
    summary = candidate.get("summary") or "No summary available."

    contacts = candidate.get("contacts") or {}
    if not isinstance(contacts, dict):
        contacts = {}
    email = contacts.get("email") or candidate.get("email")
    phone = contacts.get("phone") or candidate.get("phone")
    urls = contacts.get("urls") or []
    meta = candidate.get("meta") or {}
    recent_companies = meta.get("recent_companies", []) if isinstance(meta, dict) else []

    bd = candidate.get("breakdown") or {}
    sem_score = _safe_float(bd.get("semantic", 0))
    skl_score = _safe_float(bd.get("skills", 0))
    exp_score = _safe_float(bd.get("experience", 0))

    title_col, chip_col, select_col = st.columns([7, 2, 1.5])
    with title_col:
        st.markdown(f"### #{rank} {display_title}")
        st.caption(summary[:180])
    with chip_col:
        st.markdown(_status_chip(status), unsafe_allow_html=True)
    with select_col:
        selected = st.checkbox("Compare", key=f"{key_prefix}_sel_{rank}_{cid}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final", f"{final_score:.1f}")
    m2.metric("Semantic", f"{sem_score:.1f}")
    m3.metric("Skills", f"{skl_score:.1f}")
    m4.metric("Experience", f"{exp_score:.1f}")

    with st.expander("Details", expanded=False):
        tab_overview, tab_signals, tab_analysis = st.tabs(["Overview", "Signals", "Analysis"])

        with tab_overview:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Name:** {name or '-'}")
                st.markdown(f"**Email:** {email or '-'}")
                st.markdown(f"**Phone:** {phone or '-'}")
            with c2:
                st.markdown(f"**Status:** {status}")
                st.markdown(f"**ID/File:** {cid}")
                if urls:
                    st.markdown(f"**Links:** {', '.join(urls[:4])}")
                else:
                    st.markdown("**Links:** -")

            if recent_companies:
                st.markdown(f"**Recent Companies:** {', '.join(recent_companies[:6])}")
            else:
                st.markdown("**Recent Companies:** -")

        with tab_signals:
            st.json(
                {
                    "score_breakdown": bd,
                    "semantic_score": candidate.get("score"),
                    "skills_match": candidate.get("skills_match"),
                    "experience": candidate.get("experience"),
                    "contacts": candidate.get("contacts"),
                }
            )

        with tab_analysis:
            st.json(
                {
                    "summary": candidate.get("summary"),
                    "pros": candidate.get("pros", []),
                    "cons": candidate.get("cons", []),
                    "interview_questions": candidate.get("interview_questions", []),
                    "evidence": candidate.get("evidence", {}),
                    "explanation": candidate.get("explanation", ""),
                }
            )

    st.divider()
    return selected


def render_comparison(candidates: List[Dict[str, Any]]) -> None:
    if len(candidates) < 2:
        st.info("Select at least 2 candidates to compare.")
        return

    if len(candidates) > 4:
        st.warning("Showing first 4 selected candidates for readability.")
        candidates = candidates[:4]

    cols = st.columns(len(candidates))
    for col, cand in zip(cols, candidates):
        bd = cand.get("breakdown") or {}
        cid = cand.get("id", "?")
        name = cand.get("candidate_name")
        contacts = cand.get("contacts") or {}
        email = contacts.get("email") or cand.get("email")
        phone = contacts.get("phone") or cand.get("phone")

        with col:
            st.markdown(f"### {name or cid}")
            st.markdown(_status_chip(str(cand.get("status") or "unknown")), unsafe_allow_html=True)
            st.metric("Final", f"{_safe_float(cand.get('final_score', 0)):.1f}")
            st.metric("Semantic", f"{_safe_float(bd.get('semantic', 0)):.1f}")
            st.metric("Skills", f"{_safe_float(bd.get('skills', 0)):.1f}")
            st.metric("Experience", f"{_safe_float(bd.get('experience', 0)):.1f}")
            st.caption(f"Email: {email or '-'}")
            st.caption(f"Phone: {phone or '-'}")


def _candidate_search_blob(candidate: Dict[str, Any]) -> str:
    contacts = candidate.get("contacts") or {}
    values = [
        candidate.get("id"),
        candidate.get("candidate_name"),
        candidate.get("summary"),
        candidate.get("status"),
        contacts.get("email") if isinstance(contacts, dict) else None,
        contacts.get("phone") if isinstance(contacts, dict) else None,
        candidate.get("email"),
        candidate.get("phone"),
    ]
    return " ".join([str(v).lower() for v in values if v is not None])


def filter_results(results: List[Dict[str, Any]], query: str, allowed_status: List[str], min_score: float) -> List[Dict[str, Any]]:
    q = (query or "").strip().lower()
    allowed = {s.lower() for s in allowed_status}
    out: List[Dict[str, Any]] = []

    for c in results:
        status = str(c.get("status") or "unknown").lower()
        score = _safe_float(c.get("final_score", 0.0))
        if status not in allowed:
            continue
        if score < min_score:
            continue
        if q and q not in _candidate_search_blob(c):
            continue
        out.append(c)

    out.sort(key=lambda x: _safe_float(x.get("final_score", 0.0)), reverse=True)
    return out


def render_results_panel(data: Dict[str, Any], max_show: int, show_raw: bool, key_prefix: str = "main") -> None:
    results = data.get("results") or data.get("ranked") or []
    session_id = data.get("session_id")

    st.success(f"Ranked {len(results)} candidate(s) | Session: {session_id or '-'}")
    if not results:
        st.info("No candidates to display.")
        return

    export_col1, export_col2, _ = st.columns([1, 1, 4])
    with export_col1:
        st.download_button(
            "Export CSV",
            data=results_to_csv(results),
            file_name="talent_scout_results.csv",
            mime="text/csv",
        )
    with export_col2:
        st.download_button(
            "Export JSON",
            data=json.dumps(data, indent=2, default=str),
            file_name="talent_scout_results.json",
            mime="application/json",
        )

    st.divider()
    f1, f2, f3 = st.columns([2.5, 2, 1.5])
    query = f1.text_input(
        "Search",
        key=f"{key_prefix}_search",
        placeholder="name, id, email, phone, summary",
    )
    status_options = sorted({str(c.get("status") or "unknown") for c in results})
    selected_status = f2.multiselect(
        "Status",
        options=status_options,
        default=status_options,
        key=f"{key_prefix}_status",
    )
    min_score = f3.slider(
        "Min score",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        key=f"{key_prefix}_min_score",
    )

    filtered = filter_results(results, query, selected_status or status_options, min_score)
    shown = filtered[: int(max_show)]
    st.caption(f"Showing {len(shown)} of {len(filtered)} filtered candidates.")

    selected_candidates: List[Dict[str, Any]] = []
    for i, cand in enumerate(shown, 1):
        if render_candidate_card(cand, i, key_prefix=key_prefix):
            selected_candidates.append(cand)

    if selected_candidates:
        st.subheader("Candidate Comparison")
        render_comparison(selected_candidates)

    if show_raw:
        with st.expander("Raw API Response", expanded=False):
            st.json(data)


def fetch_sessions(
    api_base: str,
    timeout_s: int,
    limit: int,
    bearer_token: str | None = None,
) -> List[Dict[str, Any]]:
    payload = safe_get(
        _build_url(api_base, "/api/v1/sessions"),
        timeout=timeout_s,
        params={"limit": int(limit)},
        bearer_token=bearer_token,
    )
    return payload.get("sessions") or []


def fetch_session_candidates(
    api_base: str,
    session_id: str,
    timeout_s: int,
    bearer_token: str | None = None,
) -> List[Dict[str, Any]]:
    payload = safe_get(
        _build_url(api_base, f"/api/v1/sessions/{session_id}"),
        timeout=timeout_s,
        bearer_token=bearer_token,
    )
    return payload.get("candidates") or []


st.set_page_config(page_title="Talent Scout", layout="wide", page_icon="TS")
inject_css()

st.markdown(
    """
<div class="app-hero">
  <h1>Talent Scout Screening</h1>
  <p>Run new screenings, compare candidates, and reopen prior sessions in one clean flow.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    api_base = API_BASE_DEFAULT
    run_endpoint = RUN_PATH_DEFAULT
    timeout_s = REQUEST_TIMEOUT_SECONDS
    api_token = st.text_input(
        "API Bearer Token (optional)",
        value="",
        type="password",
        help="Required for protected /run and /sessions endpoints.",
    )

    role_profile = st.selectbox(
        "Role Profile",
        ["custom", "fresh_grad", "junior", "senior", "lead", "manager"],
    )
    max_show = int(st.number_input("Max candidates shown", min_value=1, max_value=200, value=50, step=1))
    history_limit = int(st.number_input("History rows", min_value=10, max_value=500, value=200, step=10))
    show_raw = st.checkbox("Show raw payload", value=False)

tab_new, tab_history = st.tabs(["New Screening", "Prior Screenings"])

with tab_new:
    st.subheader("Job Setup")
    job_title = st.text_input("Job Title (optional)")
    job_description = st.text_area(
        "Job Description",
        height=180,
        placeholder="Paste the job description here",
    )

    st.subheader("Resumes")
    upload_tab, paste_tab = st.tabs(["Upload Files", "Paste Text"])
    uploaded_files = []
    resumes_raw = ""

    with upload_tab:
        uploaded_files = st.file_uploader(
            "Upload resume files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Each file represents one candidate resume.",
        )
        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) ready")

    with paste_tab:
        resumes_raw = st.text_area(
            f"Paste resumes (split with `{SEPARATOR}`)",
            height=260,
            placeholder=f"Resume 1\n{SEPARATOR}\nResume 2\n{SEPARATOR}\nResume 3",
        )

    run_clicked = st.button("Run Screening", type="primary", use_container_width=True)

    if run_clicked:
        jd = (job_description or "").strip()
        if not jd:
            st.error("Job description is required.")
            st.stop()

        resumes: List[Dict[str, str]] = []
        for uf in uploaded_files or []:
            try:
                text = extract_text(uf).strip()
                if text:
                    resumes.append({"id": uf.name, "resume_text": text})
            except Exception as exc:
                st.warning(f"Could not read {uf.name}: {exc}")

        for i, text in enumerate(parse_resumes(resumes_raw), 1):
            resumes.append({"id": f"Pasted Resume {i}", "resume_text": text})

        if not resumes:
            st.error("Upload at least one resume file or paste resume text.")
            st.stop()

        payload = {
            "job_title": (job_title or "").strip(),
            "job_description": jd,
            "role_profile": role_profile,
            "resumes": resumes,
            "model_version_id": None,
            "scoring_config": None,
        }

        try:
            with st.spinner("Running ranking pipeline..."):
                data = safe_post(
                    _build_url(api_base, run_endpoint),
                    payload,
                    timeout=timeout_s,
                    bearer_token=api_token,
                )
            st.session_state["results_data"] = data
            st.session_state["results_origin"] = "live"
            st.session_state.pop("history_sessions", None)
        except requests.exceptions.HTTPError as exc:
            msg = f"HTTP error: {exc}"
            try:
                msg += f"\n\nResponse:\n{exc.response.text}"
            except Exception:
                pass
            st.error(msg)
        except requests.exceptions.RequestException as exc:
            st.error(f"API call failed: {exc}")

    if "results_data" in st.session_state:
        origin = st.session_state.get("results_origin", "live")
        if origin == "history":
            st.caption("Showing a prior session in candidate view.")
        render_results_panel(
            st.session_state["results_data"],
            max_show=max_show,
            show_raw=show_raw,
            key_prefix="results_panel",
        )

with tab_history:
    st.subheader("Prior Sessions")

    refresh_col, info_col = st.columns([1, 4])
    refresh_clicked = refresh_col.button("Refresh History", use_container_width=True)
    info_col.caption("Load a prior session and optionally open it in the same candidate card view.")

    if refresh_clicked or "history_sessions" not in st.session_state:
        try:
            with st.spinner("Loading session history..."):
                st.session_state["history_sessions"] = fetch_sessions(
                    api_base,
                    timeout_s,
                    history_limit,
                    bearer_token=api_token,
                )
        except requests.exceptions.RequestException as exc:
            st.session_state["history_sessions"] = []
            st.warning(f"Could not load prior screenings: {exc}")

    sessions: List[Dict[str, Any]] = st.session_state.get("history_sessions", [])
    if not sessions:
        st.info("No prior screenings found.")
    else:
        st.caption(f"Loaded {len(sessions)} session(s).")
        st.dataframe(
            [
                {
                    "created_at": s.get("created_at"),
                    "job_title": s.get("job_title"),
                    "role_profile": s.get("role_profile"),
                    "total_candidates": s.get("total_candidates"),
                    "session_id": s.get("id"),
                }
                for s in sessions
            ],
            use_container_width=True,
            hide_index=True,
        )

        labels = [
            f"{s.get('created_at', '?')} | {s.get('job_title') or '(untitled)'} | {s.get('total_candidates', 0)} candidates"
            for s in sessions
        ]
        idx = st.selectbox(
            "Choose a prior session",
            options=list(range(len(sessions))),
            format_func=lambda i: labels[i],
        )
        selected_session = sessions[int(idx)]
        selected_id = selected_session.get("id")

        preview_col, open_col = st.columns(2)
        preview_clicked = preview_col.button("Load Selected Session", use_container_width=True)
        open_clicked = open_col.button("Open In Candidate View", use_container_width=True)

        if preview_clicked or open_clicked:
            if not selected_id:
                st.error("Selected session has no id.")
            else:
                try:
                    with st.spinner("Loading session candidates..."):
                        raw_candidates = fetch_session_candidates(
                            api_base,
                            str(selected_id),
                            timeout_s,
                            bearer_token=api_token,
                        )
                    normalized = normalize_history_candidates(raw_candidates)
                    st.session_state["history_preview"] = {
                        "session": selected_session,
                        "candidates": normalized,
                    }

                    if open_clicked:
                        st.session_state["results_data"] = {
                            "session_id": str(selected_id),
                            "results": normalized,
                        }
                        st.session_state["results_origin"] = "history"
                        st.rerun()
                except requests.exceptions.RequestException as exc:
                    st.error(f"Could not load selected session: {exc}")

    if "history_preview" in st.session_state:
        preview = st.session_state["history_preview"]
        session = preview.get("session") or {}
        candidates = preview.get("candidates") or []

        st.divider()
        st.markdown("### Session Preview")
        st.caption(
            f"Session: {session.get('id')} | Job: {session.get('job_title') or '(untitled)'} | Created: {session.get('created_at')}"
        )

        if not candidates:
            st.info("No candidates stored for this session.")
        else:
            st.dataframe(
                [
                    {
                        "rank": i,
                        "id": c.get("id"),
                        "name": c.get("candidate_name"),
                        "email": (c.get("contacts") or {}).get("email") or c.get("email"),
                        "phone": (c.get("contacts") or {}).get("phone") or c.get("phone"),
                        "final_score": c.get("final_score"),
                        "status": c.get("status"),
                        "summary": (c.get("summary") or "")[:120],
                    }
                    for i, c in enumerate(candidates, 1)
                ],
                use_container_width=True,
                hide_index=True,
            )

            if show_raw:
                with st.expander("Raw Session Candidates", expanded=False):
                    st.json(candidates)
