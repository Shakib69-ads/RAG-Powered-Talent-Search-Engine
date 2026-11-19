
import os
from datetime import datetime
from html import escape
from urllib.parse import quote

import streamlit as st
from dotenv import load_dotenv
from groq import RateLimitError

from src.storage.vector_store import create_vector_store
from src.retrieval.semantic_search import search_candidates
from src.evaluation.llm_evaluator import create_groq_client, evaluate_candidates

load_dotenv()

st.set_page_config(
    page_title="RAG Talent Search",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

llama_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">
<g fill="none" stroke="#102449" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round">
<path d="M12 37V17L8 5l12 9L29 5l9 12v20c0 5-6 7-13 7S12 42 12 37Z"/>
<path d="M12 25c-6-2-9 3-7 7 2 4 7 4 9 2M38 25c5-2 8 2 7 6-1 4-5 5-8 3"/>
<path d="M19 34q6 5 12 0"/><circle cx="19" cy="26" r="1.5" fill="#102449" stroke="none"/>
<circle cx="31" cy="26" r="1.5" fill="#102449" stroke="none"/>
</g></svg>"""
llama_icon = "data:image/svg+xml," + quote(llama_svg)

css = """
<style>
:root { --navy: #102449; --teal: #0087a5; }
.stApp { background: #f4f7fc; color: #102449; }
header[data-testid="stHeader"] { height: 0; background: transparent; }
[data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"] { display: none; }

[data-testid="stMainBlockContainer"] {
    max-width: none !important;
    padding: 0 18px 45px 24px !important;
}

.st-key-top_nav {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 9999;
    height: 76px;
    padding: 12px 28px 12px 42px;
    background: linear-gradient(110deg, #102449, #0d2042);
    box-shadow: 0 2px 8px #1024491a;
}

.st-key-top_nav [data-testid="stVerticalBlock"] { gap: 0; }

.st-key-top_nav button {
    background: transparent;
    color: white;
    border: 0;
    border-radius: 11px;
    min-height: 46px;
    font-size: 15px;
    font-weight: 650;
}

.st-key-top_nav button:hover {
    background: #264767;
    color: white;
    border: 0;
}

.st-key-top_nav button[kind="primary"] {
    background: linear-gradient(120deg, #078aa9, #007b99);
    color: white;
}

.brand {
    display: flex;
    align-items: center;
    color: white;
    white-space: nowrap;
}

.brand svg {
    flex-shrink: 0;
    margin-right: 12px;
}

.brand-name {
    font-size: 26px;
    font-weight: 760;
}

.brand-subtitle {
    font-size: 15px;
    font-weight: 400;
    color: #d4e0f1;
    border-left: 1px solid #557495;
    padding-left: 20px;
    margin-left: 23px;
}

.nav-space { height: 93px; }

section[data-testid="stSidebar"] {
    top: 76px;
    height: calc(100vh - 76px);
    width: 340px !important;
    min-width: 340px !important;
    background: #f8faff;
    border: 1px solid #d7e1ee;
    border-radius: 0 13px 13px 0;
}

section[data-testid="stSidebar"] > div {
    padding-top: 28px;
}

.side-heading {
    display: flex;
    gap: 13px;
    align-items: center;
}

.side-heading svg { flex-shrink: 0; }

.side-heading strong {
    font-size: 20px;
    color: #102449;
}

.side-subtitle {
    margin: 3px 0 23px 55px;
    font-size: 13px;
    color: #314e71;
}

.st-key-provider_groq button,
.st-key-provider_gemini button,
.st-key-provider_ollama button {
    position: relative;
    width: 100%;
    height: 72px;
    background: linear-gradient(110deg, #fff, #f7f9fd);
    border: 1px solid #cbd8eb;
    border-radius: 12px;
    padding: 9px 38px 26px 62px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    text-align: left !important;
    color: #102449 !important;
    font-size: 17px !important;
    font-weight: 750 !important;
    margin-bottom: 8px;
}

.st-key-provider_groq button > div,
.st-key-provider_gemini button > div,
.st-key-provider_ollama button > div {
    width: 100%;
    text-align: left !important;
}

.st-key-provider_groq button p,
.st-key-provider_gemini button p,
.st-key-provider_ollama button p {
    text-align: left !important;
    margin: 0;
    font-weight: 750;
}

.st-key-provider_groq button:hover,
.st-key-provider_gemini button:hover,
.st-key-provider_ollama button:hover {
    border-color: #008aa6;
    background: #edf8fc;
}

.st-key-provider_groq button::before,
.st-key-provider_gemini button::before,
.st-key-provider_ollama button::before {
    position: absolute;
    left: 13px;
    top: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 38px;
}

.st-key-provider_groq button::before {
    content: "∞";
    font-family: Arial, sans-serif;
    font-size: 45px;
    font-weight: 800;
    color: #f34128;
}

.st-key-provider_gemini button::before {
    content: "G";
    font-family: Arial, sans-serif;
    font-size: 36px;
    font-weight: 850;
    background: conic-gradient(
        #4285f4 0 27%,
        #34a853 27% 48%,
        #fbbc05 48% 67%,
        #ea4335 67% 83%,
        #4285f4 83%
    );
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.st-key-provider_ollama button::before {
    content: "";
    background: url("__LLAMA_ICON__") center / contain no-repeat;
}

.st-key-provider_groq button::after,
.st-key-provider_gemini button::after,
.st-key-provider_ollama button::after {
    position: absolute;
    left: 62px;
    bottom: 12px;
    color: #344e70;
    font-size: 12px;
    font-weight: 400;
}

.st-key-provider_groq button::after {
    content: "Fast and cost-efficient";
}

.st-key-provider_gemini button::after {
    content: "Google's multimodal model";
}

.st-key-provider_ollama button::after {
    content: "Run models on your machine";
}

.st-key-provider_groq button,
.st-key-provider_gemini button,
.st-key-provider_ollama button {
    background-image:
        radial-gradient(
            circle at calc(100% - 23px) 50%,
            white 0 8px,
            #7d96b8 9px 10px,
            transparent 11px
        ),
        linear-gradient(110deg, #fff, #f7f9fd);
}

__SELECTED_PROVIDER__

.sidebar-divider {
    border-top: 1px solid #d9e3ef;
    margin: 22px 0 18px;
}

.sidebar-label {
    font-size: 18px;
    font-weight: 740;
    color: #102449;
    margin: 0 0 9px;
}

.provider-status {
    border-radius: 11px;
    padding: 15px 17px;
    background: #ddf7ea;
    color: #08704b;
    margin: 22px 0 18px;
    font-size: 14px;
}

.provider-status strong {
    display: block;
    font-size: 17px;
    margin-top: 4px;
}

.provider-status.offline {
    background: #fff1df;
    color: #895c1a;
}

.workflow {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    font-size: 14px;
    color: #263e60;
    margin: 17px 0;
}

.workflow b {
    background: #058ca6;
    color: white;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 25px;
    height: 25px;
    flex-shrink: 0;
}

.page-heading {
    display: flex;
    gap: 17px;
    align-items: center;
    margin-bottom: 19px;
}

.page-heading svg { flex-shrink: 0; }

.page-title {
    font-size: 29px;
    font-weight: 790;
    color: #102449;
    line-height: 1.25;
}

.page-subtitle {
    font-size: 15.5px;
    color: #2c476c;
    margin-top: 3px;
}

.st-key-search_panel {
    background: #fff;
    border: 1px solid #dce6f2;
    border-radius: 14px;
    box-shadow: 0 4px 14px #172b4812;
    padding: 18px 20px 16px;
}

.search-label {
    display: flex;
    align-items: center;
    gap: 12px;
    color: #102449;
    font-size: 17px;
    font-weight: 760;
    margin-bottom: 11px;
}

.st-key-search_panel textarea {
    border: 1px solid #b8cbe5;
    background: #f8faff;
    border-radius: 11px;
    font-size: 16px;
    color: #102449;
}

.st-key-search_panel textarea:focus {
    border-color: #078aa8;
    box-shadow: 0 0 0 1px #078aa830;
}

.char-count {
    color: #31517b;
    font-size: 13px;
    text-align: right;
    margin-top: 3px;
}

.st-key-search_submit button {
    background: linear-gradient(110deg, #0086a5, #007c99) !important;
    border: 0 !important;
    border-radius: 11px !important;
    color: white !important;
    font-size: 15px !important;
    font-weight: 750 !important;
    min-height: 55px !important;
}

.st-key-search_submit button:hover {
    background: #006f8c !important;
}

.results-heading {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 23px 0 17px;
}

.results-heading strong {
    font-size: 24px;
    font-weight: 780;
    color: #102449;
}

.results-count {
    margin-left: auto;
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #e5eef9;
    border-radius: 12px;
    color: #254a76;
    padding: 8px 13px;
    font-size: 14px;
}

.candidate-card {
    background: white;
    border: 1px solid #dce5f0;
    border-radius: 14px;
    padding: 17px;
    box-shadow: 0 3px 13px #152b4510;
    min-height: 538px;
    overflow-wrap: anywhere;
}

.candidate-head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 8px;
}

.candidate-person {
    display: flex;
    align-items: center;
    gap: 12px;
    min-width: 0;
}

.avatar {
    width: 50px;
    height: 50px;
    min-width: 50px;
    border-radius: 50%;
    background: linear-gradient(#dcecff, #edf5ff);
    color: #3984dc;
    display: flex;
    align-items: center;
    justify-content: center;
}

.candidate-id {
    font-size: 15px;
    font-weight: 790;
    color: #102449;
}

.candidate-role {
    font-size: 13px;
    color: #1c3556;
    margin-top: 4px;
}

.score {
    border-radius: 11px;
    padding: 8px 11px;
    white-space: nowrap;
    font-size: 16px;
    font-weight: 800;
}

.score-high {
    background: #c7f5df;
    color: #006b48;
}

.score-mid {
    background: #fff0c0;
    color: #805b00;
}

.score-low {
    background: #ffdde2;
    color: #9a2939;
}

.recommendation {
    display: inline-block;
    margin: 9px 0 2px 64px;
    padding: 5px 13px;
    background: #c9f7e2;
    color: #006a47;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 750;
}

.recommendation.weak {
    background: #fff0d8;
    color: #8a5d14;
}

.card-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    font-weight: 760;
    color: #102449;
    margin: 14px 0 7px;
}

.card-label svg { flex-shrink: 0; }

.chip {
    display: inline-block;
    border-radius: 16px;
    padding: 5px 12px;
    margin: 2px 4px 3px 0;
    font-size: 12.5px;
}

.chip-match {
    background: #dfedff;
    color: #17385e;
}

.chip-gap {
    background: #ffe0e4;
    color: #862838;
}

.summary {
    background: #f0f5fb;
    border-radius: 9px;
    padding: 10px 13px;
    color: #132b4a;
    font-size: 13.5px;
    line-height: 1.5;
}

.evidence {
    color: #1b3455;
    font-size: 12.5px;
    line-height: 1.55;
    padding-left: 22px;
    margin: 3px 0 0;
}

.evidence li { margin-bottom: 3px; }

.footnote {
    font-size: 12px;
    color: #70819a;
    margin-top: 14px;
}

@media (max-width: 1150px) {
    .brand-subtitle { display: none; }
    .candidate-card { min-height: 0; }
}
</style>
"""

if "page" not in st.session_state:
    st.session_state.page = "Search"

if "provider" not in st.session_state:
    st.session_state.provider = "Groq"

if "history" not in st.session_state:
    st.session_state.history = []

selected_provider = st.session_state.provider.lower()

selected_css = f"""
.st-key-provider_{selected_provider} button {{
    border: 1px solid #008ca9 !important;
    background-image:
        radial-gradient(
            circle at calc(100% - 23px) 50%,
            #00a96b 0 5px,
            white 6px 8px,
            #00a96b 9px 11px,
            transparent 12px
        ),
        linear-gradient(110deg, #e7f7fb, #effaff) !important;
}}
"""

st.html(
    css.replace("__LLAMA_ICON__", llama_icon)
       .replace("__SELECTED_PROVIDER__", selected_css)
)


def svg_icon(name, color="#1f6caa", size=25):
    paths = {
        "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><path d="M12 12l8-8m-4 0h4v4"/>',
        "layers": '<path d="m12 2 10 6-10 6L2 8l10-6Zm-10 11 10 6 10-6M2 18l10 5 10-5"/>',
        "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/>',
        "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M8 13h8M8 17h8"/>',
        "alert": '<path d="m12 3 10 18H2L12 3Z"/><path d="M12 9v5m0 3h.01"/>',
        "list": '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
        "info": '<circle cx="12" cy="12" r="10"/><path d="M12 11v6M12 7h.01"/>',
        "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5M3 12c0 1.7 4 3 9 3s9-1.3 9-3"/>',
    }

    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">'
        f'{paths[name]}</svg>'
    )


def choose_provider(name):
    st.session_state.provider = name


def navigate(page):
    st.session_state.page = page


@st.cache_resource(show_spinner=False)
def load_store():
    return create_vector_store()


def skill_chips(items, kind):
    if not items:
        return '<span class="chip chip-gap">Not evidenced</span>'

    return "".join(
        f'<span class="chip chip-{kind}">{escape(str(item))}</span>'
        for item in items
    )


def candidate_card(candidate):
    score = max(0, min(100, int(candidate.get("match_score", 0))))

    score_style = (
        "score-high" if score >= 75
        else "score-mid" if score >= 50
        else "score-low"
    )

    recommendation = str(candidate.get("recommendation", "Not available"))
    recommendation_style = "weak" if recommendation == "Weak Match" else ""

    evidence = "".join(
        f"<li>{escape(str(item))}</li>"
        for item in candidate.get("evidence", [])[:2]
    ) or "<li>No supporting evidence provided.</li>"


    missing_skills = [
        skill for skill in (candidate.get("missing_skills") or [])
        if str(skill).strip().lower() not in {"", "not evidenced", "none", "n/a"}
    ]

    missing_html = (
        skill_chips(missing_skills, "gap")
        if missing_skills
        else skill_chips(["No missing required skills"], "match")
    )

    st.html(f"""
    <div class="candidate-card">
        <div class="candidate-head">
            <div class="candidate-person">
                <div class="avatar">
                    <svg width="35" height="35" viewBox="0 0 40 40" fill="#458be0">
                        <circle cx="20" cy="12" r="8"/>
                        <path d="M5 34c0-10 7-15 15-15s15 5 15 15v2H5z"/>
                    </svg>
                </div>
                <div>
                    <div class="candidate-id">
                        Resume ID: {escape(str(candidate['resume_id']))}
                    </div>
                    <div class="candidate-role">
                        {escape(str(candidate.get('job_title') or 'Not specified'))}
                    </div>
                </div>
            </div>
            <span class="score {score_style}">{score} / 100</span>
        </div>

        <div class="recommendation {recommendation_style}">
            👍 {escape(recommendation)}
        </div>

        <div class="card-label">
            {svg_icon('target', '#1477af', 19)}
            Matching Skills
        </div>
        <div>{skill_chips(candidate.get('matching_skills', []), 'match')}</div>

        <div class="card-label">
            {svg_icon('alert', '#dc382c', 19)}
            Missing Skills
        </div>
        <div>{missing_html}</div>

        <div class="card-label">
            {svg_icon('file', '#2779b4', 19)}
            Summary
        </div>
        <div class="summary">
            {escape(str(candidate.get('summary') or 'No summary available.'))}
        </div>

        <div class="card-label">
            {svg_icon('list', '#2779b4', 19)}
            Evidence
        </div>
        <ul class="evidence">{evidence}</ul>
    </div>
    """)


with st.container(key="top_nav"):
    brand_col, nav_col = st.columns(
        [1.85, 1],
        vertical_alignment="center"
    )

    with brand_col:
        st.html("""
        <div class="brand">
            <svg width="50" height="46" viewBox="0 0 50 46" fill="none">
                <circle cx="24" cy="11" r="7"
                    stroke="#65d5ec" stroke-width="4"/>
                <path d="M8 38v-5c0-8 7-13 16-13"
                    stroke="#65d5ec" stroke-width="4"
                    stroke-linecap="round"/>
                <path d="M11 11H5M7 7v8"
                    stroke="#2089b9" stroke-width="3"
                    stroke-linecap="round"/>
                <circle cx="35" cy="32" r="8"
                    stroke="white" stroke-width="3"/>
                <path d="m41 38 6 6"
                    stroke="white" stroke-width="3"
                    stroke-linecap="round"/>
            </svg>
            <span class="brand-name">RAG Talent Search</span>
            <span class="brand-subtitle">AI-Powered Candidate Discovery</span>
        </div>
        """)

    with nav_col:
        for column, page, icon in zip(
            st.columns(4, gap="small"),
            ["Search", "History", "Settings", "About"],
            [
                ":material/search:",
                ":material/history:",
                ":material/settings:",
                ":material/info:"
            ],
        ):
            with column:
                st.button(
                    page,
                    icon=icon,
                    key=f"nav_{page.lower()}",
                    type=(
                        "primary"
                        if st.session_state.page == page
                        else "secondary"
                    ),
                    use_container_width=True,
                    on_click=navigate,
                    args=(page,),
                )


st.html('<div class="nav-space"></div>')


with st.sidebar:
    st.html(f"""
        <div class="side-heading">
            {svg_icon('layers', '#356da8', 40)}
            <strong>LLM Provider</strong>
        </div>
        <div class="side-subtitle">
            Select a model for candidate evaluation
        </div>
    """)

    for provider in ["Groq", "Gemini", "Ollama"]:
        with st.container(key=f"provider_{provider.lower()}"):
            st.button(
                "Ollama (Local)" if provider == "Ollama" else provider,
                key=f"choose_{provider.lower()}",
                use_container_width=True,
                on_click=choose_provider,
                args=(provider,),
            )

    st.html(
        '<div class="sidebar-divider"></div>'
        '<div class="sidebar-label">Model</div>'
    )

    if st.session_state.provider == "Groq":
        model = st.selectbox(
            "Model",
            [os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")],
            label_visibility="collapsed",
            key="groq_model",
        )

        configured = bool(os.getenv("GROQ_API_KEY"))

        status = (
            '<span>● &nbsp;Active Provider</span><strong>Groq</strong>'
            if configured
            else "Groq API key missing"
        )

    else:
        model = st.selectbox(
            "Model",
            ["Not configured"],
            label_visibility="collapsed",
            key="other_model",
        )

        configured = False
        status = f"{escape(st.session_state.provider)} is not connected yet"

    st.html(
        f'<div class="provider-status {"" if configured else "offline"}">'
        f'{status}</div>'
    )

    st.html(f"""
        <div class="sidebar-divider"></div>
        <div class="side-heading">
            {svg_icon('info', '#356da8', 29)}
            <strong>How it works</strong>
        </div>
        <div class="workflow">
            <b>✓</b><span>Search relevant resumes (RAG)</span>
        </div>
        <div class="workflow">
            <b>✓</b><span>Evaluate with selected LLM</span>
        </div>
        <div class="workflow">
            <b>✓</b><span>Get Top 3 candidates with explanation</span>
        </div>
    """)


if st.session_state.page == "Search":
    st.html(f"""
        <div class="page-heading">
            {svg_icon('target', '#31699d', 46)}
            <div>
                <div class="page-title">Find the Best Candidates</div>
                <div class="page-subtitle">
                    Enter a job description to search and evaluate the
                    most relevant candidates from your resume database.
                </div>
            </div>
        </div>
    """)

    with st.container(key="search_panel"):
        st.html(
            f'<div class="search-label">'
            f'{svg_icon("file", "#2877b1", 23)}'
            f'Job Description</div>'
        )

        text_col, button_col = st.columns(
            [4.2, 1.2],
            vertical_alignment="center"
        )

        with text_col:
            query = st.text_area(
                "Job Description",
                value="Python developer with Machine Learning and NLP experience",
                label_visibility="collapsed",
                height=100,
                max_chars=2000,
                key="job_description",
            )

            st.html(
                f'<div class="char-count">{len(query)} / 2000</div>'
            )

        with button_col:
            with st.container(key="search_submit"):
                submitted = st.button(
                    "Search Candidates",
                    icon=":material/search:",
                    type="primary",
                    use_container_width=True,
                )

    if submitted:
        job_description = query.strip()

        if not job_description:
            st.warning("Enter a job description first.")

        elif st.session_state.provider != "Groq":
            st.warning(
                f"{st.session_state.provider} is not connected yet. "
                "Select Groq for now."
            )

        elif not configured:
            st.error("GROQ_API_KEY is missing from your .env file.")

        else:
            st.session_state.pop("results", None)

            try:
                with st.spinner("Searching resumes and evaluating candidates..."):
                    store = load_store()

                    candidates = search_candidates(
                        query=job_description,
                        store=store,
                        k=20,
                        top_n=3,
                    )

                    results = (
                        evaluate_candidates(
                            client=create_groq_client(),
                            store=store,
                            job_description=job_description,
                            candidates=candidates,
                        )
                        if candidates else []
                    )

                st.session_state.results = results
                st.session_state.last_query = job_description

                st.session_state.history.insert(0, {
                    "query": job_description,
                    "results": results,
                    "time": datetime.now().strftime("%d %b %Y, %H:%M"),
                })

            except RateLimitError as error:
                print(f"Groq rate limit: {error}")
                st.error(
                    "Groq rate limit reached. "
                    "Check the terminal for details."
                )

            except Exception as error:
                print(f"Search error: {type(error).__name__}: {error}")
                st.error(
                    "Search failed. Check the terminal for details."
                )

    if "results" in st.session_state:
        results = st.session_state.results

        st.html(f"""
            <div class="results-heading">
                {svg_icon('users', '#0988a5', 35)}
                <strong>Top 3 Candidates</strong>
                <span class="results-count">
                    {svg_icon('database', '#315f93', 17)}
                    Found {len(results)} candidates
                </span>
            </div>
        """)

        if not results:
            st.info("No candidates were retrieved for this job description.")

        else:
            for column, candidate in zip(
                st.columns(3, gap="medium"),
                results
            ):
                with column:
                    candidate_card(candidate)

            st.html(
                '<div class="footnote">'
                'AI evaluations support recruiters; they are not hiring '
                'decisions. Missing CV evidence does not prove a skill '
                'is absent.'
                '</div>'
            )


elif st.session_state.page == "History":
    st.html('<div class="page-title">Search History</div>')

    st.caption("Searches are saved only for this Streamlit session.")

    if not st.session_state.history:
        st.info("No searches yet.")

    for index, item in enumerate(st.session_state.history):
        with st.container(border=True):
            st.write(f"**{item['query']}**")

            st.caption(
                f"{item['time']} · {len(item['results'])} candidates"
            )

            if st.button("View Results", key=f"history_{index}"):
                st.session_state.results = item["results"]
                st.session_state.page = "Search"
                st.rerun()


elif st.session_state.page == "Settings":
    st.html('<div class="page-title">Settings</div>')

    st.write(f"Selected provider: **{st.session_state.provider}**")
    st.write(f"Model: **{model}**")

    st.info(
        "Gemini, Ollama, and automatic fallback "
        "are not integrated yet."
    )


else:
    st.html('<div class="page-title">About</div>')

    st.write(
        "RAG Talent Search retrieves the top three resumes through "
        "semantic search and uses an LLM to explain their job fit "
        "with CV evidence."
    )

    st.write(
        "**Pipeline:** Job Description → Semantic Search → "
        "Top 3 Resumes → LLM Explanation"
    )
