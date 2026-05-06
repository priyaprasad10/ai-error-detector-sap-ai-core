# app.py — AI Error Detective Streamlit UI
from database import (save_history, get_history, init_db, mark_resolved,
                       create_team, join_team, leave_team,
                       get_user_teams, get_team_members,
                       get_team_resolved_errors, get_similar_team_error,
                       save_match_feedback, get_similar_personal_error,
                       get_user_trends)
from auth import register_user, login_user
import streamlit as st
from datetime import datetime
import hashlib
from backend import (
    analyze_error,
    get_quick_fix,
    chat_about_error,
    extract_text_from_image,
    detect_platform_mismatch,
    get_severity_color,
    get_severity_emoji,
    get_platform_suggestions,
    format_download_report,
    get_embedding,
    cosine_similarity,
    warm_up_aicore,
)

init_db()

st.set_page_config(
    page_title="AI Error Detective",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading semantic model (first run only)...")
def load_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()

@st.cache_resource(show_spinner=False)
def _init_aicore():
    warm_up_aicore()

_init_aicore()

st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e2e8f0; }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7B2FFF, #00D4FF) !important;
        color: #ffffff !important; border: none !important;
        border-radius: 10px !important; font-weight: 700 !important;
        font-size: 15px !important; padding: 12px 24px !important;
        box-shadow: 0 4px 15px rgba(123,47,255,0.4) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #9B4FFF, #00E5FF) !important;
        transform: translateY(-2px) !important;
    }
    .stButton > button[kind="secondary"] {
        background: #1a1d27 !important; color: #00D4FF !important;
        border: 1.5px solid #00D4FF !important; border-radius: 10px !important;
        font-weight: 600 !important; transition: all 0.3s ease !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: rgba(0,212,255,0.15) !important;
    }
    .stButton > button {
        background: #1a1d27 !important; color: #e2e8f0 !important;
        border: 1.5px solid #2d3048 !important; border-radius: 10px !important;
        font-weight: 600 !important; transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background: #2d3048 !important; border-color: #7B2FFF !important;
    }
    .stButton > button:disabled {
        opacity: 0.4 !important; cursor: not-allowed !important;
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #00B050, #00D480) !important;
        color: #ffffff !important; border: none !important;
        border-radius: 10px !important; font-weight: 700 !important;
    }
    .stChatInput textarea, .stTextArea textarea {
        background: #1a1d27 !important; color: #e2e8f0 !important;
        border: 1.5px solid #2d3048 !important; border-radius: 10px !important;
    }
    .stSelectbox > div > div {
        background: #1a1d27 !important; color: #e2e8f0 !important;
        border: 1.5px solid #2d3048 !important; border-radius: 10px !important;
    }
    .stFileUploader > div {
        background: #1a1d27 !important; border: 1.5px dashed #2d3048 !important;
        border-radius: 10px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1d27 !important; border-radius: 10px !important;
        padding: 4px !important; gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important; color: #888 !important;
        border-radius: 8px !important; font-weight: 600 !important;
        padding: 8px 16px !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg,#7B2FFF,#00D4FF) !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] {
        background: #0d1017 !important; border-right: 1px solid #1a1d27 !important;
    }
    .stAlert { border-radius: 10px !important; border: none !important; }
    .main-title {
        font-size: 3rem; font-weight: 900;
        background: linear-gradient(90deg, #00D4FF, #7B2FFF, #FF6B6B);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; text-align: center; padding: 10px 0;
    }
    .subtitle { text-align: center; color: #888; font-size: 1.1rem; margin-bottom: 20px; }
    .stat-box { background: #1a1d27; border: 1px solid #2d3048; border-radius: 10px; padding: 15px; text-align: center; }
    .about-card { background: linear-gradient(135deg, #1a1d27, #0d1117); border: 1px solid #7B2FFF; border-radius: 16px; padding: 28px; margin: 10px 0; }
    .chat-user { background: #1e3a5f; border-radius: 10px; padding: 12px 16px; margin: 8px 0; border-left: 4px solid #00D4FF; }
    .chat-bot { background: #1a2332; border-radius: 10px; padding: 12px 16px; margin: 8px 0; border-left: 4px solid #7B2FFF; }
    .custom-divider { height: 2px; background: linear-gradient(90deg, #00D4FF, #7B2FFF, #FF6B6B); border: none; margin: 20px 0; }
    .resolved-banner {
        background: linear-gradient(135deg, #00B05022, #00D48022);
        border: 1px solid #00B050; border-radius: 10px;
        padding: 12px 16px; margin: 10px 0;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-baseweb="checkbox"] input:checked ~ span,
    [data-baseweb="checkbox"] span[data-checked="true"] {
        background-color: #00B050 !important;
        border-color: #00B050 !important;
        outline-color: #00B050 !important;
    }
    [data-baseweb="checkbox"] span {
        border-color: #00B050 !important;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════
defaults = {
    "current_result":        None,
    "current_db_id":         None,   # DB row id of current analyzed error
    "current_is_resolved":   False,  # whether current error is marked resolved
    "chat_history":          [],
    "session_history":       [],
    "analyzed_fingerprints": [],
    "show_about":            False,
    "screenshot_extracted":  None,
    "logged_in":             False,
    "user_id":               None,
    "username":              None,
    "auth_mode":             "login",
    "confirm_clear":         False,
    "similar_team_resolved": False,
    "current_match_id":      None,
    "match_feedback_given":  False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ══════════════════════════════════════════════════════════
# LOGIN / REGISTER PAGE
# ══════════════════════════════════════════════════════════
if not st.session_state.logged_in:

    st.markdown("<div class='main-title'>🔍 AI Error Detective</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Slash Debugging Time for BTP & ABAP Developers</div>", unsafe_allow_html=True)
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        toggle_l, toggle_r = st.columns(2)
        with toggle_l:
            if st.button("🔐 Login", use_container_width=True,
                         type="primary" if st.session_state.auth_mode == "login" else "secondary"):
                st.session_state.auth_mode = "login"
                st.rerun()
        with toggle_r:
            if st.button("📝 Register", use_container_width=True,
                         type="primary" if st.session_state.auth_mode == "register" else "secondary"):
                st.session_state.auth_mode = "register"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.auth_mode == "login":
            st.markdown("<div style='text-align:center;font-size:1.4rem;font-weight:800;color:#00D4FF;margin-bottom:20px'>Welcome Back 👋</div>", unsafe_allow_html=True)
            login_username = st.text_input("👤 Username", placeholder="Enter your username", key="login_user")
            login_password = st.text_input("🔒 Password", type="password", placeholder="Enter your password", key="login_pass")
            if st.button("🚀 Login", type="primary", use_container_width=True, key="login_submit"):
                if login_username and login_password:
                    result = login_user(login_username, login_password)
                    if result["success"]:
                        st.session_state.logged_in = True
                        st.session_state.user_id   = result["user_id"]
                        st.session_state.username  = result["username"]
                        st.rerun()
                    else:
                        st.error(f"❌ {result['message']}")
                else:
                    st.warning("⚠️ Please enter both username and password.")
        else:
            st.markdown("<div style='text-align:center;font-size:1.4rem;font-weight:800;color:#7B2FFF;margin-bottom:20px'>Create Account 🆕</div>", unsafe_allow_html=True)
            reg_username  = st.text_input("👤 Choose Username", placeholder="e.g. john_dev", key="reg_user")
            reg_password  = st.text_input("🔒 Choose Password", type="password", placeholder="Min 6 characters", key="reg_pass")
            reg_password2 = st.text_input("🔒 Confirm Password", type="password", placeholder="Repeat password", key="reg_pass2")
            if st.button("✅ Create Account", type="primary", use_container_width=True, key="reg_submit"):
                if not reg_username or not reg_password:
                    st.warning("⚠️ Please fill all fields.")
                elif reg_password != reg_password2:
                    st.error("❌ Passwords do not match.")
                else:
                    result = register_user(reg_username, reg_password)
                    if result["success"]:
                        st.success(f"✅ {result['message']} Please login now.")
                        st.session_state.auth_mode = "login"
                        st.rerun()
                    else:
                        st.error(f"❌ {result['message']}")
    st.stop()


# ══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════
def error_fingerprint(error_text: str, platform: str) -> str:
    normalized = " ".join((error_text or "").strip().lower().split())
    return hashlib.sha256(f"{platform}::{normalized}".encode("utf-8")).hexdigest()


def build_report(result: dict, chat_history: list, username: str) -> str:
    lines = [
        "=" * 60,
        "AI ERROR DETECTIVE — RESOLUTION REPORT",
        "=" * 60,
        f"Analyzed by : {username}",
        f"Date & Time : {datetime.now().strftime('%d %b %Y, %H:%M:%S')}",
        f"Platform    : {result.get('error_type', 'N/A')}",
        f"Severity    : {result.get('severity',   'N/A')}",
        "=" * 60,
        "",
        "ERROR SUBMITTED:",
        "-" * 40,
        result.get("error_text", ""),
        "",
        "AI ANALYSIS:",
        "-" * 40,
        result.get("analysis", ""),
        "",
    ]
    if chat_history:
        lines += ["FOLLOW-UP Q&A:", "-" * 40]
        for msg in chat_history:
            role = "Developer" if msg["role"] == "user" else "AI Detective"
            lines += [f"{role}: {msg['content']}", ""]
    lines += ["=" * 60, "Generated by AI Error Detective", "=" * 60]
    return "\n".join(lines)


def copy_button(text: str, key: str):
    """Copy button — stores text in session state, then JS copies it."""
    # Store the text to copy in session state so JS can pick it up
    st.session_state[f"_copy_text_{key}"] = text

    if st.button("📋 Copy Analysis", key=f"copy_btn_{key}", use_container_width=True):
        # Inject JS that reads from a hidden element and copies to clipboard
        escaped = text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        js = f"""
        <script>
        (function() {{
            const text = `{escaped}`;
            if (navigator.clipboard && window.isSecureContext) {{
                navigator.clipboard.writeText(text).then(function() {{
                    const btn = window.parent.document.querySelector('[data-testid="baseButton-secondary"] p');
                    if (btn) {{ btn.innerText = '✅ Copied!'; setTimeout(() => btn.innerText = '📋 Copy Analysis', 2000); }}
                }});
            }} else {{
                const el = document.createElement('textarea');
                el.value = text;
                el.style.position = 'fixed';
                el.style.opacity = '0';
                document.body.appendChild(el);
                el.focus(); el.select();
                document.execCommand('copy');
                document.body.removeChild(el);
            }}
        }})();
        </script>
        """
        st.markdown(js, unsafe_allow_html=True)
        st.toast("✅ Analysis copied to clipboard!", icon="📋")


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0'>
        <div style='font-size:3rem'>🔍</div>
        <div style='font-size:1.3rem; font-weight:800;
             background:linear-gradient(90deg,#00D4FF,#7B2FFF);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            AI Error Detective
        </div>
        <div style='color:#888; font-size:0.8rem'>Powered by SAP AI Core · GPT-5</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"<div style='background:#1a1d27;border:1px solid #7B2FFF44;"
        f"border-radius:10px;padding:10px;text-align:center;margin:8px 0'>"
        f"<span style='color:#888;font-size:0.8rem'>Logged in as</span><br>"
        f"<b style='color:#00D4FF;font-size:1rem'>👤 {st.session_state.username}</b>"
        f"</div>",
        unsafe_allow_html=True
    )
    if st.button("🚪 Logout", use_container_width=True, key="logout_btn"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Error Platform")
    st.markdown("<div style='color:#aaa;font-size:0.8rem;margin-bottom:6px;'>Select your SAP platform — AI will only answer questions related to this platform.</div>", unsafe_allow_html=True)

    PLATFORMS = [
        "SAP BTP", "CAP (Cloud Application Programming)", "ABAP Cloud",
        "ABAP On-Premise", "SAP Fiori / UI5", "SAP S/4HANA",
        "SAP Integration Suite", "SAP HANA", "SAP Build Apps", "Other SAP",
    ]
    error_type = st.selectbox("Select platform:", PLATFORMS, label_visibility="collapsed")

    st.markdown(
        f"<div style='background:linear-gradient(135deg,#7B2FFF22,#00D4FF22);"
        f"border:1px solid #7B2FFF55;border-radius:8px;padding:8px 12px;"
        f"margin:6px 0;font-size:0.82rem;color:#00D4FF;text-align:center'>"
        f"🎯 Mode: <b>{error_type}</b><br>"
        f"<span style='color:#666;font-size:0.75rem'>Chat is locked to this platform</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### 📊 Session Stats")

    session_resolved = sum(1 for h in st.session_state.session_history if h.get("is_resolved", False))
    st.markdown(f"""<div class='stat-box'>
        <div style='font-size:2rem;font-weight:800;color:#00B050'>{session_resolved}</div>
        <div style='color:#888;font-size:0.8rem'>Errors Resolved This Session</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.session_history:
        st.markdown("### 🕐 Recent Errors")
        for item in reversed(st.session_state.session_history[-5:]):
            emoji = get_severity_emoji(item["severity"])
            resolved_tag = " ✅" if item.get("is_resolved", False) else ""
            with st.expander(f"{emoji} {item['type']} — {item['time']}{resolved_tag}", expanded=False):
                st.caption(item["preview"])

    st.markdown("---")

    if not st.session_state.confirm_clear:
        if st.button("🗑️ Clear Session", use_container_width=True):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.warning("⚠️ Are you sure? This will clear all current session data.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Yes, Clear", use_container_width=True, type="primary"):
                for key in ["current_result", "current_db_id", "current_is_resolved",
                            "chat_history", "session_history",
                            "analyzed_fingerprints", "screenshot_extracted",
                            "confirm_clear"]:
                    val = st.session_state.get(key)
                    st.session_state[key] = (
                        [] if isinstance(val, list)
                        else (0 if isinstance(val, int)
                        else (False if isinstance(val, bool) else None))
                    )
                st.rerun()
        with c2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()

    if st.button("ℹ️ About This App", use_container_width=True, type="secondary"):
        st.session_state.show_about = not st.session_state.show_about


# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════
st.markdown("<div class='main-title'>🔍 AI Error Detective</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Slash Debugging Time for BTP & ABAP Developers · Powered by Free AI · Platform-Aware Intelligence</div>",
    unsafe_allow_html=True
)
st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

if st.session_state.show_about:
    st.markdown("""<div class='about-card'>
        <h2 style='color:#00D4FF;margin-top:0'>🎯 What is AI Error Detective?</h2>
        <p style='color:#ccc;font-size:1rem;line-height:1.7'>
            AI Error Detective is an intelligent debugging co-pilot built specifically for SAP developers.
            It uses state-of-the-art LLMs to instantly analyze cryptic SAP errors and provide clear,
            actionable fixes — saving hours of manual investigation.
        </p>
        <h3 style='color:#7B2FFF'>🎯 Platform-Aware Intelligence</h3>
        <p style='color:#aaa'>Select your platform and the AI locks into that ecosystem.
        Every analysis and follow-up is specific to your selected platform.</p>
        <div style='text-align:center;margin-top:20px;padding:15px;background:#0d1117;border-radius:10px'>
            <span style='color:#888'>Built with </span><b style='color:#00D4FF'>SAP AI Core GPT-5</b>
            <span style='color:#888'> · </span><b style='color:#FF6B6B'>Streamlit</b>
            <span style='color:#888'> · SAP DCOM 2026</span>
        </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Analyze Error",
    "💬 Chat with Detective",
    "📋 Session History",
    "📜 My Error History",
    "👥 My Team",
    "📊 My Trends",
])


# ══ TAB 1 — Analyze Error ══════════════════════════════
with tab1:
    st.markdown("### 📝 Paste Your SAP Error")
    st.caption(f"Platform mode: **{error_type}** — AI will analyze strictly within this platform's context")

    # ── Text input ─────────────────────────────────────
    error_text = st.text_area(
        "Paste error message, stack trace, or log:",
        height=180,
        placeholder="Paste your SAP error here...\n\nE.g. stack trace, error message, short dump text, CF logs...",
        key="error_input"
    )

    # ── Screenshot upload ──────────────────────────────
    with st.expander("📸 Upload Screenshot", expanded=False):
        st.caption("Upload a screenshot of your error. If you also pasted text above, both will be analyzed together.")
        uploaded = st.file_uploader(
            "Upload error screenshot",
            type=["png", "jpg", "jpeg"],
            key="screenshot_uploader"
        )
        if uploaded:
            st.image(uploaded, caption="Uploaded Screenshot", use_container_width=True)
            if st.session_state.screenshot_extracted is None:
                with st.spinner("🔍 AI Vision reading screenshot..."):
                    extracted = extract_text_from_image(uploaded)
                    if extracted and "Could not read" not in extracted:
                        st.session_state.screenshot_extracted = extracted
                        st.success("✅ Text extracted — included in analysis automatically.")
                    else:
                        st.error("⚠️ Could not extract text from image. Please paste error manually above.")
                        st.session_state.screenshot_extracted = None
        else:
            st.session_state.screenshot_extracted = None

    # ── Build final error text to analyze ─────────────
    # Rule: text only / image only / both combined
    def build_error_text(typed: str, extracted) -> str:
        typed_clean     = (typed or "").strip()
        extracted_clean = (extracted or "").strip()
        if typed_clean and extracted_clean:
            return f"{typed_clean}\n\n--- Screenshot Content ---\n{extracted_clean}"
        elif typed_clean:
            return typed_clean
        elif extracted_clean:
            return extracted_clean
        return ""

    final_error_text = build_error_text(error_text, st.session_state.screenshot_extracted)

    mismatch_warning  = st.empty()
    similar_error_box = st.empty()
    feedback_box      = st.empty()

    # ── Buttons — Quick Fix disabled until Analyze done ─
    a1, a2 = st.columns([3, 1])
    with a1:
        analyze_clicked = st.button(
            f"🔍 Analyze {error_type} Error",
            type="primary",
            use_container_width=True,
            key="analyze_btn",
        )
    with a2:
        quick_fix_clicked = st.button(
            "⚡ Quick Fix",
            use_container_width=True,
            key="quick_fix_btn",
        )

    # ══ ANALYZE CLICKED ════════════════════════════════
    if analyze_clicked:
        if not final_error_text:
            st.warning("⚠️ Please paste an error message or upload a screenshot first.")
        else:
            mismatch = detect_platform_mismatch(final_error_text, error_type)
            if mismatch["is_mismatch"]:
                mismatch_warning.markdown(
                    f"<div style='background:linear-gradient(135deg,#FFB80022,#FF6B0022);"
                    f"border:2px solid #FFB800;border-radius:12px;padding:16px 20px;margin:10px 0'>"
                    f"<div style='font-size:1.1rem;font-weight:800;color:#FFB800;margin-bottom:6px'>"
                    f"⚠️ Platform Mismatch Detected — Analysis Blocked</div>"
                    f"<div style='color:#ffe082;font-size:0.9rem'>"
                    f"This looks like a <b style='color:#FFA000'>{mismatch['detected_platform']}</b> error "
                    f"but you selected <b style='color:#FFB800'>{error_type}</b>.<br>"
                    f"👉 Please switch to <b style='color:#FFA000'>{mismatch['detected_platform']}</b> "
                    f"in the sidebar and try again.</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                with st.spinner(f"🔍 AI Detective investigating your {error_type} error..."):
                    try:
                        result = analyze_error(final_error_text, error_type)
                        st.session_state.current_result        = result
                        st.session_state.current_is_resolved   = False
                        st.session_state.similar_team_resolved = False
                        st.session_state.current_match_id      = None
                        st.session_state.match_feedback_given  = False
                        st.session_state.chat_history          = []

                        fp = error_fingerprint(final_error_text, error_type)
                        query_embedding = get_embedding(final_error_text, embed_model)

                        if fp not in st.session_state.analyzed_fingerprints:
                            st.session_state.analyzed_fingerprints.append(fp)
                            st.session_state.session_history.append({
                                "severity":    result["severity"],
                                "type":        error_type,
                                "preview":     final_error_text[:80] + "...",
                                "time":        datetime.now().strftime("%H:%M"),
                                "result":      result,
                                "is_resolved": False,
                                "db_id":       None,
                            })
                            db_id = save_history(
                                user_id    = st.session_state.user_id,
                                error_text = final_error_text,
                                analysis   = result["analysis"],
                                severity   = result["severity"],
                                platform   = error_type,
                                embedding  = query_embedding,
                            )
                            st.session_state.current_db_id = db_id
                            st.session_state.session_history[-1]["db_id"] = db_id
                        else:
                            st.info("ℹ️ This same error was already counted in this session.")

                        # ── Check team pool for similar resolved error (always runs) ──
                        try:
                            user_teams = get_user_teams(st.session_state.user_id)
                            for team in user_teams:
                                match = get_similar_team_error(
                                    team_id         = team['team_id'],
                                    platform        = error_type,
                                    query_embedding = query_embedding,
                                    similarity_fn   = cosine_similarity,
                                )
                                if match:
                                    ts = match['resolved_at']
                                    ts_str = ts.strftime('%d %b %Y') if hasattr(ts, 'strftime') else str(ts)
                                    resolver = "you" if match['resolved_by'] == st.session_state.username else match['resolved_by']
                                    fix_label = "Your fix:" if resolver == "you" else "Their fix:"
                                    similar_error_box.info(
                                        f"💡 **Similar error resolved in your team '{team['team_name']}'!**\n\n"
                                        f"**Resolved by:** {resolver} on {ts_str}\n\n"
                                        f"**{fix_label}** {match['resolution_text'][:400]}{'...' if len(match['resolution_text'] or '') > 400 else ''}"
                                    )
                                    st.session_state.similar_team_resolved = True
                                    st.session_state.current_match_id      = match['id']
                                    st.session_state.match_feedback_given  = False
                                    break
                        except Exception as e:
                            st.warning(f"⚠️ Team check failed: {str(e)}")

                        # ── Check personal resolved errors if no team match ──
                        if not st.session_state.similar_team_resolved:
                            try:
                                personal_match = get_similar_personal_error(
                                    user_id         = st.session_state.user_id,
                                    platform        = error_type,
                                    query_embedding = query_embedding,
                                    similarity_fn   = cosine_similarity,
                                    exclude_id      = st.session_state.current_db_id,
                                )
                                if personal_match:
                                    ts = personal_match['resolved_at']
                                    ts_str = ts.strftime('%d %b %Y') if hasattr(ts, 'strftime') else str(ts)
                                    similar_error_box.info(
                                        f"💡 **You resolved a similar error before!**\n\n"
                                        f"**Resolved on:** {ts_str}\n\n"
                                        f"**Your previous fix:** {personal_match['resolution_text'][:400]}{'...' if len(personal_match['resolution_text'] or '') > 400 else ''}"
                                    )
                            except Exception as e:
                                st.warning(f"⚠️ Personal history check failed: {str(e)}")

                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        st.stop()

    # ══ QUICK FIX CLICKED ══════════════════════════════
    if quick_fix_clicked and final_error_text:
        with st.spinner(f"⚡ Getting quick fix for {error_type}..."):
            try:
                fix = get_quick_fix(final_error_text, error_type)
                st.info(f"⚡ **Quick Fix:**\n\n{fix}")
            except Exception as e:
                st.error(f"❌ Quick fix failed: {str(e)}")

    # ══ SHOW ANALYSIS RESULT ═══════════════════════════
    if st.session_state.current_result:
        result   = st.session_state.current_result
        severity = result["severity"]
        sev_colors = {
            "CRITICAL": "#FF0000", "HIGH": "#FF6B00",
            "MEDIUM": "#FFB800", "LOW": "#00B050", "UNKNOWN": "#808080"
        }
        color = sev_colors.get(severity, "#808080")
        emoji = get_severity_emoji(severity)

        # ── Feedback buttons for similar team match ────────
        if st.session_state.current_match_id:
            if not st.session_state.match_feedback_given:
                with feedback_box.container():
                    st.markdown(
                        "<div style='background:#1a1d27;border:1px solid #2d3048;"
                        "border-radius:10px;padding:10px 16px;margin:6px 0'>"
                        "<span style='color:#aaa;font-size:0.85rem'>Was this suggestion helpful?</span>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                    fb1, fb2 = st.columns(2)
                    with fb1:
                        if st.button("👍 Yes, helpful", use_container_width=True, key="fb_helpful"):
                            save_match_feedback(st.session_state.user_id, st.session_state.current_match_id, True)
                            st.session_state.match_feedback_given = True
                            st.rerun()
                    with fb2:
                        if st.button("👎 Not relevant", use_container_width=True, key="fb_not_helpful"):
                            save_match_feedback(st.session_state.user_id, st.session_state.current_match_id, False)
                            st.session_state.match_feedback_given = True
                            st.rerun()
            else:
                feedback_box.success("✅ Thanks for your feedback! This helps improve future suggestions.")

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

        # Severity badge
        st.markdown(
            f"<div style='background:linear-gradient(90deg,{color}22,transparent);"
            f"border-left:4px solid {color};border-radius:8px;padding:16px;margin:10px 0'>"
            f"<span style='font-size:1.3rem;font-weight:800;color:{color}'>{emoji} SEVERITY: {severity}</span>"
            f"<span style='color:#888;margin-left:20px;font-size:0.9rem'>Platform: {result['error_type']}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("### 🧠 Full Analysis")
        st.markdown(result["analysis"])

        # ── Copy button ────────────────────────────────
        copy_button(result["analysis"], key="copy_analysis")

        st.markdown("---")

        # ── Action row: Download + Mark as Resolved ────
        act1, act2 = st.columns(2)
        with act1:
            report = build_report(result, [], st.session_state.username)
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with act2:
            if st.session_state.current_is_resolved:
                st.markdown(
                    "<div class='resolved-banner' style='text-align:center;font-weight:700;color:#00B050'>"
                    "✅ Marked as Resolved"
                    "</div>",
                    unsafe_allow_html=True
                )
            elif st.session_state.similar_team_resolved:
                st.markdown(
                    "<div class='resolved-banner' style='text-align:center;font-weight:700;color:#00B050'>"
                    "✅ Already resolved by your team"
                    "</div>",
                    unsafe_allow_html=True
                )
            else:
                share_tab1 = st.checkbox(
                    "Share with my team",
                    key="share_team_tab1",
                    help="If checked, this resolved error will be visible to your teammates."
                )
                if st.button("✅ Mark as Resolved", use_container_width=True, key="mark_resolved_btn"):
                    db_id = st.session_state.current_db_id
                    if db_id:
                        mark_resolved(
                            record_id       = db_id,
                            resolution_text = result["analysis"],
                            share_with_team = share_tab1,
                        )
                    st.session_state.current_is_resolved = True
                    for item in st.session_state.session_history:
                        if item.get("db_id") == db_id:
                            item["is_resolved"] = True
                            break
                    st.rerun()

        st.info(f"💬 Switch to **Chat with Detective** tab to ask {error_type}-specific follow-up questions!")


# ══ TAB 2 — Chat with Detective ════════════════════════
with tab2:
    st.markdown("### 💬 Chat with AI Error Detective")

    if not st.session_state.current_result:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;
             background:#1a1d27;border-radius:16px;border:1px dashed #2d3048'>
            <div style='font-size:3rem'>🔍</div>
            <h3 style='color:#888'>No Error Analyzed Yet</h3>
            <p style='color:#555'>Go to <b>Analyze Error</b> tab first, then come back here to chat.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        result     = st.session_state.current_result
        _platform  = result['error_type']
        _severity  = result['severity']
        _sev_color = get_severity_color(_severity)

        st.markdown(
            f"<div style='background:linear-gradient(135deg,#7B2FFF15,#00D4FF15);"
            f"border:1px solid #7B2FFF44;border-radius:10px;padding:12px 16px;margin-bottom:12px'>"
            f"<b style='color:#00D4FF'>🎯 Platform Mode: {_platform}</b> &nbsp;·&nbsp; "
            f"<span style='color:#888'>Severity: <b style='color:{_sev_color}'>{_severity}</b></span><br>"
            f"<span style='color:#666;font-size:0.82rem'>⚠️ Chat is locked to <b>{_platform}</b> — "
            f"off-topic questions will be redirected.</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("**💡 Quick Questions for this platform:**")
        suggestions = get_platform_suggestions(result["error_type"])
        q1, q2 = st.columns(2)
        for i, s in enumerate(suggestions):
            with (q1 if i % 2 == 0 else q2):
                if st.button(s, key=f"q_{i}", use_container_width=True):
                    st.session_state["prefill_q"] = s

        st.markdown("---")

        if not st.session_state.chat_history:
            st.markdown(
                f"<div style='text-align:center;color:#555;padding:20px'>"
                f"🤖 Ask anything about this {result['error_type']} error...</div>",
                unsafe_allow_html=True
            )
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(
                        f"<div class='chat-user'><b style='color:#00D4FF'>👨‍💻 Developer:</b><br>"
                        f"<span style='color:#eee'>{msg['content']}</span></div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='chat-bot'><b style='color:#7B2FFF'>🔍 AI Detective ({result['error_type']}):</b><br>"
                        f"<span style='color:#eee'>{msg['content']}</span></div>",
                        unsafe_allow_html=True
                    )

        prefill_q  = st.session_state.pop("prefill_q", "")
        user_input = st.chat_input(f"Ask a {result['error_type']}-specific question...")
        question   = prefill_q or user_input

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.spinner(f"🔍 Detective thinking in {result['error_type']} mode..."):
                try:
                    answer = chat_about_error(
                        error_text=result["error_text"],
                        previous_analysis=result["analysis"],
                        question=question,
                        error_type=result["error_type"],
                    )
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.session_state.chat_history.pop()
                    st.error(f"❌ Error: {str(e)}")
            st.rerun()

        if st.session_state.chat_history:
            st.markdown("---")
            dl1, dl2 = st.columns(2)
            with dl1:
                report = build_report(result, st.session_state.chat_history, st.session_state.username)
                st.download_button(
                    "📥 Download Full Report",
                    data=report,
                    file_name=f"resolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with dl2:
                if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat"):
                    st.session_state.chat_history = []
                    st.rerun()


# ══ TAB 3 — Session History ════════════════════════════
with tab3:
    st.markdown("### 📋 Session Error History")

    if not st.session_state.session_history:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;
             background:#1a1d27;border-radius:16px;border:1px dashed #2d3048'>
            <div style='font-size:3rem'>📋</div>
            <h3 style='color:#888'>No History Yet</h3>
            <p style='color:#555'>Analyzed errors will appear here.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        total    = len(st.session_state.session_history)
        critical = sum(1 for h in st.session_state.session_history if h["severity"] == "CRITICAL")
        high     = sum(1 for h in st.session_state.session_history if h["severity"] == "HIGH")
        resolved = sum(1 for h in st.session_state.session_history if h.get("is_resolved", False))

        m1, m2, m3, m4 = st.columns(4)
        for col, label, val, clr in [
            (m1, "Total",    total,    "#00D4FF"),
            (m2, "Critical", critical, "#FF0000"),
            (m3, "High",     high,     "#FF6B00"),
            (m4, "Resolved", resolved, "#00B050"),
        ]:
            with col:
                st.markdown(
                    f"<div class='stat-box'>"
                    f"<div style='font-size:2rem;font-weight:800;color:{clr}'>{val}</div>"
                    f"<div style='color:#888;font-size:0.8rem'>{label}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        st.markdown("---")
        for i, item in enumerate(reversed(st.session_state.session_history)):
            emoji        = get_severity_emoji(item["severity"])
            resolved_tag = " ✅" if item.get("is_resolved", False) else ""
            with st.expander(
                f"{emoji} [{item['severity']}] {item['type']} — {item['time']}{resolved_tag}",
                expanded=False
            ):
                st.caption(f"**Error:** {item['preview']}")
                st.markdown(item["result"]["analysis"])

                b1, b2 = st.columns(2)
                with b1:
                    report = build_report(item["result"], [], st.session_state.username)
                    st.download_button(
                        "📥 Download",
                        data=report,
                        file_name=f"error_{i}.txt",
                        mime="text/plain",
                        key=f"dl_{i}"
                    )
                with b2:
                    if not item.get("is_resolved", False):
                        share_sess = st.checkbox(
                            "Share with team",
                            key=f"share_sess_{i}",
                            help="Share this resolved error with your teammates."
                        )
                        if st.button("✅ Mark Resolved", use_container_width=True, key=f"resolve_{i}"):
                            db_id = item.get("db_id")
                            if db_id:
                                mark_resolved(
                                    record_id       = db_id,
                                    resolution_text = item["result"]["analysis"],
                                    share_with_team = share_sess,
                                )
                            item["is_resolved"] = True
                            if item["result"] == st.session_state.current_result:
                                st.session_state.current_is_resolved = True
                            st.rerun()
                    else:
                        st.markdown(
                            "<div style='text-align:center;color:#00B050;font-weight:700;padding:8px'>✅ Resolved</div>",
                            unsafe_allow_html=True
                        )


# ══ TAB 4 — My Error History (Persistent) ═════════════
with tab4:
    st.markdown(f"### 📜 Your Top 10 Resolved Errors — *{st.session_state.username}*")
    st.caption("Your personal fix library — persists across all sessions. Only errors you marked as resolved appear here.")

    try:
        history = get_history(st.session_state.user_id)

        if not history:
            st.markdown("""
            <div style='text-align:center;padding:60px 20px;
                 background:#1a1d27;border-radius:16px;border:1px dashed #2d3048'>
                <div style='font-size:3rem'>📜</div>
                <h3 style='color:#888'>No Resolved Errors Yet</h3>
                <p style='color:#555'>Analyze an error and mark it as resolved — it will appear here.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='color:#888;font-size:0.85rem;margin-bottom:12px'>"
                f"Showing <b style='color:#00D4FF'>{len(history)}</b> saved error(s)</div>",
                unsafe_allow_html=True
            )
            for idx, entry in enumerate(reversed(history), 1):
                ts           = entry["created_at"]
                platform     = entry.get("platform",    "Unknown")
                severity     = entry.get("severity",    "UNKNOWN")
                is_resolved  = entry.get("is_resolved", False)
                ts_str       = ts.strftime("%d %b %Y, %H:%M") if hasattr(ts, "strftime") else str(ts)
                emoji        = get_severity_emoji(severity)
                clr          = get_severity_color(severity)
                resolved_tag = " ✅" if is_resolved else ""

                with st.expander(f"{emoji} [{severity}] {platform} — {ts_str}{resolved_tag}", expanded=False):
                    st.markdown(
                        f"<div style='font-size:0.78rem;color:#666;margin-bottom:10px'>"
                        f"🕐 {ts_str} &nbsp;·&nbsp; 🎯 <b style='color:#00D4FF'>{platform}</b>"
                        f" &nbsp;·&nbsp; <b style='color:{clr}'>{severity}</b>"
                        f"{'&nbsp;·&nbsp;<b style=\"color:#00B050\">✅ Resolved</b>' if is_resolved else ''}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<div class='chat-user'><b style='color:#00D4FF'>🐛 Error Submitted:</b><br>"
                        f"<span style='color:#eee;font-size:0.9rem'>"
                        f"{entry['error_text'][:300]}{'...' if len(entry['error_text']) > 300 else ''}"
                        f"</span></div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<div class='chat-bot'><b style='color:#7B2FFF'>🔍 AI Analysis:</b><br>"
                        f"<span style='color:#eee'>{entry['analysis']}</span></div>",
                        unsafe_allow_html=True
                    )

                    h1, h2 = st.columns(2)
                    with h1:
                        report = build_report(
                            {"error_text": entry["error_text"], "analysis": entry["analysis"],
                             "error_type": platform, "severity": severity},
                            [], st.session_state.username
                        )
                        st.download_button(
                            "📥 Download Report",
                            data=report,
                            file_name=f"error_history_{idx}.txt",
                            mime="text/plain",
                            key=f"hist_dl_{idx}"
                        )
                    with h2:
                        if not is_resolved:
                            share_hist = st.checkbox(
                                "Share with team",
                                key=f"share_hist_{idx}",
                                help="Share this resolved error with your teammates."
                            )
                            if st.button("✅ Mark Resolved", use_container_width=True, key=f"hist_resolve_{idx}"):
                                mark_resolved(
                                    record_id       = entry["id"],
                                    resolution_text = entry["analysis"],
                                    share_with_team = share_hist,
                                )
                                st.rerun()
                        else:
                            st.markdown(
                                "<div style='text-align:center;color:#00B050;font-weight:700;padding:8px'>✅ Resolved</div>",
                                unsafe_allow_html=True
                            )

    except Exception as e:
        st.error(f"❌ Could not load history: {str(e)}")
        st.info("Make sure PostgreSQL is running and your .env DB settings are correct.")



# ══ TAB 5 — Team Workspace ═════════════════════════════
with tab5:
    st.markdown("### 👥 Team Workspace")
    st.caption("Create or join a team. Resolved errors are shared with teammates automatically.")

    try:
        user_teams = get_user_teams(st.session_state.user_id)
    except Exception as e:
        st.error(f"❌ Could not load teams: {str(e)}")
        user_teams = []

    # ── Create / Join section ──────────────────────────
    st.markdown("#### ➕ Create or Join a Team")
    cj1, cj2 = st.columns(2)

    with cj1:
        st.markdown("**Create a new team:**")
        new_team_name = st.text_input("Team name", placeholder="e.g. SAP BTP Project Alpha", key="new_team_name")
        if st.button("🚀 Create Team", use_container_width=True, type="primary", key="create_team_btn"):
            if new_team_name.strip():
                result = create_team(new_team_name, st.session_state.user_id)
                if result["success"]:
                    st.success(f"✅ {result['message']}")
                    st.info(f"🔑 Your Team Code: **{result['team_code']}** — share this with colleagues")
                    st.rerun()
                else:
                    st.error(f"❌ {result['message']}")
            else:
                st.warning("⚠️ Please enter a team name.")

    with cj2:
        st.markdown("**Join an existing team:**")
        join_code = st.text_input("Team code", placeholder="e.g. SAP-A3X9B2", key="join_team_code")
        if st.button("🤝 Join Team", use_container_width=True, type="primary", key="join_team_btn"):
            if join_code.strip():
                result = join_team(join_code, st.session_state.user_id)
                if result["success"]:
                    st.success(f"✅ {result['message']}")
                    st.rerun()
                else:
                    st.error(f"❌ {result['message']}")
            else:
                st.warning("⚠️ Please enter a team code.")

    st.markdown("---")

    # ── My Teams list ──────────────────────────────────
    if not user_teams:
        st.markdown("""
        <div style='text-align:center;padding:40px 20px;
             background:#1a1d27;border-radius:16px;border:1px dashed #2d3048'>
            <div style='font-size:3rem'>👥</div>
            <h3 style='color:#888'>No Teams Yet</h3>
            <p style='color:#555'>Create a team or join one using a team code above.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for team in user_teams:
            tid        = team["team_id"]
            tname      = team["team_name"]
            tcode      = team["team_code"]
            mcount     = team["member_count"]
            is_admin   = team["created_by"] == st.session_state.user_id

            with st.expander(
                f"👥 {tname}  ·  {mcount} member{'s' if mcount != 1 else ''}  "
                f"{'👤 Admin' if is_admin else ''}",
                expanded=True
            ):
                # Team code + leave button
                tc1, tc2 = st.columns([3, 1])
                with tc1:
                    st.markdown(
                        f"<div style='background:#1a1d27;border:1px solid #7B2FFF44;"
                        f"border-radius:8px;padding:10px;'>"
                        f"<span style='color:#888;font-size:0.8rem'>Team Code (share with colleagues):</span><br>"
                        f"<b style='color:#00D4FF;font-size:1.1rem;letter-spacing:2px'>{tcode}</b>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with tc2:
                    if st.button("🚪 Leave", key=f"leave_{tid}", use_container_width=True):
                        result = leave_team(tid, st.session_state.user_id)
                        if result["success"]:
                            st.rerun()
                        else:
                            st.error(result.get("message", "Failed to leave team."))

                st.markdown("---")

                # Members list
                try:
                    members = get_team_members(tid)
                    st.markdown("**👤 Members:**")
                    member_cols = st.columns(min(len(members), 4))
                    for i, m in enumerate(members):
                        with member_cols[i % 4]:
                            admin_tag = " 👤" if m.get("is_admin") else ""
                            joined    = m["joined_at"]
                            joined_str = joined.strftime("%d %b %Y") if hasattr(joined, "strftime") else str(joined)
                            st.markdown(
                                f"<div style='background:#1a1d27;border:1px solid #2d3048;"
                                f"border-radius:8px;padding:8px;text-align:center'>"
                                f"<b style='color:#00D4FF'>{m['username']}{admin_tag}</b><br>"
                                f"<span style='color:#666;font-size:0.75rem'>Joined {joined_str}</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                except Exception as e:
                    st.error(f"Could not load members: {e}")

                st.markdown("---")

                # Shared resolved error pool
                st.markdown("**🔧 Team Resolved Errors:**")
                try:
                    resolved_errors = get_team_resolved_errors(tid)
                    if not resolved_errors:
                        st.markdown(
                            "<div style='color:#555;padding:12px;background:#1a1d27;"
                            "border-radius:8px;text-align:center'>"
                            "No resolved errors yet. When a teammate marks an error as resolved, "
                            "it will appear here.</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.caption(f"Showing {len(resolved_errors)} resolved error(s) from all teammates")
                        for idx, err in enumerate(resolved_errors, 1):
                            ts      = err.get("resolved_at") or err.get("created_at")
                            ts_str  = ts.strftime("%d %b %Y, %H:%M") if hasattr(ts, "strftime") else str(ts)
                            emoji   = get_severity_emoji(err.get("severity", "UNKNOWN"))
                            sev_clr = get_severity_color(err.get("severity", "UNKNOWN"))

                            with st.expander(
                                f"{emoji} [{err.get('severity','?')}] {err.get('platform','?')} "
                                f"— resolved by **{err['resolved_by']}** on {ts_str}",
                                expanded=False
                            ):
                                st.markdown(
                                    f"<div class='chat-user'>"
                                    f"<b style='color:#00D4FF'>🐛 Error:</b><br>"
                                    f"<span style='color:#eee;font-size:0.9rem'>"
                                    f"{err['error_text'][:300]}{'...' if len(err['error_text']) > 300 else ''}"
                                    f"</span></div>",
                                    unsafe_allow_html=True
                                )
                                st.markdown(
                                    f"<div class='chat-bot'>"
                                    f"<b style='color:#7B2FFF'>✅ Resolution:</b><br>"
                                    f"<span style='color:#eee'>{err.get('resolution_text') or err.get('analysis','')}</span>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                except Exception as e:
                    st.error(f"Could not load team errors: {e}")


# ══ TAB 6 — My Trends ═════════════════════════════════
with tab6:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    _PLOT_LAYOUT = dict(
        template="plotly_dark",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a1d27",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(gridcolor="#2d3048", linecolor="#2d3048"),
        yaxis=dict(gridcolor="#2d3048", linecolor="#2d3048"),
    )

    st.markdown(f"### 📊 My Error Trends — *{st.session_state.username}*")
    st.caption("Personal insights based on your complete error history.")

    try:
        trends = get_user_trends(st.session_state.user_id)
        stats  = trends["stats"]
        total  = stats["total"] or 0

        if total == 0:
            st.markdown("""
            <div style='text-align:center;padding:60px 20px;
                 background:#1a1d27;border-radius:16px;border:1px dashed #2d3048'>
                <div style='font-size:3rem'>📊</div>
                <h3 style='color:#888'>No Data Yet</h3>
                <p style='color:#555'>Analyze some errors and your trends will appear here.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            resolved  = stats["resolved"] or 0
            critical  = stats["critical"] or 0
            res_rate  = round((resolved / total) * 100) if total else 0

            # ── Overview metrics ───────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            for col, label, val, clr in [
                (m1, "Total Errors",    total,          "#00D4FF"),
                (m2, "Resolved",        resolved,       "#00B050"),
                (m3, "Critical",        critical,       "#FF0000"),
                (m4, "Resolution Rate", f"{res_rate}%", "#7B2FFF"),
            ]:
                with col:
                    st.markdown(
                        f"<div class='stat-box'>"
                        f"<div style='font-size:2rem;font-weight:800;color:{clr}'>{val}</div>"
                        f"<div style='color:#888;font-size:0.8rem'>{label}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

            col_left, col_right = st.columns(2)

            # ── Errors by Platform ─────────────────────────
            with col_left:
                st.markdown("#### 🎯 Errors by Platform")
                if trends["by_platform"]:
                    df_plat = pd.DataFrame(trends["by_platform"])
                    fig_plat = px.bar(
                        df_plat, x="platform", y="count",
                        color="count",
                        color_continuous_scale=[[0, "#3B1F7F"], [1, "#7B2FFF"]],
                        labels={"platform": "Platform", "count": "Errors"},
                        title="Errors by Platform",
                    )
                    fig_plat.update_layout(**_PLOT_LAYOUT, coloraxis_showscale=False,
                                           title_font_color="#7B2FFF")
                    fig_plat.update_traces(
                        marker_line_width=0,
                        hovertemplate="<b>%{x}</b><br>Errors: %{y}<extra></extra>",
                    )
                    st.plotly_chart(fig_plat, use_container_width=True)
                else:
                    st.caption("No data yet.")

            # ── Errors by Severity ─────────────────────────
            with col_right:
                st.markdown("#### 🔴 Errors by Severity")
                if trends["by_severity"]:
                    _sev_color_map = {
                        "CRITICAL": "#FF0000", "HIGH": "#FF6B00",
                        "MEDIUM": "#FFB800", "LOW": "#00B050", "UNKNOWN": "#808080",
                    }
                    df_sev = pd.DataFrame(trends["by_severity"])
                    df_sev["color"] = df_sev["severity"].map(
                        lambda s: _sev_color_map.get(s, "#808080")
                    )
                    fig_sev = px.bar(
                        df_sev, x="severity", y="count",
                        color="severity",
                        color_discrete_map=_sev_color_map,
                        labels={"severity": "Severity", "count": "Errors"},
                        title="Errors by Severity",
                    )
                    fig_sev.update_layout(**_PLOT_LAYOUT, showlegend=False,
                                          title_font_color="#FF6B00")
                    fig_sev.update_traces(
                        marker_line_width=0,
                        hovertemplate="<b>%{x}</b><br>Errors: %{y}<extra></extra>",
                    )
                    st.plotly_chart(fig_sev, use_container_width=True)
                else:
                    st.caption("No data yet.")

            st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

            col_left2, col_right2 = st.columns(2)

            # ── Errors by Day of Week ──────────────────────
            with col_left2:
                st.markdown("#### 📅 Errors by Day of Week")
                if trends["by_day"]:
                    day_order = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
                    df_day = pd.DataFrame(trends["by_day"])
                    df_day["day"] = df_day["day"].str.strip()
                    df_day = df_day.set_index("day").reindex(
                        [d for d in day_order if d in df_day.index]
                    ).reset_index().dropna()
                    fig_day = px.bar(
                        df_day, x="day", y="count",
                        color="count",
                        color_continuous_scale=[[0, "#004D7A"], [1, "#00D4FF"]],
                        labels={"day": "Day", "count": "Errors"},
                        title="Errors by Day of Week",
                        category_orders={"day": day_order},
                    )
                    fig_day.update_layout(**_PLOT_LAYOUT, coloraxis_showscale=False,
                                          title_font_color="#00D4FF")
                    fig_day.update_traces(
                        marker_line_width=0,
                        hovertemplate="<b>%{x}</b><br>Errors: %{y}<extra></extra>",
                    )
                    st.plotly_chart(fig_day, use_container_width=True)
                else:
                    st.caption("No data yet.")

            # ── Monthly Trend ──────────────────────────────
            with col_right2:
                st.markdown("#### 📈 Monthly Trend (Last 6 Months)")
                if trends["by_month"]:
                    df_month = pd.DataFrame(trends["by_month"])
                    fig_month = go.Figure()
                    fig_month.add_trace(go.Scatter(
                        x=df_month["month"],
                        y=df_month["count"],
                        mode="lines+markers",
                        line=dict(color="#00B050", width=3),
                        marker=dict(size=8, color="#00D480",
                                    line=dict(color="#00B050", width=2)),
                        fill="tozeroy",
                        fillcolor="rgba(0,176,80,0.1)",
                        hovertemplate="<b>%{x}</b><br>Errors: %{y}<extra></extra>",
                        name="Errors",
                    ))
                    fig_month.update_layout(
                        **_PLOT_LAYOUT,
                        title="Monthly Error Volume",
                        title_font_color="#00B050",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_month, use_container_width=True)
                else:
                    st.caption("No data yet.")

            st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

            # ── Key Insight ────────────────────────────────
            st.markdown("#### 💡 Key Insight")
            if trends["by_platform"]:
                top_platform = trends["by_platform"][0]
                top_pct      = round((top_platform["count"] / total) * 100)
                st.info(
                    f"**{top_pct}%** of your errors are **{top_platform['platform']}**-related. "
                    f"You've resolved **{res_rate}%** of all errors you've analyzed."
                )

    except Exception as e:
        st.error(f"❌ Could not load trends: {str(e)}")


# ── Footer ─────────────────────────────────────────────
st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;color:#555;padding:10px;font-size:0.85rem'>"
    "🔍 <b style='color:#00D4FF'>AI Error Detective</b> · Built for SAP DCOM 2026 · "
    "Powered by <b style='color:#7B2FFF'>SAP AI Core GPT-5</b> + "
    "<b style='color:#00D4FF'>Streamlit</b> "
    "· 100% Free AI Tools"
    "</div>",
    unsafe_allow_html=True
)
