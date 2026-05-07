# backend.py — AI Error Detective Core Engine
import os
import re
import time
import base64
import numpy as np
import requests as _http
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

_AICORE_BASE          = os.getenv("AICORE_BASE_URL", "").rstrip("/")
_AICORE_RG            = os.getenv("AICORE_RESOURCE_GROUP", "default")
_AICORE_DEPLOYMENT_ID = os.getenv("AICORE_DEPLOYMENT_ID", "")
_AICORE_MODEL         = "gpt-5"
_VISION_DEPLOYMENT_ID = "db5002b869569b2c"  # GPT-4o — supports image_url vision

# Module-level cache — survives Streamlit reruns within the same worker process
_TOKEN_CACHE: dict = {}
_DEP_CACHE:   dict = {}


def _get_token() -> str:
    now = time.time()
    if _TOKEN_CACHE.get("token") and now < _TOKEN_CACHE.get("expires_at", 0) - 300:
        return _TOKEN_CACHE["token"]
    r = _http.post(
        os.getenv("AICORE_AUTH_URL", "").rstrip("/") + "/oauth/token",
        data={"grant_type": "client_credentials"},
        auth=(os.getenv("AICORE_CLIENT_ID", ""), os.getenv("AICORE_CLIENT_SECRET", "")),
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    _TOKEN_CACHE["token"]      = data["access_token"]
    _TOKEN_CACHE["expires_at"] = now + float(data.get("expires_in", 3600))
    return _TOKEN_CACHE["token"]


def _get_dep_url(model: str = _AICORE_MODEL) -> str:
    if model in _DEP_CACHE:
        return _DEP_CACHE[model]
    # If deployment ID is set directly, skip the API lookup entirely
    if _AICORE_DEPLOYMENT_ID:
        url = f"{_AICORE_BASE}/v2/inference/deployments/{_AICORE_DEPLOYMENT_ID}"
        _DEP_CACHE[model] = url
        return url
    token = _get_token()
    r = _http.get(
        f"{_AICORE_BASE}/v2/lm/deployments",
        headers={"Authorization": f"Bearer {token}", "AI-Resource-Group": _AICORE_RG},
        timeout=30,
    )
    r.raise_for_status()
    for dep in r.json().get("resources", []):
        if dep.get("status") != "RUNNING":
            continue
        model_name = (
            dep.get("details", {})
               .get("resources", {})
               .get("backend_details", {})
               .get("model", {})
               .get("name", "")
            or dep.get("configurationName", "")
        )
        if model.lower() in model_name.lower():
            url = f"{_AICORE_BASE}/v2/inference/deployments/{dep['id']}"
            _DEP_CACHE[model] = url
            return url
    raise RuntimeError(
        f"No running '{model}' deployment found in AI Core. "
        "Check your AI Core cockpit and ensure the deployment is in RUNNING state."
    )


def _aicore_chat(messages: list, max_completion_tokens: int = None) -> str:
    token   = _get_token()
    dep_url = _get_dep_url(_AICORE_MODEL)
    body: dict = {"model": _AICORE_MODEL, "messages": messages}
    if max_completion_tokens:
        body["max_completion_tokens"] = max_completion_tokens
    r = _http.post(
        f"{dep_url}/v1/chat/completions",
        json=body,
        headers={
            "Authorization":     f"Bearer {token}",
            "AI-Resource-Group": _AICORE_RG,
            "Content-Type":      "application/json",
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def warm_up_aicore():
    """Pre-fetch OAuth token + deployment URL at startup to avoid cold-start lag."""
    try:
        _get_dep_url("gpt-5")
    except Exception:
        pass


def _ai_core_invoke(prompt_str: str, system_prompt: str = None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt_str})
    return _aicore_chat(messages)


# ─────────────────────────────────────────────
# PLATFORM CONTEXT MAP
# Rich context injected into every prompt so
# the LLM stays strictly within that ecosystem.
# ─────────────────────────────────────────────

PLATFORM_CONTEXT = {
    "SAP BTP": """
You are an expert EXCLUSIVELY in SAP Business Technology Platform (BTP).
Your deep knowledge covers:
- Cloud Foundry: cf CLI, app deployments, buildpacks, VCAP_SERVICES, service bindings
- BTP Cockpit: subaccounts, spaces, service instances, destinations, entitlements
- SAP HANA Cloud, SAP AI Core, SAP Integration Suite on BTP
- MTA (Multi-Target Application): mta.yaml, mtad.yaml, mbt build & deploy
- BTP Security: XSUAA, OAuth2/JWT, role collections, trust configurations, IAS
- BTP errors: 502/503 gateway errors, app crashes, OOM, binding failures, quota issues

STRICT RULE: Answer ONLY questions about SAP BTP.
If question is unrelated to BTP, respond exactly:
"I am in SAP BTP mode. This question is outside BTP scope — please ask a BTP-related question or switch the platform in the sidebar."
""",

    "CAP (Cloud Application Programming)": """
You are an expert EXCLUSIVELY in SAP CAP (Cloud Application Programming Model).
Your deep knowledge covers:
- CDS (Core Data Services): schema, service definitions, projections, annotations
- CAP Node.js runtime: service handlers, before/on/after hooks, custom logic
- CAP Java runtime: Spring Boot, CAP Java SDK, event handlers
- Database adapters: SQLite (dev/test), SAP HANA (prod), PostgreSQL
- CDS CLI: cds build, cds deploy, cds watch, cds compile
- OData V4 protocol, CAP Fiori Elements integration, remote services
- CAP on BTP: MTA deployment, HANA bindings, XSUAA, multitenancy
- Common CAP errors: entity not deployed, SQLite mismatch, MODULE_NOT_FOUND, schema errors

STRICT RULE: Answer ONLY questions about CAP framework.
If unrelated, respond exactly:
"I am in CAP mode. This question is outside CAP scope — please ask a CAP-related question or switch the platform in the sidebar."
""",

    "ABAP Cloud": """
You are an expert EXCLUSIVELY in ABAP Cloud (Clean Core ABAP).
Your deep knowledge covers:
- ABAP RESTful Application Programming Model (RAP): BOs, behavior definitions, projections, actions
- Clean Core rules: released APIs only, no classic/forbidden statements, tier-1/2/3
- SAP S/4HANA Cloud ABAP Environment (Steampunk / BTP ABAP Environment)
- ADT (ABAP Development Tools) in Eclipse: packages, transport, activation
- CDS views in ABAP Cloud: VDM layers, annotations (@AccessControl, @UI, @Semantics)
- Key User Extensibility vs Developer Extensibility in S/4HANA Cloud
- Common ABAP Cloud errors: API not released, clean core violations, RAP BO issues

STRICT RULE: Answer ONLY questions about ABAP Cloud / Clean Core.
If unrelated, respond exactly:
"I am in ABAP Cloud mode. This question is outside ABAP Cloud scope — please ask an ABAP Cloud question or switch the platform in the sidebar."
""",

    "ABAP On-Premise": """
You are an expert EXCLUSIVELY in classic ABAP On-Premise development.
Your deep knowledge covers:
- ABAP language: reports, function modules, classes, BAPIs, RFCs, IDocs
- Enhancements: user exits, customer exits, BADIs, implicit/explicit enhancements
- SAP GUI transactions: SE38, SE80, SE24, SE09/SE10, SM21, ST22, SM50, SM66
- ABAP Workbench, Object Navigator, transport management
- Runtime dump analysis: ST22, ABAP short dumps, memory analysis
- Performance tools: SE30, SAT, secondary indexes, buffering
- Common errors: DYNPRO_SEND_IN_BACKGROUND, RAISE_EXCEPTION, access violations

STRICT RULE: Answer ONLY questions about ABAP On-Premise.
If unrelated, respond exactly:
"I am in ABAP On-Premise mode. This question is outside ABAP On-Premise scope — please ask an ABAP On-Premise question or switch the platform in the sidebar."
""",

    "SAP Fiori / UI5": """
You are an expert EXCLUSIVELY in SAP Fiori and SAPUI5 development.
Your deep knowledge covers:
- SAPUI5 framework: MVC pattern, controllers, XML views, JSON/OData models, routing
- Fiori Elements: List Report, Object Page, Analytical List Page, Worklist
- OData V2/V4 service binding, CDS annotations (@UI.*, @Common.*, @Consumption.*)
- Fiori Launchpad (FLP): tiles, target mappings, catalogs, semantic objects
- SAP Business Application Studio (BAS): Fiori tools extension, generators
- Fiori preview, deployment to ABAP backend and BTP
- Common errors: binding context undefined, manifest routing errors, CORS, OData 400/500

STRICT RULE: Answer ONLY questions about SAP Fiori/UI5.
If unrelated, respond exactly:
"I am in SAP Fiori/UI5 mode. This question is outside Fiori/UI5 scope — please ask a Fiori/UI5 question or switch the platform in the sidebar."
""",

    "SAP S/4HANA": """
You are an expert EXCLUSIVELY in SAP S/4HANA (Cloud and On-Premise).
Your deep knowledge covers:
- S/4HANA functional modules: FI/CO, MM, SD, PP, PM, QM, PS
- S/4HANA technical: CDS views, AMDP, virtual data model (VDM), AIF
- S/4HANA migration: SAP Readiness Check, Simplification List, custom code adaptation
- SAP Activate methodology, fit-to-standard, delta design workshops
- S/4HANA Cloud Public/Private: SSCUI, key user extensibility, in-app/side-by-side
- S/4HANA APIs: SAP API Business Hub, OData APIs, SOAP, IDoc
- Common S/4HANA errors: posting failures, config gaps, deprecation issues, migration errors

STRICT RULE: Answer ONLY questions about SAP S/4HANA.
If unrelated, respond exactly:
"I am in SAP S/4HANA mode. This question is outside S/4HANA scope — please ask an S/4HANA question or switch the platform in the sidebar."
""",

    "SAP Integration Suite": """
You are an expert EXCLUSIVELY in SAP Integration Suite (CPI/HCI).
Your deep knowledge covers:
- SAP Cloud Integration: iFlows, adapters (SFTP, SOAP, REST, OData, JDBC, AS2, AMQP)
- iFlow design: message mapping, XSLT mapping, Groovy scripts, content modifier, router
- Message processing log (MPL): error analysis, trace mode, header/property inspection
- API Management: API proxies, policies (rate limit, OAuth, JWT), products, developer portal
- Event Mesh: queues, topics, webhook subscriptions, SAP Event Broker
- Certificates and credentials: keystore, credential store, OAuth2 config
- Common CPI errors: adapter timeout, mapping failures, certificate expired, auth errors

STRICT RULE: Answer ONLY questions about SAP Integration Suite.
If unrelated, respond exactly:
"I am in SAP Integration Suite mode. This question is outside Integration Suite scope — please ask an Integration Suite question or switch the platform in the sidebar."
""",

    "SAP HANA": """
You are an expert EXCLUSIVELY in SAP HANA database.
Your deep knowledge covers:
- SAP HANA SQL: column store, calculation engine, SQL script, joins, window functions
- HANA Modeling: calculation views (graphical + SQL), analytic privileges
- HANA Administration: backup/recovery, memory management, system replication, HA/DR
- HDI (HANA Deployment Infrastructure): .hdbtable, .hdbview, .hdbcalculationview artifacts
- SAP HANA Cloud vs on-premise differences, HANA Cloud Central
- HANA XSA (extended application services), multi-target application on HANA
- Common HANA errors: out-of-memory, lock timeout, column store errors, HDI deploy failures

STRICT RULE: Answer ONLY questions about SAP HANA.
If unrelated, respond exactly:
"I am in SAP HANA mode. This question is outside HANA scope — please ask an SAP HANA question or switch the platform in the sidebar."
""",

    "SAP Build Apps": """
You are an expert EXCLUSIVELY in SAP Build Apps (formerly AppGyver / SAP AppGyver).
Your deep knowledge covers:
- SAP Build Apps visual development: UI canvas, component library, drag-and-drop
- Logic canvas: flow functions, events, custom JavaScript, REST API integration
- Data resources: OData (V2/V4), REST API, SAP BTP destinations, direct service calls
- Formula editor: AppGyver formula language, data/app/page variables, bindings
- Authentication: SAP BTP auth flow, OAuth2 PKCE, role-based access
- Build & Deploy: web app, iOS (Xcode), Android (Android Studio), MDK comparison
- Common errors: formula type mismatch, data binding undefined, auth redirect loops, API 401/403

STRICT RULE: Answer ONLY questions about SAP Build Apps.
If unrelated, respond exactly:
"I am in SAP Build Apps mode. This question is outside Build Apps scope — please ask a Build Apps question or switch the platform in the sidebar."
""",

    "Other SAP": """
You are a broad SAP expert covering all SAP products and technologies.
Answer any SAP-related question including SAP ECC, BW, GRC, SuccessFactors, Ariba, Concur,
SAP Basis, Netweaver, ABAP, Java stack, and all BTP services.
If the question is completely unrelated to SAP, politely redirect back to SAP topics.
""",
}


# ─────────────────────────────────────────────
# PLATFORM-SPECIFIC FOLLOW-UP SUGGESTIONS
# Shown as quick-click buttons in the chat tab
# ─────────────────────────────────────────────

PLATFORM_SUGGESTIONS = {
    "SAP BTP": [
        "Which BTP service binding is causing this?",
        "How do I check BTP app logs with cf CLI?",
        "Is this related to XSUAA/OAuth2 config?",
        "How do I fix this in mta.yaml?",
        "Which BTP Cockpit section should I check?",
        "How do I increase BTP memory quota?",
    ],
    "CAP (Cloud Application Programming)": [
        "Which CDS entity definition is wrong?",
        "Which cds command do I run to fix this?",
        "Is this a SQLite vs HANA difference?",
        "How do I redeploy after the fix?",
        "Is this in the service handler or schema?",
        "How do I check the CDS build log?",
    ],
    "ABAP Cloud": [
        "Is this a clean core / released API violation?",
        "Which RAP behavior definition needs changing?",
        "How do I check this error in ADT Eclipse?",
        "Which released API should I use instead?",
        "Is this related to a CDS view annotation?",
        "How do I activate the ABAP object after fix?",
    ],
    "ABAP On-Premise": [
        "Which transaction code to check first?",
        "How do I analyze this dump in ST22?",
        "Is there a SAP Note for this error?",
        "How do I trace this in SM50 or SM66?",
        "Which BADI or user exit is involved?",
        "How do I transport this fix via SE10?",
    ],
    "SAP Fiori / UI5": [
        "Which OData binding property is failing?",
        "How do I debug this in Chrome DevTools?",
        "Is this a manifest.json routing issue?",
        "Which UI5 controller method to check?",
        "Is this a Fiori Launchpad config issue?",
        "Which CDS annotation change is needed?",
    ],
    "SAP S/4HANA": [
        "Which S/4HANA config step is missing?",
        "Is this listed in the Simplification List?",
        "Which SSCUI should I check for this?",
        "Is this a custom code adaptation issue?",
        "Which S/4HANA OData API should I use?",
        "How do I use SAP Readiness Check here?",
    ],
    "SAP Integration Suite": [
        "Which iFlow adapter is causing this?",
        "How do I read the message processing log?",
        "Is this a certificate or credential issue?",
        "Which Groovy script change is needed?",
        "How do I fix the message mapping?",
        "Is this an API Management proxy policy issue?",
    ],
    "SAP HANA": [
        "Which HANA calculation view is wrong?",
        "How do I check HANA memory usage?",
        "Is this an HDI deploy artifact error?",
        "Which HANA SQL change is needed?",
        "How do I analyze this in HANA cockpit?",
        "Is this a column store vs row store issue?",
    ],
    "SAP Build Apps": [
        "Which data variable binding is failing?",
        "How do I fix this formula expression?",
        "Is this a BTP destination config issue?",
        "Which logic canvas node is causing this?",
        "How do I reconnect the OData resource?",
        "Is this a Build Apps auth redirect issue?",
    ],
    "Other SAP": [
        "What SAP component is causing this?",
        "Is there a relevant SAP Note?",
        "Which transaction code should I check?",
        "How do I prevent this in future?",
        "Is there an alternative approach?",
        "How long will this fix typically take?",
    ],
}


# Lightweight keyword model for mismatch detection.
# This is intentionally conservative: it only flags mismatch on strong signals.
PLATFORM_KEYWORDS = {
    "SAP BTP": [
        "cf cli", "cloud foundry", "vcap_services", "mta.yaml", "xsuaa",
        "subaccount", "entitlement", "service binding", "buildpack",
    ],
    "CAP (Cloud Application Programming)": [
        "@sap/cds", "cds build", "cds deploy", "cds watch", "catalogservice",
        "entity", "sqlite_error", "cap", "core data services",
    ],
    "ABAP Cloud": [
        "rap", "behavior definition", "clean core", "released api",
        "steampunk", "abap environment", "adt", "projection view",
    ],
    "ABAP On-Premise": [
        "st22", "se80", "se38", "dynpro_send_in_background", "short dump",
        "sm50", "sm66", "badi", "user exit", "abap runtime error",
    ],
    "SAP Fiori / UI5": [
        "sapui5", "ui5", "manifest.json", "odata v2", "odata v4",
        "fiori launchpad", "xml view", "controller.js", "binding context",
    ],
    "SAP S/4HANA": [
        "s/4hana", "s4hana", "sscui", "simplification list", "fi/co",
        "mm", "sd", "readiness check", "activate methodology",
    ],
    "SAP Integration Suite": [
        "iflow", "cpi", "cloud integration", "message processing log",
        "api management", "event mesh", "groovy script", "adapter timeout",
    ],
    "SAP HANA": [
        "hdi", "hdbtable", "hdbview", "calculation view", "column store",
        "hana cockpit", "sqlscript", "lock timeout",
    ],
    "SAP Build Apps": [
        "build apps", "appgyver", "logic canvas", "formula", "data variable",
        "oauth2 pkce", "binding undefined",
    ],
}


# ─────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────

def get_platform_context(error_type: str) -> str:
    """Return the platform-specific context string."""
    return PLATFORM_CONTEXT.get(error_type, PLATFORM_CONTEXT["Other SAP"])


def get_platform_suggestions(error_type: str) -> list:
    """Return platform-specific follow-up question buttons."""
    return PLATFORM_SUGGESTIONS.get(error_type, PLATFORM_SUGGESTIONS["Other SAP"])


def detect_platform_mismatch(error_text: str, selected_platform: str) -> dict:
    """Detect strong platform mismatch signals from raw error text.

    A mismatch is flagged only when:
    - another platform has strong evidence (>= 2 keyword hits), and
    - selected platform has no keyword evidence.
    """
    text = (error_text or "").lower()

    if not text.strip() or selected_platform == "Other SAP":
        return {
            "is_mismatch": False,
            "detected_platform": "Unknown",
            "selected_score": 0,
            "detected_score": 0,
        }

    scores = {}
    for platform, keywords in PLATFORM_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in text:
                score += 1
        scores[platform] = score

    detected_platform = max(scores, key=scores.get)
    detected_score = scores[detected_platform]
    selected_score = scores.get(selected_platform, 0)

    is_mismatch = (
        detected_score >= 1
        and detected_platform != selected_platform
        and selected_score == 0
    )

    return {
        "is_mismatch": is_mismatch,
        "detected_platform": detected_platform if detected_score > 0 else "Unknown",
        "selected_score": selected_score,
        "detected_score": detected_score,
    }


def analyze_error(error_text: str, error_type: str) -> dict:
    if not error_text.strip():
        raise ValueError("Error text cannot be empty")

    platform_context = get_platform_context(error_type)

    prompt = PromptTemplate(
        input_variables=["error_text", "error_type", "platform_context"],
        template=ERROR_ANALYSIS_PROMPT
    )
    response = _ai_core_invoke(prompt.format(
        error_text=error_text[:5000],
        error_type=error_type,
        platform_context=platform_context,
    ), system_prompt=platform_context)
    severity = extract_severity(response)

    return {
        "analysis":   response,
        "severity":   severity,
        "error_text": error_text,
        "error_type": error_type,
    }


def extract_severity(analysis_text: str) -> str:
    text_upper = analysis_text.upper()
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        if level in text_upper:
            return level
    return "UNKNOWN"


def get_quick_fix(error_text: str, error_type: str) -> str:
    platform_context = get_platform_context(error_type)

    prompt = PromptTemplate(
        input_variables=["error_text", "error_type", "platform_context"],
        template=QUICK_FIX_PROMPT
    )
    return _ai_core_invoke(prompt.format(
        error_text=error_text[:2000],
        error_type=error_type,
        platform_context=platform_context,
    ), system_prompt=platform_context)


def chat_about_error(
    error_text: str,
    previous_analysis: str,
    question: str,
    error_type: str,          # ← now required, passed from app.py
) -> str:
    if not question.strip():
        return "Please ask a question about the error."

    platform_context = get_platform_context(error_type)

    prompt = PromptTemplate(
        input_variables=[
            "error_text", "previous_analysis",
            "question", "error_type", "platform_context"
        ],
        template=CHAT_PROMPT
    )
    return _ai_core_invoke(prompt.format(
        error_text=error_text[:2000],
        previous_analysis=previous_analysis[:3000],
        question=question,
        error_type=error_type,
        platform_context=platform_context,
    ), system_prompt=platform_context)


def extract_text_from_image(image_file) -> str:
    """Extract text from SAP error screenshot using SAP AI Core Claude vision."""
    try:
        from PIL import Image

        image_file.seek(0)
        img_bytes = image_file.read()
        img_b64   = base64.b64encode(img_bytes).decode("utf-8")

        image_file.seek(0)
        img  = Image.open(image_file)
        fmt  = (img.format or "PNG").lower()
        mime = f"image/{fmt}"

        # Use GPT-4o deployment for vision — supports image_url format natively
        token   = _get_token()
        dep_url = f"{_AICORE_BASE}/v2/inference/deployments/{_VISION_DEPLOYMENT_ID}"
        body = {
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{img_b64}"}
                    },
                    {
                        "type": "text",
                        "text": (
                            "This is a screenshot of an SAP error message. "
                            "Extract ALL visible text exactly as shown. "
                            "Include error codes, stack traces, program names, "
                            "line numbers, and all technical details."
                        )
                    }
                ]
            }],
            "max_tokens": 1000,
        }
        r = _http.post(
            f"{dep_url}/v1/chat/completions",
            json=body,
            headers={
                "Authorization":     f"Bearer {token}",
                "AI-Resource-Group": _AICORE_RG,
                "Content-Type":      "application/json",
            },
            timeout=60,
        )
        r.raise_for_status()
        extracted = r.json()["choices"][0]["message"]["content"].strip()
        return extracted if extracted else "No text found in image."

    except Exception as e:
        err = str(e)
        # If vision is unsupported, return a clean message instead of a raw error
        if "image" in err.lower() or "vision" in err.lower() or "multimodal" in err.lower() or "400" in err:
            return "Could not read image: Vision not supported for this model — please paste the error text manually."
        return f"Could not read image: {err}\nPlease copy-paste the error text in the Analyze Error tab."


def get_embedding(text: str, model) -> list:
    """Generate a semantic embedding vector for the given text."""
    return model.encode(text[:1000], show_progress_bar=False).tolist()


def cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def get_severity_color(severity: str) -> str:    return {
        "CRITICAL": "#FF0000",
        "HIGH":     "#FF6B00",
        "MEDIUM":   "#FFB800",
        "LOW":      "#00B050",
        "UNKNOWN":  "#808080",
    }.get(severity, "#808080")


def get_severity_emoji(severity: str) -> str:
    return {
        "CRITICAL": "🔴",
        "HIGH":     "🟠",
        "MEDIUM":   "🟡",
        "LOW":      "🟢",
        "UNKNOWN":  "⚪",
    }.get(severity, "⚪")


def format_download_report(result: dict, chat_history: list) -> str:
    lines = [
        "=" * 60,
        "AI ERROR DETECTIVE — RESOLUTION REPORT",
        "=" * 60,
        f"Platform : {result.get('error_type', 'N/A')}",
        f"Severity : {result.get('severity',   'N/A')}",
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

    lines += [
        "=" * 60,
        "Generated by AI Error Detective",
        "Built with SAP AI Core GPT-5 + Streamlit",
        "=" * 60,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────
# PROMPTS  (defined after functions so
#           PLATFORM_CONTEXT is already set)
# ─────────────────────────────────────────────

ERROR_ANALYSIS_PROMPT = """
{platform_context}

A developer has submitted the following error on the {error_type} platform.
Analyze it strictly within this platform's context and respond in EXACT format:

---
## 🔴 SEVERITY
[Write ONLY one of: CRITICAL / HIGH / MEDIUM / LOW]
[One sentence explaining why, referencing {error_type} specifically]

## 📋 ERROR SUMMARY
[2-3 sentences explaining what this error means, in context of {error_type}]

## 🔍 ROOT CAUSE ANALYSIS
[Detailed explanation of WHY this error is happening]
[Mention exact {error_type} components, modules, configurations, or APIs involved]

## 🛠️ STEP-BY-STEP FIX
[Numbered steps — use {error_type}-specific transaction codes, CLI commands, config paths, or code]

## 💻 CORRECTED CODE
[Provide corrected {error_type}-specific code block if needed, else write "No code change required"]

## ⚡ QUICK TIPS
[3 bullet points to prevent this error in future — specific to {error_type}]

## 📚 SAP REFERENCES
[List relevant SAP Note topics and descriptions (e.g. "SAP Note ~2345678 — addresses X in {error_type}").
Note numbers are approximate — always verify in SAP Support Portal (support.sap.com).
Also include relevant transaction codes, official SAP documentation links, or Help Portal pages.]
---

ERROR DETAILS:
Platform: {error_type}
Error Message:
{error_text}

Base your response strictly on the error and the {error_type} platform. Do not hallucinate.
"""


CHAT_PROMPT = """
{platform_context}

You are an AI debugging assistant STRICTLY specialized in {error_type}.

The developer submitted this error on {error_type}:
{error_text}

Your previous analysis of this error:
{previous_analysis}

IMPORTANT RULES:
1. ONLY answer questions that are directly related to {error_type} or this specific error.
2. If the developer asks about a DIFFERENT SAP platform or a completely unrelated topic, respond:
   "I am currently in {error_type} mode. Please switch the platform in the sidebar or ask a {error_type}-related question."
3. Use {error_type}-specific terminology, transaction codes, CLI commands, and best practices in every answer.
4. Keep answers concise, technical, and actionable.

Developer's Follow-up Question: {question}
"""


QUICK_FIX_PROMPT = """
{platform_context}

You are an expert in {error_type}. Give ONLY a 3-step quick fix for this error.
Be direct. Use {error_type}-specific commands, transaction codes, or config changes.
No long explanation — just the 3 fix steps.

Platform: {error_type}
Error: {error_text}
"""
