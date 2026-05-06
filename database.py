# database.py — PostgreSQL handler for AI Error Detective
import os
import random
import string
import psycopg2
import psycopg2.extras
from datetime import datetime

# ── Load .env only if running locally ────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _get(key: str, default: str = "") -> str:
    """
    Read a config value. Priority:
    1. st.secrets  (Streamlit Cloud)
    2. os.environ  (Railway, Heroku, Docker, local .env)
    3. default
    """
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, default)


DB_CONFIG = {
    "host":     _get("DB_HOST",     "localhost"),
    "port":     _get("DB_PORT",     "5432"),
    "dbname":   _get("DB_NAME",     "defaultdb"),
    "user":     _get("DB_USER",     "avnadmin"),
    "password": _get("DB_PASSWORD", ""),
    "sslmode":  _get("DB_SSLMODE",  "require"),
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


# ══════════════════════════════════════════════════════════
# INIT — create all tables
# ══════════════════════════════════════════════════════════
def init_db():
    """Create all required tables if they don't exist."""
    conn = get_connection()
    cur  = conn.cursor()

    # ── Users ─────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            SERIAL PRIMARY KEY,
            username      VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at    TIMESTAMP DEFAULT NOW()
        );
    """)

    # ── Error history ──────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS error_history (
            id              SERIAL PRIMARY KEY,
            user_id         INTEGER REFERENCES users(id) ON DELETE CASCADE,
            error_text      TEXT NOT NULL,
            analysis        TEXT NOT NULL,
            severity        VARCHAR(20),
            platform        VARCHAR(100),
            is_resolved     BOOLEAN DEFAULT FALSE,
            resolution_text TEXT,
            resolved_at      TIMESTAMP,
            shared_with_team BOOLEAN DEFAULT FALSE,
            created_at       TIMESTAMP DEFAULT NOW()
        );
    """)

    # ── Teams ──────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            id          SERIAL PRIMARY KEY,
            team_name   VARCHAR(100) NOT NULL,
            team_code   VARCHAR(20)  UNIQUE NOT NULL,
            created_by  INTEGER REFERENCES users(id) ON DELETE CASCADE,
            created_at  TIMESTAMP DEFAULT NOW()
        );
    """)

    # ── Team members ───────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS team_members (
            id         SERIAL PRIMARY KEY,
            team_id    INTEGER REFERENCES teams(id) ON DELETE CASCADE,
            user_id    INTEGER REFERENCES users(id) ON DELETE CASCADE,
            joined_at  TIMESTAMP DEFAULT NOW(),
            UNIQUE(team_id, user_id)
        );
    """)

    # ── Safely add new columns to error_history if upgrading ──
    for col, col_def in [
        ("is_resolved",      "BOOLEAN DEFAULT FALSE"),
        ("resolution_text",  "TEXT"),
        ("resolved_at",      "TIMESTAMP"),
        ("shared_with_team", "BOOLEAN DEFAULT FALSE"),
        ("embedding",        "TEXT"),
    ]:
        try:
            cur.execute(f"ALTER TABLE error_history ADD COLUMN IF NOT EXISTS {col} {col_def};")
        except Exception:
            pass

    # ── Match feedback ─────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS match_feedback (
            id               SERIAL PRIMARY KEY,
            user_id          INTEGER REFERENCES users(id) ON DELETE CASCADE,
            error_history_id INTEGER REFERENCES error_history(id) ON DELETE CASCADE,
            is_helpful       BOOLEAN NOT NULL,
            created_at       TIMESTAMP DEFAULT NOW()
        );
    """)

    conn.commit()
    cur.close()
    conn.close()


# ══════════════════════════════════════════════════════════
# ERROR HISTORY
# ══════════════════════════════════════════════════════════
def save_history(user_id: int, error_text: str, analysis: str,
                 severity: str, platform: str, embedding: list = None) -> int:
    """Save analyzed error. Keeps last 10 per user. Returns new row id."""
    import json
    conn = get_connection()
    cur  = conn.cursor()

    emb_json = json.dumps(embedding) if embedding else None

    cur.execute("""
        INSERT INTO error_history
            (user_id, error_text, analysis, severity, platform, embedding, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (user_id, error_text, analysis, severity, platform, emb_json, datetime.now()))

    new_id = cur.fetchone()[0]

    cur.execute("""
        DELETE FROM error_history
        WHERE user_id = %s
          AND id NOT IN (
              SELECT id FROM error_history
              WHERE user_id = %s
              ORDER BY created_at DESC
              LIMIT 10
          )
    """, (user_id, user_id))

    conn.commit()
    cur.close()
    conn.close()
    return new_id


def mark_resolved(record_id: int, resolution_text: str, share_with_team: bool = False):
    """Mark an error as resolved. Optionally share with team."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        UPDATE error_history
        SET is_resolved      = TRUE,
            resolution_text  = %s,
            resolved_at      = %s,
            shared_with_team = %s
        WHERE id = %s
    """, (resolution_text, datetime.now(), share_with_team, record_id))
    conn.commit()
    cur.close()
    conn.close()


def get_history(user_id: int) -> list:
    """Fetch last 10 resolved errors for the user."""
    conn = get_connection()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT id, error_text, analysis, severity, platform,
               is_resolved, resolution_text, resolved_at,
               shared_with_team, created_at
        FROM error_history
        WHERE user_id = %s
          AND is_resolved = TRUE
        ORDER BY resolved_at DESC
        LIMIT 10
    """, (user_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return list(reversed([dict(r) for r in rows]))


def save_match_feedback(user_id: int, error_history_id: int, is_helpful: bool):
    """Save user feedback on a similar error match."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO match_feedback (user_id, error_history_id, is_helpful, created_at)
        VALUES (%s, %s, %s, %s)
    """, (user_id, error_history_id, is_helpful, datetime.now()))
    conn.commit()
    cur.close()
    conn.close()


def get_similar_personal_error(user_id: int, platform: str, query_embedding: list,
                               similarity_fn, threshold: float = 0.65,
                               exclude_id: int = None) -> dict | None:
    """
    Semantic similarity search within the user's own resolved errors.
    Used to remind the user of a fix they already found before.
    """
    import json

    conn = get_connection()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    sql = """
        SELECT id, error_text, resolution_text, platform, resolved_at, embedding
        FROM error_history
        WHERE user_id = %s
          AND is_resolved = TRUE
          AND platform = %s
          AND embedding IS NOT NULL
    """
    params = [user_id, platform]

    if exclude_id:
        sql += " AND id != %s"
        params.append(exclude_id)

    sql += " ORDER BY resolved_at DESC LIMIT 50"

    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    best_match = None
    best_score = threshold

    for row in rows:
        row_dict   = dict(row)
        stored_emb = json.loads(row_dict.pop("embedding"))
        score      = similarity_fn(query_embedding, stored_emb)
        if score > best_score:
            best_score = score
            best_match = row_dict

    return best_match


def get_user_trends(user_id: int) -> dict:
    """Fetch aggregated trend data for the user's error history."""
    conn = get_connection()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Overall stats
    cur.execute("""
        SELECT
            COUNT(*)                                                AS total,
            COUNT(CASE WHEN is_resolved = TRUE  THEN 1 END)        AS resolved,
            COUNT(CASE WHEN severity = 'CRITICAL' THEN 1 END)      AS critical
        FROM error_history WHERE user_id = %s
    """, (user_id,))
    stats = dict(cur.fetchone())

    # By platform
    cur.execute("""
        SELECT platform, COUNT(*) AS count
        FROM error_history WHERE user_id = %s
        GROUP BY platform ORDER BY count DESC
    """, (user_id,))
    by_platform = [dict(r) for r in cur.fetchall()]

    # By severity
    cur.execute("""
        SELECT severity, COUNT(*) AS count
        FROM error_history WHERE user_id = %s
        GROUP BY severity ORDER BY count DESC
    """, (user_id,))
    by_severity = [dict(r) for r in cur.fetchall()]

    # By day of week
    cur.execute("""
        SELECT EXTRACT(DOW FROM created_at) AS day_num,
               TO_CHAR(created_at, 'Dy')   AS day,
               COUNT(*)                    AS count
        FROM error_history WHERE user_id = %s
        GROUP BY day_num, day ORDER BY day_num
    """, (user_id,))
    by_day = [dict(r) for r in cur.fetchall()]

    # Monthly trend (last 6 months)
    cur.execute("""
        SELECT TO_CHAR(DATE_TRUNC('month', created_at), 'Mon YYYY') AS month,
               DATE_TRUNC('month', created_at)                      AS month_date,
               COUNT(*)                                             AS count
        FROM error_history
        WHERE user_id = %s AND created_at >= NOW() - INTERVAL '6 months'
        GROUP BY month, month_date ORDER BY month_date
    """, (user_id,))
    by_month = [dict(r) for r in cur.fetchall()]

    cur.close()
    conn.close()
    return {
        "stats":       stats,
        "by_platform": by_platform,
        "by_severity": by_severity,
        "by_day":      by_day,
        "by_month":    by_month,
    }


# ══════════════════════════════════════════════════════════
# TEAMS
# ══════════════════════════════════════════════════════════
def _generate_team_code() -> str:
    """Generate a unique 8-char alphanumeric team code e.g. SAP-A3X9."""
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"SAP-{suffix}"


def create_team(team_name: str, user_id: int) -> dict:
    """
    Create a new team. Creator is automatically added as first member.
    Returns: { success, message, team_code }
    """
    if not team_name.strip():
        return {"success": False, "message": "Team name cannot be empty.", "team_code": None}

    conn = get_connection()
    cur  = conn.cursor()

    # Generate unique team code
    for _ in range(10):
        code = _generate_team_code()
        cur.execute("SELECT id FROM teams WHERE team_code = %s", (code,))
        if not cur.fetchone():
            break

    try:
        cur.execute("""
            INSERT INTO teams (team_name, team_code, created_by, created_at)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """, (team_name.strip(), code, user_id, datetime.now()))
        team_id = cur.fetchone()[0]

        # Auto-add creator as member
        cur.execute("""
            INSERT INTO team_members (team_id, user_id, joined_at)
            VALUES (%s, %s, %s)
        """, (team_id, user_id, datetime.now()))

        conn.commit()
        cur.close()
        conn.close()
        return {"success": True, "message": f"Team '{team_name}' created!", "team_code": code}

    except Exception as e:
        conn.rollback()
        cur.close()
        conn.close()
        return {"success": False, "message": str(e), "team_code": None}


def join_team(team_code: str, user_id: int) -> dict:
    """
    Join an existing team by code.
    Returns: { success, message, team_name }
    """
    if not team_code.strip():
        return {"success": False, "message": "Please enter a team code.", "team_name": None}

    conn = get_connection()
    cur  = conn.cursor()

    cur.execute("SELECT id, team_name FROM teams WHERE team_code = %s", (team_code.strip().upper(),))
    row = cur.fetchone()

    if not row:
        cur.close()
        conn.close()
        return {"success": False, "message": "Team code not found. Please check and try again.", "team_name": None}

    team_id, team_name = row

    # Check if already a member
    cur.execute("SELECT id FROM team_members WHERE team_id = %s AND user_id = %s", (team_id, user_id))
    if cur.fetchone():
        cur.close()
        conn.close()
        return {"success": False, "message": f"You are already a member of '{team_name}'.", "team_name": team_name}

    try:
        cur.execute("""
            INSERT INTO team_members (team_id, user_id, joined_at)
            VALUES (%s, %s, %s)
        """, (team_id, user_id, datetime.now()))
        conn.commit()
        cur.close()
        conn.close()
        return {"success": True, "message": f"Joined team '{team_name}' successfully!", "team_name": team_name}
    except Exception as e:
        conn.rollback()
        cur.close()
        conn.close()
        return {"success": False, "message": str(e), "team_name": None}


def leave_team(team_id: int, user_id: int) -> dict:
    """Leave a team."""
    conn = get_connection()
    cur  = conn.cursor()
    try:
        cur.execute("DELETE FROM team_members WHERE team_id = %s AND user_id = %s", (team_id, user_id))
        conn.commit()
        cur.close()
        conn.close()
        return {"success": True}
    except Exception as e:
        conn.rollback()
        cur.close()
        conn.close()
        return {"success": False, "message": str(e)}


def get_user_teams(user_id: int) -> list:
    """
    Get all teams the user belongs to.
    Returns list of { team_id, team_name, team_code, created_by, member_count }
    """
    conn = get_connection()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT t.id AS team_id, t.team_name, t.team_code, t.created_by,
               COUNT(tm2.user_id) AS member_count
        FROM teams t
        JOIN team_members tm ON tm.team_id = t.id AND tm.user_id = %s
        JOIN team_members tm2 ON tm2.team_id = t.id
        GROUP BY t.id, t.team_name, t.team_code, t.created_by
        ORDER BY t.created_at DESC
    """, (user_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def get_team_members(team_id: int) -> list:
    """Get all members of a team with their username and join date."""
    conn = get_connection()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT u.username, tm.joined_at, t.created_by = u.id AS is_admin
        FROM team_members tm
        JOIN users u ON u.id = tm.user_id
        JOIN teams t ON t.id = tm.team_id
        WHERE tm.team_id = %s
        ORDER BY tm.joined_at ASC
    """, (team_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def get_team_resolved_errors(team_id: int, limit: int = 20) -> list:
    """
    Get resolved errors from all members of a team.
    Only resolved errors are shared — private unresolved errors stay private.
    """
    conn = get_connection()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT eh.id, eh.error_text, eh.analysis, eh.severity, eh.platform,
               eh.resolution_text, eh.resolved_at, eh.created_at,
               u.username AS resolved_by
        FROM error_history eh
        JOIN users u ON u.id = eh.user_id
        JOIN team_members tm ON tm.user_id = eh.user_id AND tm.team_id = %s
        WHERE eh.is_resolved = TRUE
          AND eh.shared_with_team = TRUE
        ORDER BY eh.resolved_at DESC
        LIMIT %s
    """, (team_id, limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def get_similar_team_error(team_id: int, platform: str, query_embedding: list,
                           similarity_fn, threshold: float = 0.65) -> dict | None:
    """
    Semantic similarity search within team's resolved error pool.
    Compares query_embedding against stored embeddings using cosine similarity.
    Returns the best match above threshold or None.
    """
    import json

    conn = get_connection()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT eh.id, eh.error_text, eh.resolution_text, eh.platform,
               eh.resolved_at, eh.embedding,
               u.username AS resolved_by,
               COUNT(CASE WHEN mf.is_helpful = TRUE  THEN 1 END) AS helpful_count,
               COUNT(CASE WHEN mf.is_helpful = FALSE THEN 1 END) AS unhelpful_count
        FROM error_history eh
        JOIN users u ON u.id = eh.user_id
        JOIN team_members tm ON tm.user_id = eh.user_id AND tm.team_id = %s
        LEFT JOIN match_feedback mf ON mf.error_history_id = eh.id
        WHERE eh.is_resolved = TRUE
          AND eh.shared_with_team = TRUE
          AND eh.platform = %s
          AND eh.embedding IS NOT NULL
        GROUP BY eh.id, eh.error_text, eh.resolution_text, eh.platform,
                 eh.resolved_at, eh.embedding, u.username
        ORDER BY eh.resolved_at DESC
        LIMIT 50
    """, (team_id, platform))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    best_match = None
    best_score = threshold

    for row in rows:
        row_dict = dict(row)
        stored_emb      = json.loads(row_dict.pop("embedding"))
        helpful_count   = row_dict.pop("helpful_count", 0) or 0
        unhelpful_count = row_dict.pop("unhelpful_count", 0) or 0

        base_score     = similarity_fn(query_embedding, stored_emb)
        feedback_boost = (helpful_count - unhelpful_count) * 0.05
        adjusted_score = base_score + feedback_boost

        if adjusted_score > best_score:
            best_score = adjusted_score
            best_match = row_dict

    return best_match