# auth.py — User authentication for AI Error Detective
import bcrypt
import psycopg2
from database import get_connection


def hash_password(plain_password: str) -> str:
    """Hash a plain text password using bcrypt."""
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed: str) -> bool:
    """Check if a plain password matches the stored hash."""
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed.encode("utf-8"))


def register_user(username: str, password: str) -> dict:
    """
    Register a new user.
    Returns: { success: bool, message: str, user_id: int or None }
    """
    if not username.strip() or not password.strip():
        return {"success": False, "message": "Username and password cannot be empty.", "user_id": None}

    if len(password) < 6:
        return {"success": False, "message": "Password must be at least 6 characters.", "user_id": None}

    try:
        conn = get_connection()
        cur  = conn.cursor()

        password_hash = hash_password(password)

        cur.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
            (username.strip().lower(), password_hash)
        )
        user_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        return {"success": True, "message": "Account created successfully!", "user_id": user_id}

    except psycopg2.errors.UniqueViolation:
        return {"success": False, "message": "Username already exists. Please choose another.", "user_id": None}
    except Exception as e:
        return {"success": False, "message": f"Registration failed: {str(e)}", "user_id": None}


def login_user(username: str, password: str) -> dict:
    """
    Authenticate an existing user.
    Returns: { success: bool, message: str, user_id: int or None, username: str or None }
    """
    if not username.strip() or not password.strip():
        return {"success": False, "message": "Please enter username and password.", "user_id": None, "username": None}

    try:
        conn = get_connection()
        cur  = conn.cursor()

        cur.execute(
            "SELECT id, password_hash FROM users WHERE username = %s",
            (username.strip().lower(),)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            return {"success": False, "message": "Username not found.", "user_id": None, "username": None}

        user_id, password_hash = row

        if not verify_password(password, password_hash):
            return {"success": False, "message": "Incorrect password.", "user_id": None, "username": None}

        return {
            "success":  True,
            "message":  f"Welcome back, {username.strip()}! 👋",
            "user_id":  user_id,
            "username": username.strip().lower(),
        }

    except Exception as e:
        return {"success": False, "message": f"Login failed: {str(e)}", "user_id": None, "username": None}
