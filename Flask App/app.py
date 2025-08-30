import json 
import re
import requests
from collections import Counter
import os
import threading
from datetime import datetime, timezone, timedelta
from math import isfinite
from functools import wraps
from flask import request, redirect, url_for, render_template, flash
from models import SessionLocal, init_db  # make sure this import works


from flask import Flask, render_template, redirect, url_for, request, flash, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, case
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("FLASK_SECRET_KEY", "dev-secret")
with app.app_context():
    init_db()

# Basic session hardening (safe defaults)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=not app.debug,  # set True in prod
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
)
def _utcnow():
    return datetime.now(timezone.utc)

def _has_archived_at(ResponseModel) -> bool:
    return hasattr(ResponseModel, "archived_at")
# ---------------------------
# Database (Settings + new models)
# ---------------------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///weeklybuddy.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    bot_name = db.Column(db.String(100), default="WeeklyBuddy")
    theme = db.Column(db.String(20), default="Light")
    meeting_frequency = db.Column(db.String(20), default="Weekly")
    notifications = db.Column(db.Boolean, default=True)
    deploy_slack = db.Column(db.Boolean, default=True)
    deploy_discord = db.Column(db.Boolean, default=False)
    deploy_email = db.Column(db.Boolean, default=False)
    deploy_web = db.Column(db.Boolean, default=True)

class MeetingPolicy(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    blocker_threshold = db.Column(db.Integer, default=5)
    stale_days_threshold = db.Column(db.Integer, default=7)
    notes = db.Column(db.Text, default="")
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(500), nullable=False)
    position = db.Column(db.Integer, default=0)
    active = db.Column(db.Boolean, default=True)
    tag = db.Column(db.String(64), default="")

with app.app_context():
    db.create_all()
    if not Settings.query.first():
        db.session.add(Settings()); db.session.commit()
    if not MeetingPolicy.query.first():
        db.session.add(MeetingPolicy()); db.session.commit()
    if Question.query.count() == 0:
        db.session.add_all([
            Question(text="1️⃣ What did you work on this week?", position=1, active=True, tag="status"),
            Question(text="2️⃣ What are you planning next?", position=2, active=True, tag="plan"),
            Question(text="3️⃣ Any blockers?", position=3, active=True, tag="blockers"),
        ])
        db.session.commit()

# ---------------------------
# Global bot state (dashboard toggle)
# ---------------------------
BOT_THREAD = None
BOT_HANDLER = None
BOT_RUNNING = False
BOT_LOCK = threading.Lock()

@app.context_processor
def inject_globals():
    s = Settings.query.first()
    # expose the API URL safely to Jinja
    try:
        theme_api = url_for("api_theme")
    except Exception:
        theme_api = "/api/theme"
    return dict(
        settings=s,
        bot_running=BOT_RUNNING,
        authed=session.get("authed", False),
        theme_api=theme_api,
    )


# ---------------------------
# Auth
# ---------------------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authed"):
            nxt = request.full_path if request.query_string else request.path
            return redirect(url_for("login", next=nxt))
        return f(*args, **kwargs)
    return decorated

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        password = request.form.get("password", "")
        expected = os.environ.get("ADMIN_PASSWORD", "admin")
        if password == expected:
            session["authed"] = True
            next_url = request.args.get("next") or url_for("dashboard")
            return redirect(next_url)
        flash("Invalid password", "danger")
    return render_template("login.html")

@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------------------
# Slack bot controls (reuse weeklybuddy.py unchanged)
# ---------------------------
def _create_handler_from_weeklybuddy():
    try:
        import weeklybuddy as wb  # your bot script
        from slack_bolt.adapter.socket_mode import SocketModeHandler
    except Exception as e:
        print(f"[bot] import failed: {e}")
        raise
    app_token = os.getenv("SLACK_APP_TOKEN") or getattr(wb, "SLACK_APP_TOKEN", None)
    if not app_token:
        raise RuntimeError("Missing SLACK_APP_TOKEN")
    return SocketModeHandler(wb.app, app_token)

def _bot_thread_target():
    global BOT_HANDLER, BOT_RUNNING
    try:
        BOT_HANDLER = _create_handler_from_weeklybuddy()
        BOT_RUNNING = True
        print("🤖 WeeklyBuddy bot starting (Socket Mode)…")
        BOT_HANDLER.start()
    except Exception as e:
        BOT_RUNNING = False
        print(f"[bot] stopped with error: {e}")

@app.post("/bot/start")
@login_required
def bot_start():
    global BOT_THREAD, BOT_RUNNING
    with BOT_LOCK:
        if BOT_RUNNING:
            return ("", 204)
        BOT_THREAD = threading.Thread(target=_bot_thread_target, daemon=True)
        BOT_THREAD.start()
        return ("", 204)

@app.post("/bot/stop")
@login_required
def bot_stop():
    global BOT_HANDLER, BOT_RUNNING, BOT_THREAD
    with BOT_LOCK:
        if not BOT_RUNNING or not BOT_HANDLER:
            return ("", 204)
        try:
            close_fn = getattr(BOT_HANDLER, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception as e:
            print(f"[bot] stop failed: {e}")
        BOT_RUNNING = False
        BOT_HANDLER = None
        BOT_THREAD = None
        return ("", 204)

# ---------------------------
# JSON fallback (sample data)
# ---------------------------
def load_responses_json():
    path = os.path.join(os.path.dirname(__file__), "responses.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)

# ---------------------------
# Prefer DB (models.py) else JSON
# ---------------------------
def load_any_responses():
    """
    Returns (list_of_entries, using_live_db: bool)
      entry: { user_id, name, answers:[worked_on,next_up,blockers], ts: datetime|None, resp_id: int|None }
    """
    try:
        from models import SessionLocal, User, Response
        sess = SessionLocal()
        try:
            rows = (
                sess.query(User, Response)
                .join(Response, Response.user_id == User.id)
                .order_by(User.id.asc(), Response.id.desc())
                .all()
            )
            seen = set()
            out = []
            for user, resp in rows:
                if user.id in seen:
                    continue
                seen.add(user.id)
                ts = None
                if hasattr(resp, "created_at") and resp.created_at:
                    ts = resp.created_at if resp.created_at.tzinfo else resp.created_at.replace(tzinfo=timezone.utc)
                out.append({
                    "user_id": getattr(user, "slack_id", None) or getattr(user, "id", None),
                    "name": getattr(user, "name", None),
                    "answers": [
                        (resp.worked_on or "No response"),
                        (resp.next_up or "No response"),
                        (resp.blockers or "No response"),
                    ],
                    "ts": ts,
                    "resp_id": getattr(resp, "id", None),
                })
            if out:
                return out, True
        finally:
            sess.close()
    except Exception:
        pass

    out = []
    for r in load_responses_json():
        answers = r.get("answers", [])
        ts = None
        raw_ts = r.get("timestamp")
        if raw_ts:
            try:
                ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            except Exception:
                ts = None
        out.append({
            "user_id": r.get("user_id") or r.get("user"),
            "name": r.get("name"),
            "answers": [
                answers[0] if len(answers) > 0 else "No response",
                answers[1] if len(answers) > 1 else "No response",
                answers[2] if len(answers) > 2 else "No response",
            ],
            "ts": ts,
            "resp_id": None,
        })
    return out, False

# ---------------------------
# Trends data
# ---------------------------
def build_trends_data(fallback_users):
    try:
        from models import SessionLocal, User, Response
        sess = SessionLocal()
        try:
            results = (
                sess.query(User, Response)
                .join(Response, Response.user_id == User.id)
                .all()
            )
            counts = {}
            for user, resp in results:
                b = (resp.blockers or "").strip().lower()
                if b not in ["", "no blockers", "no response", "none", "n/a"]:
                    key = user.name or getattr(user, "slack_id", None) or f"User {user.id}"
                    counts[key] = counts.get(key, 0) + 1
            if counts:
                return counts
        finally:
            sess.close()
    except Exception:
        pass

    counts = {}
    for u in fallback_users:
        b = (u["answers"][2] or "").strip().lower()
        if b not in ["", "no blockers", "no response", "none", "n/a"]:
            key = u["name"] or u["user_id"] or "Unknown"
            counts[key] = counts.get(key, 0) + 1
    return counts

# ---------------------------
# Weekly time series (last 8 weeks)
# ---------------------------
def compute_weekly_blockers_series():
    end = datetime.now(timezone.utc)
    start = end - timedelta(weeks=7)
    week_starts = []
    cur = (start - timedelta(days=start.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    while cur <= end:
        week_starts.append(cur)
        cur = cur + timedelta(weeks=1)
    labels = [ws.strftime("Wk %V — %b %d") for ws in week_starts]
    counts = [0] * len(week_starts)

    used_db = False
    try:
        from models import SessionLocal, Response
        sess = SessionLocal()
        try:
            if hasattr(Response, "created_at"):
                rows = sess.query(Response).all()
                for resp in rows:
                    btxt = (resp.blockers or "").strip().lower()
                    if btxt in ["", "no blockers", "no response", "none", "n/a"]:
                        continue
                    ts = resp.created_at
                    if not ts:
                        continue
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    for i, ws in enumerate(week_starts):
                        we = ws + timedelta(weeks=1)
                        if ws <= ts < we:
                            counts[i] += 1
                            break
                used_db = True
        finally:
            sess.close()
    except Exception:
        pass

    if not used_db:
        data, _ = load_any_responses()
        for u in data:
            btxt = (u["answers"][2] or "").strip().lower()
            ts = u.get("ts")
            if btxt in ["", "no blockers", "no response", "none", "n/a"] or not ts:
                continue
            for i, ws in enumerate(week_starts):
                we = ws + timedelta(weeks=1)
                if ws <= ts.replace(tzinfo=timezone.utc) < we:
                    counts[i] += 1
                    break

    return labels, counts

# ---------------------------
# Helpers / shared
# ---------------------------
def _is_blocker_text(txt: str) -> bool:
    if not txt:
        return False
    t = txt.strip().lower()
    return t not in ("", "no blockers", "no response", "none", "n/a")

def _tz_aware(ts):
    if not ts:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)

def _human_date(dt):
    if not dt:
        return '—'
    now = datetime.now(timezone.utc) if dt.tzinfo else datetime.utcnow()
    delta = now - (dt if dt.tzinfo else dt.replace(tzinfo=None))
    days = delta.days
    if days == 0:
        return 'Today'
    if days == 1:
        return 'Yesterday'
    if days < 7:
        return f"{days}d ago"
    return dt.strftime('%b %d')

def _normalize_status(s: str) -> str:
    return (s or "").strip().lower().replace("_", " ").replace("-", " ")

# ---------------------------
# Dashboard helpers
# ---------------------------
def compute_dashboard_metrics(users, total_blockers, stale_days_threshold=7):
    team_size = len(users)
    submitted_latest = 0
    for u in users:
        answers = u.get("answers") or []
        if any(a and a != "No response" for a in answers):
            submitted_latest += 1
    active_this_week = 0
    stale_users = []
    now_utc = datetime.now(timezone.utc)
    for u in users:
        ts = u.get("ts")
        if ts and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts:
            days = (now_utc - ts).days
            if days <= 7:
                active_this_week += 1
            if days > stale_days_threshold:
                stale_users.append(u)
    blocker_rate = (total_blockers / team_size) if team_size else 0.0
    return {
        "team_size": team_size,
        "submitted_latest": submitted_latest,
        "active_this_week": active_this_week,
        "stale_users_count": len(stale_users),
        "stale_users": stale_users,
        "blocker_rate": round(blocker_rate, 3),
    }

def generate_ai_suggestions(users, total_blockers, meeting_recommendation, metrics, week_labels, week_counts, policy, settings):
    suggestions = []
    suggestions.append(meeting_recommendation)
    if week_counts and len(week_counts) >= 3:
        if week_counts[-3] < week_counts[-2] < week_counts[-1]:
            suggestions.append("📈 Blockers have risen 3 weeks in a row — consider a focused unblocking sync.")
    blocker_users = []
    for u in users:
        blockers = (u.get("answers", ["", "", ""])[2] or "")
        if _is_blocker_text(blockers):
            blocker_users.append((u.get("name") or u.get("user_id") or "Unknown", blockers))
    if blocker_users:
        top = blocker_users[:3]
        pretty = "; ".join([f"{n}: {b[:80]}{'…' if len(b) > 80 else ''}" for n, b in top])
        suggestions.append(f"🧱 Current blockers — reach out to: {pretty}.")
    if metrics["stale_users_count"] > 0:
        names = ", ".join([(u.get("name") or u.get("user_id") or "Unknown") for u in metrics["stale_users"][:5]])
        suggestions.append(f"⏱️ {metrics['stale_users_count']} teammate(s) look stale (> {policy.stale_days_threshold}d): {names}. Give them a nudge.")
    rate = metrics["blocker_rate"]
    if isfinite(rate):
        if rate >= 0.5 and settings.meeting_frequency != "Daily":
            suggestions.append("🗓️ >50% with blockers — temporarily increase cadence (e.g., Daily) until stable.")
        elif rate == 0 and settings.meeting_frequency in ("Daily", "Bi-weekly"):
            suggestions.append("🌤️ No blockers — maintain async updates and trim sync time this cycle.")
    suggestions.append("🧩 Tip: tag blockers by area (backend/data/ops) and auto-assign owners; triage tagged items first.")
    return suggestions

# ---------------------------
# Member classification
# ---------------------------
def _classify_member(latest_resp, resp_cnt: int, block_cnt: int, policy):
    last_ts = _tz_aware(getattr(latest_resp, "created_at", None))
    stale_days_threshold = (policy.stale_days_threshold if policy else 7)
    now = datetime.now(timezone.utc)
    stale = (not last_ts) or ((now - last_ts).days > stale_days_threshold)

    blockers_text = (getattr(latest_resp, "blockers", "") or "")
    has_blocker = _is_blocker_text(blockers_text)

    if has_blocker:
        status, tone = "Blocked", "danger"
    elif stale:
        status, tone = "Needs Check-in", "warning"
    else:
        status, tone = "On Track", "success"

    base = min(100, resp_cnt * 12)
    penalty = min(60, block_cnt * 15)
    bonus = 10 if status == "On Track" else 0
    progress = max(5, min(100, base - penalty + bonus))
    return status, tone, int(progress), last_ts

def generate_people_insights(members: list, policy) -> list:
    insights = []
    blocked = [m for m in members if _normalize_status(m["status"]) == "blocked"]
    stale = [m for m in members if _normalize_status(m["status"]) == "needs check in"]

    if blocked:
        names = ", ".join((m["name"] or str(m["user_id"])) for m in blocked[:5])
        insights.append(f"🧱 {len(blocked)} teammate(s) blocked: {names}. Run a quick async unblocking thread.")
    if stale:
        names = ", ".join((m["name"] or str(m["user_id"])) for m in stale[:5])
        insights.append(f"⏱️ {len(stale)} need a check-in (> {policy.stale_days_threshold}d): {names}.")
    top_resp = sorted(members, key=lambda m: m["responses_count"], reverse=True)[:3]
    if top_resp:
        insights.append("🏅 Top responders: " + ", ".join(f'{m["name"] or m["user_id"]} ({m["responses_count"]})' for m in top_resp))
    if not insights:
        insights.append("✅ Team looks steady. Keep cadence and keep tagging blockers for fast triage.")
    return insights

# ---------------------------
# Summarizer
# ---------------------------
def summarize(responses_like):
    total_blockers = 0
    users = []
    for r in responses_like:
        worked_on, next_up, blockers = r["answers"]
        btxt = (blockers or "").lower()
        if btxt not in ["no blockers", "no response", "none", "n/a", ""]:
            total_blockers += 1
        users.append({
            "user_id": r.get("user_id"),
            "name": r.get("name"),
            "answers": [worked_on, next_up, blockers],
            "ts": r.get("ts"),
            "resp_id": r.get("resp_id"),
        })
    policy = MeetingPolicy.query.first()
    threshold = policy.blocker_threshold if policy else 5
    if total_blockers >= threshold:
        meeting_recommendation = f"🚨 {total_blockers} blockers ≥ threshold ({threshold}). Recommend an extra meeting."
    elif total_blockers > 0:
        meeting_recommendation = f"⚠️ {total_blockers} blocker(s). Monitor and consider a check-in."
    else:
        meeting_recommendation = "✅ No blockers — no extra meeting needed."
    return users, total_blockers, meeting_recommendation

# ============================================================
# Metrics for a member (manager-facing)
# ============================================================
def compute_member_metrics(responses):
    """
    responses: list[Response]
    Returns: dict (total, streak_weeks, avg_interval_days, blocker_rate,
                   last_blocker_days_ago, avg_answer_len)
    """
    if not responses:
        return {
            "total": 0,
            "streak_weeks": 0,
            "avg_interval_days": None,
            "blocker_rate": 0.0,
            "last_blocker_days_ago": None,
            "avg_answer_len": 0,
        }

    ts = []
    for r in responses:
        t = getattr(r, "created_at", None)
        if t:
            ts.append(_tz_aware(t))
    ts = sorted([t for t in ts if t is not None])

    if len(ts) >= 2:
        diffs = [(t2 - t1).days for t1, t2 in zip(ts, ts[1:])]
        avg_interval = round(sum(diffs) / len(diffs), 1)
    else:
        avg_interval = None

    seen_weeks = set()
    for t in ts:
        y, w, _ = t.isocalendar()
        seen_weeks.add((y, w))
    now = datetime.now(timezone.utc)
    streak = 0
    for i in range(0, 26):
        wdt = now - timedelta(weeks=i)
        key = wdt.isocalendar()[:2]
        if key in seen_weeks:
            streak += 1
        else:
            break

    def _has_blocker(r): return _is_blocker_text(getattr(r, "blockers", "") or "")
    total = len(responses)
    blockers = sum(1 for r in responses if _has_blocker(r))
    blocker_rate = round(blockers / total, 2) if total else 0.0

    last_blocker_ts = None
    for r in sorted(responses, key=lambda x: _tz_aware(getattr(x, "created_at", None)) or datetime.min.replace(tzinfo=timezone.utc), reverse=True):
        if _has_blocker(r):
            last_blocker_ts = _tz_aware(getattr(r, "created_at", None))
            break
    last_blocker_days_ago = (datetime.now(timezone.utc) - last_blocker_ts).days if last_blocker_ts else None

    def _len(s): return len((s or "").strip())
    total_len = sum((_len(r.worked_on) + _len(r.next_up) + _len(r.blockers)) for r in responses)
    avg_answer_len = int(total_len / total) if total else 0

    return {
        "total": total,
        "streak_weeks": streak,
        "avg_interval_days": avg_interval,
        "blocker_rate": blocker_rate,
        "last_blocker_days_ago": last_blocker_days_ago,
        "avg_answer_len": avg_answer_len,
    }

# Slack helpers for nudges
def _slack_client():
    from slack_sdk import WebClient
    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        raise RuntimeError("Missing SLACK_BOT_TOKEN")
    return WebClient(token=token)

def _open_dm(client, slack_user_id):
    dm = client.conversations_open(users=slack_user_id)
    return dm["channel"]["id"]

# ---------------------------
# Pages
# ---------------------------
@app.get("/")
@login_required
def root_redirect():
    return redirect(url_for("dashboard"))

@app.get("/dashboard")
@login_required
def dashboard():
    data, _ = load_any_responses()
    users, total_blockers, meeting_recommendation = summarize(data)
    week_labels, week_counts = compute_weekly_blockers_series()
    policy = MeetingPolicy.query.first()
    settings_row = Settings.query.first()
    metrics = compute_dashboard_metrics(
        users, total_blockers,
        stale_days_threshold=(policy.stale_days_threshold if policy else 7)
    )
    suggestions = generate_ai_suggestions(
        users, total_blockers, meeting_recommendation, metrics,
        week_labels, week_counts,
        policy or MeetingPolicy(blocker_threshold=5, stale_days_threshold=7),
        settings_row or Settings(meeting_frequency="Weekly")
    )
    return render_template(
        "dashboard.html",
        users=users,
        total_blockers=total_blockers,
        meeting_recommendation=meeting_recommendation,
        week_labels=week_labels,
        week_counts=week_counts,
        metrics=metrics,
        suggestions=suggestions
    )

@app.get("/trend")
@login_required
def trend():
    # timeframe: ?w=4|8|12 (default 8)
    try:
        w = int(request.args.get("w", "8"))
    except Exception:
        w = 8
    w = max(4, min(12, w))

    tr = build_trends_advanced(timeframe_weeks=w)
    ai_tips = generate_trend_ai_insights(tr)

    return render_template(
        "trends.html",
        tf_weeks=w,
        tr=tr,
        ai_tips=ai_tips
    )



@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    s = Settings.query.first()
    if request.method == "POST":
        s.bot_name = request.form.get("bot_name", "WeeklyBuddy")
        s.theme = request.form.get("theme", "Light")
        s.meeting_frequency = request.form.get("meeting_frequency", "Weekly")
        s.notifications = "notifications" in request.form
        s.deploy_slack = "deploy_slack" in request.form
        s.deploy_discord = "deploy_discord" in request.form
        s.deploy_email = "deploy_email" in request.form
        s.deploy_web = "deploy_web" in request.form
        db.session.commit()
        flash("✅ Settings saved", "success")
        return redirect(url_for("settings"))
    return render_template("settings.html", settings=s)

@app.route("/api/theme", methods=["POST"], endpoint="api_theme")
@login_required
def api_theme():
    data = request.get_json(silent=True) or {}
    mode = (data.get("theme") or "").capitalize()
    if mode not in ("Light", "Dark", "Auto"):
        return {"ok": False, "error": "bad theme"}, 400

    s = Settings.query.first() or Settings()
    db.session.add(s)
    s.theme = mode
    db.session.commit()
    return {"ok": True, "theme": mode}


@app.route("/context", methods=["GET", "POST"])
@login_required
def context():
    policy = MeetingPolicy.query.first()
    if request.method == "POST":
        policy.blocker_threshold = int(request.form.get("blocker_threshold", policy.blocker_threshold or 5))
        policy.stale_days_threshold = int(request.form.get("stale_days_threshold", policy.stale_days_threshold or 7))
        policy.notes = request.form.get("notes", policy.notes or "")
        db.session.commit()
        flash("✅ Context policy saved", "success")
        return redirect(url_for("context"))
    return render_template("context.html", policy=policy)

@app.route("/questions", methods=["GET", "POST"])
@login_required
def questions():
    if request.method == "POST":
        action = request.form.get("action")
        if action == "add":
            text = request.form.get("text", "").strip()
            if text:
                max_pos = db.session.query(db.func.max(Question.position)).scalar() or 0
                db.session.add(Question(text=text, position=max_pos + 1, active=True,
                                        tag=request.form.get("category", "").strip()))
                db.session.commit()
                flash("✅ Question added", "success")
        elif action == "update":
            qid = request.form.get("id")
            q = Question.query.get(int(qid)) if qid else None
            if q:
                q.text = request.form.get("text", q.text)
                q.position = int(request.form.get("position", q.position or 0))
                q.active = (request.form.get("active") == "on") or (request.form.get("active") == "true")
                q.tag = request.form.get("category", q.tag or "")
                db.session.commit()
                flash("✅ Question updated", "success")
        elif action == "delete":
            qid = request.form.get("id")
            q = Question.query.get(int(qid)) if qid else None
            if q:
                db.session.delete(q)
                db.session.commit()
                flash("🗑️ Question deleted", "warning")
            else:
                flash("Question not found", "danger")
        return redirect(url_for("questions"))

    qs = Question.query.order_by(Question.position.asc(), Question.id.asc()).all()
    return render_template("questions.html", questions=qs)

# ---------------------------
# Tag / Assign (kept for compatibility; safe to ignore if unused)
# ---------------------------
@app.route("/tag/<int:response_id>", methods=["GET", "POST"])
@login_required
def tag_response(response_id):
    try:
        from models import SessionLocal, User, Response, Tag, ResponseTag, init_db
        init_db()
        sess = SessionLocal()
    except Exception:
        flash("Tagging requires DB tables (Tag, ResponseTag) in models.py.", "warning")
        return redirect(url_for("dashboard"))

    try:
        resp = sess.query(Response).filter_by(id=response_id).first()
        if not resp:
            flash("Response not found", "warning")
            return redirect(url_for("dashboard"))
        users = sess.query(User).all()
        all_tags = sess.query(Tag).order_by(Tag.name.asc()).all()

        if request.method == "POST":
            new_tags_raw = request.form.get("new_tags", "")
            for name in [t.strip() for t in new_tags_raw.split(",") if t.strip()]:
                existing = sess.query(Tag).filter(Tag.name.ilike(name)).first()
                if not existing:
                    sess.add(Tag(name=name))
            sess.commit()
            tag_ids = request.form.getlist("tag_ids")
            assignees = request.form.getlist("assignees")
            sess.query(ResponseTag).filter_by(response_id=response_id).delete()
            sess.commit()
            for tid in tag_ids:
                sess.add(ResponseTag(response_id=response_id, tag_id=int(tid)))
            for uid in assignees:
                sess.add(ResponseTag(response_id=response_id, tagged_user_id=int(uid)))
            sess.commit()
            flash("✅ Tags/assignees saved", "success")
            return redirect(url_for("dashboard"))

        existing = sess.query(ResponseTag).filter_by(response_id=response_id).all()
        selected_tag_ids = {rt.tag_id for rt in existing if rt.tag_id}
        selected_user_ids = {rt.tagged_user_id for rt in existing if rt.tagged_user_id}
        return render_template(
            "tag.html",
            resp=resp, users=users, tags=all_tags,
            selected_tag_ids=selected_tag_ids,
            selected_user_ids=selected_user_ids,
        )
    finally:
        sess.close()

# ---------------------------
# Team Members (search + status filter + multi-response history)
# ---------------------------
@app.route('/people')
@login_required
def person():
    try:
        from models import SessionLocal, User, Response, Tag, ResponseTag  # Tag tables optional
        have_tags = True
    except Exception:
        from models import SessionLocal, User, Response
        Tag = ResponseTag = None
        have_tags = False

    q = (request.args.get("q") or "").strip().lower()
    status_filter_raw = (request.args.get("status") or "").strip()
    status_filter = _normalize_status(status_filter_raw)

    session_db = SessionLocal()
    try:
        policy = MeetingPolicy.query.first() or MeetingPolicy(stale_days_threshold=7)

        users = session_db.query(User).all()

        sub_last = (
            session_db.query(Response.user_id, func.max(Response.created_at).label('last_ts'))
            .group_by(Response.user_id)
            .subquery()
        )

        latest_rows = (
            session_db.query(Response)
            .join(sub_last, (Response.user_id == sub_last.c.user_id) & (Response.created_at == sub_last.c.last_ts))
            .all()
        )
        latest_by_user = {r.user_id: r for r in latest_rows}

        counts = (
            session_db.query(
                Response.user_id,
                func.count(Response.id).label('resp_cnt'),
                func.sum(case((Response.blockers != 'No response', 1), else_=0)).label('block_cnt')
            )
            .group_by(Response.user_id)
            .all()
        )
        cnt_by_user = {u_id: (resp_cnt, (block_cnt or 0)) for (u_id, resp_cnt, block_cnt) in counts}

        members = []
        for u in users:
            resp_cnt, block_cnt = cnt_by_user.get(u.id, (0, 0))
            latest = latest_by_user.get(u.id)

            history_rows = (
                session_db.query(Response)
                .filter(Response.user_id == u.id)
                .order_by(Response.created_at.desc())
                .limit(10)
                .all()
            )

            history_dots = [{
                "date": _tz_aware(r.created_at),
                "blocked": _is_blocker_text(r.blockers)
            } for r in history_rows[:4]]

            history_items = [{
                "date": _tz_aware(r.created_at),
                "worked_on": r.worked_on,
                "next_up": r.next_up,
                "blockers": r.blockers
            } for r in history_rows]

            tags = []
            if have_tags and latest:
                tag_rows = (
                    session_db.query(Tag.name)
                    .join(ResponseTag, ResponseTag.tag_id == Tag.id)
                    .filter(ResponseTag.response_id == latest.id, ResponseTag.tag_id.isnot(None))
                    .all()
                )
                tags = [t[0] for t in tag_rows]

            status, tone, progress_pct, last_ts = _classify_member(latest, resp_cnt, block_cnt, policy)
            status_norm = _normalize_status(status)

            members.append({
                "user_id": getattr(u, "slack_id", None) or u.id,
                "name": u.name,
                "avatar_url": getattr(u, "avatar_url", None),
                "responses_count": resp_cnt,
                "blockers_count": block_cnt,
                "last_seen_human": _human_date(last_ts) if last_ts else '—',
                "latest_worked": getattr(latest, "worked_on", None) if latest else None,
                "latest_next": getattr(latest, "next_up", None) if latest else None,
                "latest_blockers": getattr(latest, "blockers", None) if latest else None,
                "status": status,
                "status_norm": status_norm,
                "tone": tone,
                "progress_pct": progress_pct,
                "tags": tags,
                "history": history_dots,
                "history_items": history_items,
                "resp_id": getattr(latest, "id", None) if latest else None,
            })

        total_count = len(members)

        def _matches(m):
            if status_filter and m["status_norm"] != status_filter:
                return False
            if not q:
                return True
            hay_parts = [
                str(m.get("name") or m.get("user_id") or ""),
                str(m.get("latest_worked") or ""),
                str(m.get("latest_next") or ""),
                str(m.get("latest_blockers") or ""),
                " ".join(m.get("tags") or []),
            ]
            for item in (m.get("history_items") or []):
                hay_parts.extend([
                    str(item.get("worked_on") or ""),
                    str(item.get("next_up") or ""),
                    str(item.get("blockers") or "")
                ])
            hay = " ".join(hay_parts).lower()
            return q in hay

        filtered = [m for m in members if _matches(m)]
        insights = generate_people_insights(filtered, policy)
        return render_template(
            "person.html",
            members=filtered,
            insights=insights,
            q=q,
            status_filter=status_filter_raw,
            total_count=total_count,
            filtered_count=len(filtered)
        )
    finally:
        session_db.close()

# Back-compat: /person -> /people
@app.route('/person')
def person_redirect():
    return redirect(url_for('person'))

@app.get("/pricing")
@login_required
def pricing():
    return render_template("pricing.html")

# ============================================================
# Member Detail page — all responses aligned to questions + metrics
# ============================================================
# app.py — REPLACE your existing member_detail route with this version
# app.py — REPLACE your existing member_detail route with this version
@app.route("/member/<user_key>", methods=["GET"])
@login_required
def member_detail(user_key):
    from models import User, Response, ResponseTag, Tag
    sess = SessionLocal()
    try:
        # Resolve user by numeric id OR slack_id
        user = None
        try:
            user = sess.query(User).filter_by(id=int(user_key)).first()
        except Exception:
            user = sess.query(User).filter_by(slack_id=str(user_key)).first()

        if not user:
            flash("User not found.", "warning")
            return redirect(url_for("person"))

        # Archive filter: active|archived|all (default active)
        show = (request.args.get("show") or "active").lower()

        q = sess.query(Response).filter(Response.user_id == user.id)
        if _has_archived_at(Response):
            if show == "active":
                q = q.filter((Response.archived_at.is_(None)) | (Response.archived_at == None))
            elif show == "archived":
                q = q.filter(Response.archived_at.isnot(None))
            else:
                pass  # all
        responses = q.order_by(Response.created_at.desc(), Response.id.desc()).all()

        # Build rows for template
        rows = []
        for r in responses:
            tag_names = [
                t.name for (_rt, t) in (
                    sess.query(ResponseTag, Tag)
                    .join(Tag, ResponseTag.tag_id == Tag.id)
                    .filter(ResponseTag.response_id == r.id)
                    .all()
                ) if t and t.name
            ]
            rows.append({
                "created_at": getattr(r, "created_at", None),
                "answers": [r.worked_on or "—", r.next_up or "—", r.blockers or "—"],
                "tags": tag_names,
                "resp_id": r.id,
                "archived": bool(getattr(r, "archived_at", None)),
            })

        # ----- metrics minimal (unchanged from yours) -----
        def _days_between(ts1, ts2):
            if not ts1 or not ts2: return None
            if ts1.tzinfo is None: ts1 = ts1.replace(tzinfo=timezone.utc)
            if ts2.tzinfo is None: ts2 = ts2.replace(tzinfo=timezone.utc)
            return abs((ts2 - ts1).days)

        total = len(responses)
        created_list = [r.created_at for r in responses if getattr(r, "created_at", None)]
        created_list_sorted = sorted(created_list, reverse=True)
        avg_interval_days = None
        if len(created_list_sorted) >= 2:
            gaps = []
            for i in range(len(created_list_sorted) - 1):
                d = _days_between(created_list_sorted[i], created_list_sorted[i+1])
                if d is not None:
                    gaps.append(d)
            if gaps:
                avg_interval_days = round(sum(gaps) / len(gaps), 1)

        last_blocker_days_ago = None
        now = _utcnow()
        for r in responses:
            blk = (r.blockers or "").strip().lower()
            if blk and blk not in ("no", "none", "n/a", "—"):
                delta = _days_between(getattr(r, "created_at", None), now)
                if delta is not None:
                    last_blocker_days_ago = delta
                    break

        blocker_count = sum(
            1 for r in responses
            if (r.blockers or "").strip().lower() not in ("", "no", "none", "n/a", "—")
        )
        blocker_rate = (blocker_count / total) if total > 0 else 0.0

        avg_answer_len = 0
        if total > 0:
            lengths = []
            for r in responses:
                lengths.extend([len(r.worked_on or ""), len(r.next_up or ""), len(r.blockers or "")])
            avg_answer_len = int(round(sum(lengths) / max(1, len(lengths))))

        streak_weeks = 0
        seen_weeks = set()
        for ts in created_list:
            if ts:
                if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
                seen_weeks.add((ts.year, ts.isocalendar().week))
        if seen_weeks:
            y, w, _ = _utcnow().isocalendar()
            while (y, w) in seen_weeks:
                streak_weeks += 1
                w -= 1
                if w == 0:
                    y -= 1
                    w = 52

        metrics = {
            "total": total,
            "streak_weeks": streak_weeks,
            "avg_interval_days": avg_interval_days,
            "blocker_rate": blocker_rate,
            "last_blocker_days_ago": last_blocker_days_ago,
            "avg_answer_len": avg_answer_len,
        }

        questions = [
            {"text": "What did you work on?", "tag": "worked_on"},
            {"text": "What’s next?", "tag": "next_up"},
            {"text": "Any blockers?", "tag": "blockers"},
        ]

        return render_template(
            "member.html",
            member={"name": user.name, "user_id": user.slack_id or user.id},
            metrics=metrics,
            questions=questions,
            rows=rows,
            archiving_supported=_has_archived_at(Response),
            show=show,
        )
    finally:
        sess.close()


# Back-compat alias (if any templates used url_for('member'))
app.add_url_rule("/member/<user_key>", endpoint="member", view_func=member_detail)

@app.get("/member/<user_key>/export.csv")
@login_required
def member_export_csv(user_key):
    """Export a member's full response history as CSV."""
    try:
        from models import SessionLocal, User, Response
    except Exception:
        flash("Export requires DB models.", "warning")
        return redirect(url_for("person"))

    sess = SessionLocal()
    try:
        user = None
        try:
            user = sess.query(User).filter_by(id=int(user_key)).first()
        except Exception:
            user = sess.query(User).filter_by(slack_id=str(user_key)).first()
        if not user:
            flash("User not found", "warning")
            return redirect(url_for("person"))

        responses = (sess.query(Response)
                     .filter(Response.user_id == user.id)
                     .order_by(Response.created_at.desc(), Response.id.desc())
                     .all())

        import csv, io
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["date", "worked_on", "next_up", "blockers", "response_id", "user_id", "name"])
        for r in responses:
            ts = getattr(r, "created_at", None)
            if ts and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            w.writerow([
                ts.isoformat() if ts else "",
                r.worked_on or "",
                r.next_up or "",
                r.blockers or "",
                r.id,
                getattr(user, "slack_id", None) or user.id,
                user.name or ""
            ])
        out = buf.getvalue().encode("utf-8")
        buf.close()
        filename = f"weeklybuddy_{(user.name or user.id)}.csv"
        from flask import Response as FlaskResponse
        return FlaskResponse(out, mimetype="text/csv",
                             headers={"Content-Disposition": f"attachment; filename={filename}"})
    finally:
        sess.close()
# app.py — ADD this route
# app.py — ADD this route
@app.post("/response/<int:resp_id>/archive")
@login_required
def response_archive(resp_id):
    from models import Response
    sess = SessionLocal()
    try:
        resp = sess.query(Response).filter_by(id=resp_id).first()
        if not resp:
            flash("Response not found.", "warning")
            return redirect(url_for("person"))

        if not _has_archived_at(Response):
            flash("Archiving is unavailable — add 'archived_at' to Response.", "warning")
            user_key = getattr(resp, "user_id", None)
            return redirect(url_for("member_detail", user_key=user_key) if user_key else url_for("person"))

        resp.archived_at = _utcnow()
        sess.commit()
        flash("Entry archived.", "success")

        user_key = getattr(resp, "user_id", None)
        return redirect(url_for("member_detail", user_key=user_key) if user_key else url_for("person"))
    except Exception as e:
        sess.rollback()
        flash(f"Could not archive: {e}", "danger")
        return redirect(url_for("person"))
    finally:
        sess.close()

@app.post("/response/<int:resp_id>/unarchive")
@login_required
def response_unarchive(resp_id):
    from models import Response
    sess = SessionLocal()
    try:
        resp = sess.query(Response).filter_by(id=resp_id).first()
        if not resp:
            flash("Response not found.", "warning")
            return redirect(url_for("person"))
        if not _has_archived_at(Response):
            flash("Unarchiving is unavailable — add 'archived_at' to Response.", "warning")
            user_key = getattr(resp, "user_id", None)
            return redirect(url_for("member_detail", user_key=user_key) if user_key else url_for("person"))
        resp.archived_at = None
        sess.commit()
        flash("Entry unarchived.", "success")
        user_key = getattr(resp, "user_id", None)
        return redirect(url_for("member_detail", user_key=user_key) if user_key else url_for("person"))
    except Exception as e:
        sess.rollback()
        flash(f"Could not unarchive: {e}", "danger")
        return redirect(url_for("person"))
    finally:
        sess.close()
# ============================================================
# Nudge: DM the teammate (optional escalation to manager)
# ============================================================
@app.route("/nudge/<user_key>", methods=["GET", "POST"])
@login_required
def nudge(user_key):
    try:
        from models import SessionLocal, User, Response
    except Exception:
        flash("DB models not available for nudges.", "danger")
        return redirect(url_for("dashboard"))

    sess = SessionLocal()
    try:
        user = None
        try:
            user = sess.query(User).filter_by(id=int(user_key)).first()
        except Exception:
            user = sess.query(User).filter_by(slack_id=str(user_key)).first()
        if not user:
            flash("User not found.", "warning")
            return redirect(url_for("person"))

        latest = (sess.query(Response)
                  .filter(Response.user_id == user.id)
                  .order_by(Response.created_at.desc(), Response.id.desc())
                  .first())

        if request.method == "POST":
            try:
                client = _slack_client()
            except Exception as e:
                flash(str(e), "danger")
                return redirect(url_for("person"))

            kind = request.form.get("kind", "gentle")
            custom = (request.form.get("message") or "").strip()
            escalate = (request.form.get("escalate") == "on")
            manager_id = os.getenv("MANAGER_SLACK_ID")  # optional

            worked = getattr(latest, "worked_on", None) or "—"
            nxt = getattr(latest, "next_up", None) or "—"
            blockers = getattr(latest, "blockers", None) or "—"

            presets = {
                "gentle": f"Quick check-in — when you have a moment, please post your weekly update. 🙏\nLast worked: {worked}",
                "blocker": f"I saw a blocker in your last check-in: “{blockers}”. Anything I can do to unblock?",
                "deadline": f"Reminder on our delivery this week — does your “next up” ({nxt}) still look on track?",
                "celebrate": f"Nice progress on “{worked}” last week. Anything we should highlight this week? 🎉",
            }
            text = custom or presets.get(kind, presets["gentle"])

            try:
                slack_id = getattr(user, "slack_id", None)
                if slack_id:
                    channel = _open_dm(client, slack_id)
                    client.chat_postMessage(channel=channel, text=text)
                else:
                    flash("User has no Slack ID; DM skipped.", "warning")

                if escalate and manager_id:
                    ch2 = _open_dm(client, manager_id)
                    mention = f"<@{slack_id}>" if slack_id else (user.name or str(user.id))
                    client.chat_postMessage(channel=ch2, text=f"Escalation: {mention}\n{text}")

                flash("Nudge sent.", "success")
            except Exception as e:
                flash(f"Nudge failed: {e}", "danger")
            return redirect(url_for("member_detail", user_key=(user.slack_id or user.id)))

        return render_template("nudge.html", user={
            "id": user.id,
            "user_id": getattr(user, "slack_id", None) or user.id,
            "name": user.name,
        }, latest=latest)
    finally:
        sess.close()

def _clean_blocker_text(s: str) -> str:
    return (s or "").strip().lower()

_STOPWORDS = {
    "the","a","an","and","or","to","of","for","in","on","at","is","are","am","be","was","were","it",
    "this","that","with","by","as","we","i","you","they","our","my","your","have","has","had","do",
    "did","does","from","about","there","here","week","weeks","next","plan","planning","worked","work",
    "any","none","no","blockers","blocker","n/a","na"
}

def _keyword_counts(texts, top_n=10):
    bag = Counter()
    for t in texts:
        t = re.sub(r"[^a-z0-9\s]+", " ", (t or "").lower())
        for w in t.split():
            if len(w) <= 2 or w in _STOPWORDS:
                continue
            bag[w] += 1
    return bag.most_common(top_n)

def _within_weeks(ts, weeks):
    if not ts:
        return False
    now = datetime.now(timezone.utc) if ts.tzinfo else datetime.utcnow().replace(tzinfo=timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (now - ts).days <= (weeks * 7)

def build_trends_advanced(timeframe_weeks=8):
    """
    Returns a dict of trend metrics for the last `timeframe_weeks`.
    Works with DB; falls back to responses.json.
    """
    # collector
    rows = []  # (user_name, created_at, worked_on, next_up, blockers, user_id, resp_id)
    used_db = False

    # Try DB
    try:
        from models import SessionLocal, User, Response, Tag, ResponseTag
        have_tags = True
    except Exception:
        have_tags = False

    try:
        from models import SessionLocal, User, Response
        sess = SessionLocal()
        try:
            q = (
                sess.query(User, Response)
                .join(Response, Response.user_id == User.id)
                .order_by(Response.created_at.desc())
                .all()
            )
            for u, r in q:
                ts = getattr(r, "created_at", None)
                if not ts: 
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if not _within_weeks(ts, timeframe_weeks):
                    continue
                rows.append((
                    u.name or getattr(u, "slack_id", None) or f"User {u.id}",
                    ts, r.worked_on, r.next_up, r.blockers, u.id, r.id
                ))
            used_db = True
        finally:
            sess.close()
    except Exception:
        pass

    # Fallback JSON
    if not used_db:
        for r in load_responses_json():
            ts = r.get("timestamp")
            dt = None
            if ts:
                try: dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except: dt = None
            if not dt or not _within_weeks(dt.replace(tzinfo=timezone.utc), timeframe_weeks):
                continue
            name = r.get("name") or r.get("user_id") or "Unknown"
            answers = r.get("answers") or ["","",""]
            rows.append((name, dt.replace(tzinfo=timezone.utc), answers[0], answers[1], answers[2], None, None))

    # Aggregate
    blockers_by_user = Counter()
    responses_by_user = Counter()
    blocker_texts = []
    weekday_counts = Counter()
    weekly_buckets = Counter()  # week label -> count

    # Build week labels
    end = datetime.now(timezone.utc)
    start = end - timedelta(weeks=timeframe_weeks-1)
    week_starts = []
    cur = (start - timedelta(days=start.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    while cur <= end:
        week_starts.append(cur)
        cur += timedelta(weeks=1)
    week_labels = [ws.strftime("Wk %V · %b %d") for ws in week_starts]

    def week_label_for(ts):
        for i, ws in enumerate(week_starts):
            we = ws + timedelta(weeks=1)
            if ws <= ts < we:
                return week_labels[i]
        return week_labels[-1]

    for name, ts, w, n, b, uid, rid in rows:
        responses_by_user[name] += 1
        if _is_blocker_text(b):
            blockers_by_user[name] += 1
            blocker_texts.append(b)
            weekday_counts[ts.weekday()] += 1
            weekly_buckets[week_label_for(ts)] += 1

    # Weekly series in correct order
    weekly_counts = [weekly_buckets.get(lbl, 0) for lbl in week_labels]

    # Change (last 4 vs prev 4)
    last4 = sum(weekly_counts[-4:]) if len(weekly_counts) >= 4 else sum(weekly_counts)
    prev4 = sum(weekly_counts[-8:-4]) if len(weekly_counts) >= 8 else 0
    change_pct = None
    if prev4 == 0 and last4 > 0:
        change_pct = 100.0
    elif prev4 > 0:
        change_pct = round((last4 - prev4) * 100.0 / prev4, 1)

    # Keyword extraction
    top_keywords = _keyword_counts(blocker_texts, top_n=10)

    # Tag breakdown (if tables exist)
    tag_counts = []
    if used_db:
        try:
            from models import SessionLocal, Tag, ResponseTag, Response
            sess = SessionLocal()
            try:
                # only within timeframe
                ids_in_window = [rid for (_,ts,_,_,_,_,rid) in rows if rid]
                if ids_in_window:
                    q = (sess.query(Tag.name, func.count(ResponseTag.id))
                         .join(ResponseTag, ResponseTag.tag_id == Tag.id)
                         .filter(ResponseTag.response_id.in_(ids_in_window), ResponseTag.tag_id.isnot(None))
                         .group_by(Tag.name)
                         .order_by(func.count(ResponseTag.id).desc())
                         .all())
                    tag_counts = [(n, c) for n, c in q]
            finally:
                sess.close()
        except Exception:
            tag_counts = []

    total_blockers = sum(blockers_by_user.values())
    total_responses = sum(responses_by_user.values())
    blocker_rate = (total_blockers / total_responses) if total_responses else 0.0

    return {
        "timeframe_weeks": timeframe_weeks,
        "blockers_by_user": blockers_by_user.most_common(),
        "responses_by_user": responses_by_user.most_common(),
        "total_blockers": total_blockers,
        "total_responses": total_responses,
        "blocker_rate": round(blocker_rate, 3),
        "week_labels": week_labels,
        "weekly_counts": weekly_counts,
        "change_pct": change_pct,
        "weekday_counts": [weekday_counts.get(i,0) for i in range(7)],  # Mon=0..Sun=6
        "top_keywords": top_keywords,          # list[(word, count)]
        "tag_counts": tag_counts               # list[(tag, count)]
    }

def generate_trend_ai_insights(tr):
    tips = []
    # Rising trend?
    wc = tr["weekly_counts"]
    if len(wc) >= 3 and wc[-3] < wc[-2] < wc[-1]:
        tips.append("📈 Blockers have increased 3 weeks in a row — schedule a short unblocker standup.")
    # Big jump last 4 weeks
    if tr["change_pct"] is not None and tr["change_pct"] >= 50:
        tips.append(f"🚨 Last 4 weeks blockers up {tr['change_pct']}% vs prior — review intake/process bottlenecks.")
    # High weekday concentration
    wk = tr["weekday_counts"]
    if wk and max(wk) > (sum(wk) * 0.35 if sum(wk) else 0):
        day = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][wk.index(max(wk))]
        tips.append(f"🗓️ Most blockers hit on {day}. Plan a recurring triage right before that day.")
    # Keywords
    if tr["top_keywords"]:
        kws = ", ".join(k for k,_ in tr["top_keywords"][:5])
        tips.append(f"🔎 Frequent themes: {kws}. Create playbooks/owners for these.")
    # Tags
    if tr["tag_counts"]:
        top_tag, cnt = tr["tag_counts"][0]
        tips.append(f"🏷️ Tag ‘{top_tag}’ leads with {cnt} items — ensure DRIs and SLAs are clear.")
    # Users with many blockers
    if tr["blockers_by_user"]:
        names = ", ".join(n for n,_ in tr["blockers_by_user"][:3])
        tips.append(f"👥 Heaviest load: {names}. Consider 1:1s and pairing to unblock.")
    if not tips:
        tips.append("✅ Trendline looks steady. Keep cadence and tag new blockers for faster routing.")
    return tips
# --- Send the 3 questions to one member (ADD) ---
from flask import request, redirect, url_for, flash
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")  # set in your env

def slack_post_dm(slack_user_id: str, text: str, blocks: list | None = None) -> tuple[bool, str]:
    """Send a DM to a Slack user by slack_id using chat.postMessage with optional Block Kit blocks."""
    if not SLACK_BOT_TOKEN:
        return False, "SLACK_BOT_TOKEN missing"
    url = "https://slack.com/api/chat.postMessage"
    headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
        "Content-Type": "application/json; charset=utf-8",
    }
    payload = {"channel": slack_user_id, "text": text}
    if blocks is not None:
        payload["blocks"] = blocks
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
    try:
        data = r.json()
    except Exception:
        return False, f"Slack error: {r.status_code}"
    return (data.get("ok", False), data.get("error", ""))

@app.post("/member/<user_key>/send-checkin")
@login_required
def send_checkin(user_key):
    from models import SessionLocal, User
    sess = SessionLocal()
    try:
        # Resolve by numeric id OR slack_id
        user = None
        try:
            user = sess.query(User).filter_by(id=int(user_key)).first()
        except Exception:
            user = sess.query(User).filter_by(slack_id=str(user_key)).first()

        if not user:
            flash("User not found.", "warning")
            return redirect(url_for("person"))

        if not user.slack_id:
            flash("No Slack ID on this user — add slack_id to enable DM.", "warning")
            return redirect(url_for("member_detail", user_key=user_key))

        intro = (
            f"Hey {user.name or 'there'}! It’s check‑in time.\n"
            "Click the button to open a short form (3 questions)."
        )
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": intro}},
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Open check‑in"},
                        "style": "primary",
                        "action_id": "open_checkin",
                        "value": user.slack_id,
                    }
                ],
            },
        ]

        ok, err = slack_post_dm(user.slack_id, "It’s check‑in time.", blocks=blocks)
        if ok:
            flash("Interactive check‑in sent via Slack ✅", "success")
        else:
            flash(f"Slack send failed: {err}", "danger")

        return redirect(url_for("member_detail", user_key=user.slack_id or user.id))
    finally:
        sess.close()

# --- Bulk send to all users (ADD) ---
@app.post("/team/send-checkins", endpoint="team_send_checkins")
@login_required
def team_send_checkins():
    from models import SessionLocal, User
    sess = SessionLocal()
    try:
        users = sess.query(User).all()
        sent, skipped = 0, 0
        for u in users:
            if not getattr(u, "slack_id", None):
                skipped += 1
                continue
            intro = (
                "Hey! It’s check‑in time.\n"
                "Click the button below to fill it out (3 quick questions)."
            )
            blocks = [
                {"type": "section", "text": {"type": "mrkdwn", "text": intro}},
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Open check‑in"},
                            "style": "primary",
                            "action_id": "open_checkin",
                            "value": u.slack_id,
                        }
                    ],
                },
            ]
            ok, _ = slack_post_dm(u.slack_id, "It’s check‑in time.", blocks=blocks)
            sent += 1 if ok else 0
        flash(f"Sent {sent} interactive check‑ins. Skipped {skipped} (no Slack ID).", "success")
        return redirect(url_for("person"))
    finally:
        sess.close()


# app.py — ADD this route
@app.post("/member/<user_key>/bulk-archive")
@login_required
def member_bulk_archive(user_key):
    from models import User, Response
    sess = SessionLocal()
    try:
        if not _has_archived_at(Response):
            flash("Archiving is unavailable — add 'archived_at' to Response.", "warning")
            return redirect(url_for("member_detail", user_key=user_key))

        older_than_days = int(request.form.get("older_than_days", "90"))
        cutoff = _utcnow() - timedelta(days=older_than_days)

        # Resolve user by id or slack_id
        user = None
        try:
            user = sess.query(User).filter_by(id=int(user_key)).first()
        except Exception:
            user = sess.query(User).filter_by(slack_id=str(user_key)).first()

        if not user:
            flash("User not found.", "warning")
            return redirect(url_for("person"))

        # Bulk update (fast)
        updated = (
            sess.query(Response)
            .filter(Response.user_id == user.id)
            .filter((Response.archived_at.is_(None)) | (Response.archived_at == None))
            .filter(Response.created_at <= cutoff)
            .update({Response.archived_at: _utcnow()}, synchronize_session=False)
        )
        sess.commit()
        flash(f"Archived {updated} update(s) older than {older_than_days} days.", "success")
        return redirect(url_for("member_detail", user_key=user.slack_id or user.id))
    except Exception as e:
        sess.rollback()
        flash(f"Bulk archive failed: {e}", "danger")
        return redirect(url_for("member_detail", user_key=user_key))
    finally:
        sess.close()



# Back-compat: if something still posts to /unblock, redirect to /nudge
@app.post("/unblock/<user_key>")
@login_required
def start_unblock_thread(user_key):
    flash("Unblock now routes to Nudge. Customize your message and send via Slack.", "info")
    return redirect(url_for("nudge", user_key=user_key))

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
