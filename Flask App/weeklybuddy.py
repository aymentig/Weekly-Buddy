# weeklybuddy.py  — MVP: channel fan-out + start button + modal
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
from models import SessionLocal, User, Response, init_db

load_dotenv()
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

app = App(token=SLACK_BOT_TOKEN)
import json
from datetime import datetime, timezone
from models import SessionLocal, User, Response, init_db

init_db()

@app.action("open_checkin")
def handle_open_checkin(ack, body, client, logger):
    ack()
    try:
        trigger_id = body["trigger_id"]
        clicked_by = body.get("user", {}).get("id")
        target = None
        actions = body.get("actions", [])
        if actions:
            target = actions[0].get("value")
        slack_user_id = target or clicked_by

        view = {
            "type": "modal",
            "callback_id": "checkin_modal",
            "title": {"type": "plain_text", "text": "Weekly Check‑in"},
            "submit": {"type": "plain_text", "text": "Submit"},
            "private_metadata": json.dumps({"slack_user_id": slack_user_id}),
            "blocks": [
                {"type": "input", "block_id": "worked", "label": {"type": "plain_text", "text": "What did you work on?"}, "element": {"type": "plain_text_input", "action_id": "worked_on", "multiline": True}},
                {"type": "input", "block_id": "next",   "label": {"type": "plain_text", "text": "What’s next?"},           "element": {"type": "plain_text_input", "action_id": "next_up",   "multiline": True}},
                {"type": "input", "block_id": "blockers","label": {"type": "plain_text", "text": "Any blockers?"},         "element": {"type": "plain_text_input", "action_id": "blockers",  "multiline": True}},
            ],
        }
        client.views_open(trigger_id=trigger_id, view=view)
    except Exception as e:
        logger.exception(e)

@app.view("checkin_modal")
def handle_checkin_submit(ack, body, client, logger):
    ack()
    try:
        md = json.loads(body.get("view", {}).get("private_metadata", "{}"))
        slack_user_id = md.get("slack_user_id") or body.get("user", {}).get("id")
        state = body["view"]["state"]["values"]
        worked_on = state["worked"]["worked_on"]["value"].strip()
        next_up   = state["next"]["next_up"]["value"].strip()
        blockers  = state["blockers"]["blockers"]["value"].strip()

        sess = SessionLocal()
        try:
            user = sess.query(User).filter_by(slack_id=slack_user_id).first()
            if not user:
                prof_name = None
                try:
                    info = client.users_info(user=slack_user_id)
                    prof_name = info.get("user", {}).get("profile", {}).get("real_name")
                except Exception:
                    pass
                user = User(slack_id=slack_user_id, name=prof_name or slack_user_id)
                sess.add(user)
                sess.flush()

            resp = Response(
                user_id=user.id,
                worked_on=worked_on,
                next_up=next_up,
                blockers=blockers,
                created_at=datetime.now(timezone.utc)
            )
            sess.add(resp)
            sess.commit()
        finally:
            sess.close()

        try:
            client.chat_postMessage(channel=slack_user_id, text="Thanks! I recorded your weekly update. ✅")
        except Exception:
            pass
    except Exception as e:
        logger.exception(e)

# DB
init_db()

# ------------------------------------------------------------
# UI bits
# ------------------------------------------------------------
START_BLOCKS = [
    {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": "*Hey!* Ready to post your WeeklyBuddy check-in?"
        }
    },
    {
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Start check-in"},
                "style": "primary",
                "action_id": "checkin_start"
            }
        ]
    }
]

def open_checkin_modal(client, trigger_id, user_id):
    """Open the 3-question modal for a user."""
    client.views_open(
        trigger_id=trigger_id,
        view={
            "type": "modal",
            "callback_id": "submit_checkin",
            "private_metadata": user_id,
            "title": {"type": "plain_text", "text": "WeeklyBuddy Check-in"},
            "submit": {"type": "plain_text", "text": "Submit"},
            "blocks": [
                {
                    "type": "input",
                    "block_id": "worked_on",
                    "element": {"type": "plain_text_input", "action_id": "input", "multiline": True},
                    "label": {"type": "plain_text", "text": "1️⃣ What did you work on this week?"}
                },
                {
                    "type": "input",
                    "block_id": "next_up",
                    "element": {"type": "plain_text_input", "action_id": "input", "multiline": True},
                    "label": {"type": "plain_text", "text": "2️⃣ What are you planning next?"}
                },
                {
                    "type": "input",
                    "block_id": "blockers",
                    "optional": True,
                    "element": {"type": "plain_text_input", "action_id": "input", "multiline": True},
                    "label": {"type": "plain_text", "text": "3️⃣ Any blockers?"}
                }
            ]
        },
    )

def send_start_dm(client, user_id):
    """DM a user the Start button."""
    dm = client.conversations_open(users=user_id)  # needs appropriate scopes + bot in workspace
    ch = dm["channel"]["id"]
    client.chat_postMessage(channel=ch, text="Start your check-in", blocks=START_BLOCKS)

# ------------------------------------------------------------
# Fan out from a channel
# ------------------------------------------------------------
def start_channel_checkins(client, channel_id: str, requester: str, include_self: bool) -> int:
    """Send Start button DMs to all humans in the channel (optionally excluding requester)."""
    # Try to join public channels so we can access members
    try:
        info = client.conversations_info(channel=channel_id)
        ch = info.get("channel") or {}
        is_public = ch.get("is_channel", False) and not ch.get("is_private", False)
        is_member = ch.get("is_member", True)
        if is_public and not is_member:
            try:
                client.conversations_join(channel=channel_id)  # requires channels:join
            except Exception:
                pass
    except Exception:
        ch = {}

    # Pull members (with pagination)
    members = []
    cursor = None
    try:
        while True:
            resp = client.conversations_members(channel=channel_id, limit=1000, cursor=cursor)
            members.extend(resp.get("members", []))
            cursor = (resp.get("response_metadata") or {}).get("next_cursor")
            if not cursor:
                break
    except Exception:
        # Tell the requester how to fix it
        try:
            client.chat_postEphemeral(
                channel=channel_id,
                user=requester,
                text=("I couldn't read members here. Make sure I'm invited and have "
                      "`channels:read`, `groups:read`, `mpim:read`, `im:read` (and `channels:join` for public).")
            )
        except Exception:
            pass
        return 0

    # Filter to humans
    humans = [m for m in members if m.startswith("U")]
    if not include_self:
        humans = [m for m in humans if m != requester]

    if not humans:
        try:
            client.chat_postEphemeral(
                channel=channel_id,
                user=requester,
                text="No teammates found to nudge here (is this a DM or only two members?). Add teammates or use `--me` to include yourself."
            )
        except Exception:
            pass
        return 0

    sent = 0
    for m in humans:
        try:
            send_start_dm(client, m)
            sent += 1
        except Exception as e:
            # Optional: log to server
            print(f"[weeklybuddy] DM failed to {m}: {e}")
            continue
    return sent

# ------------------------------------------------------------
# Interactions
# ------------------------------------------------------------
@app.action("checkin_start")
def handle_checkin_start(ack, body, client):
    """User clicked Start -> open modal using this interaction's trigger_id."""
    ack()
    trigger_id = body.get("trigger_id")
    user_id = (body.get("user") or {}).get("id")
    if trigger_id and user_id:
        open_checkin_modal(client, trigger_id, user_id)

@app.view("submit_checkin")
def handle_submission(ack, body, view, client):
    """Persist the check-in to DB and thank the user."""
    ack()
    user_id = view.get("private_metadata") or (body.get("user") or {}).get("id")
    values = view["state"]["values"]
    worked_on = values["worked_on"]["input"]["value"]
    next_up = values["next_up"]["input"]["value"]
    blockers = values.get("blockers", {}).get("input", {}).get("value", "")

    db = SessionLocal()
    try:
        user = db.query(User).filter_by(slack_id=user_id).first()
        if not user:
            try:
                profile = client.users_info(user=user_id)["user"]["profile"]
                display_name = profile.get("display_name") or profile.get("real_name") or "Unknown"
            except Exception:
                display_name = "Unknown"
            user = User(slack_id=user_id, name=display_name)
            db.add(user)
            db.commit()
            db.refresh(user)

        resp = Response(user_id=user.id, worked_on=worked_on, next_up=next_up, blockers=blockers)
        db.add(resp)
        db.commit()
    finally:
        db.close()

    try:
        dm = client.conversations_open(users=user_id)
        client.chat_postMessage(channel=dm["channel"]["id"], text="✅ Thanks! Your check-in has been saved.")
    except Exception:
        pass

# ------------------------------------------------------------
# Channel entry points
# ------------------------------------------------------------
@app.command("/weeklybuddy")
def handle_command(ack, body, client, say):
    """Usage: /weeklybuddy weekly [--me]"""
    ack()
    text = (body.get("text") or "").strip().lower()
    channel = body.get("channel_id")
    requester = body.get("user_id")

    if text.startswith("weekly"):
        include_self = ("--me" in text) or ("include me" in text)
        sent = start_channel_checkins(client, channel, requester, include_self)
        say(f"Got it <@{requester}> — starting the check-in! 🏁  (DM’d {sent} teammate(s))")
    else:
        say("Usage: `/weeklybuddy weekly` (add `--me` to include yourself)")

@app.event("app_mention")
def handle_mention(event, say, client):
    """@weeklybuddy weekly (same behavior as the slash command)."""
    text = (event.get("text") or "").lower()
    channel = event.get("channel")
    requester = event.get("user")

    if "weekly" in text:
        include_self = ("--me" in text) or ("include me" in text)
        sent = start_channel_checkins(client, channel, requester, include_self)
        say(f"Got it <@{requester}> — starting the check-in! 🏁  (DM’d {sent} teammate(s))")

# ------------------------------------------------------------
# Boot
# ------------------------------------------------------------
if __name__ == "__main__":
    print("🤖 WeeklyBuddy bot running...")
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
