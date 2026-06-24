from core.database import SessionLocal
from models.bot_models import ConversationMessage

db = SessionLocal()
try:
    new_msg = ConversationMessage(
        bot_id=1,
        userbot_identifier='test',
        phone='123',
        role='test',
        message='test',
        timestamp_ms=123
    )
    db.add(new_msg)
    db.commit()
    print("Success inserting message!")
except Exception as e:
    print(f"Error inserting: {e}")
finally:
    db.close()
