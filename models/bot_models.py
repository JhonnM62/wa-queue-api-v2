from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Text, Float
from sqlalchemy.orm import relationship
from core.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="client")  # "admin" or "client"

    bots = relationship("BotConfig", back_populates="owner")

class BotConfig(Base):
    __tablename__ = "bots"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))

    # === IDENTIFICACION ===
    userbot_identifier = Column(String, unique=True, index=True)  # req.userbot

    # === API KEY DEL CLIENTE ===
    # Se rellena automáticamente desde el campo `apikey` de la primera petición recibida.
    # Se usa tanto para el chat (Gemini) como para los embeddings del RAG.
    apikey = Column(String, default="")

    # === IA / PROMPT ===
    system_prompt = Column(Text, default="")
    ai_model = Column(String, default="gemini-2.5-flash")
    thinking_budget = Column(Integer, default=-1)  # -1=razonar, 0=no razonar
    thinking_level = Column(String, default="HIGH") # MINIMAL, LOW, MEDIUM, HIGH
    use_google_search = Column(Boolean, default=False)
    use_google_maps = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)

    # === CONTEXTO REGIONAL ===
    pais = Column(String, default="colombia")
    idioma = Column(String, default="es")

    # === TIEMPOS ===
    delay_seconds = Column(Float, default=0.0)
    pause_timeout_minutes = Column(Integer, default=30)

    # === NOTIFICACIONES ===
    activarnotificacion = Column(Boolean, default=False)
    estado_notificacion = Column(String, default="")        # El estado que dispara la notificación ej: "Pedido"
    lineaogruponotificacion = Column(String, default="")    # Número o ID de grupo destino
    es_grupo_notificacion = Column(Boolean, default=False)  # True si el destino es un grupo
    activaruserbotopcional = Column(Boolean, default=False)
    userbotopcional = Column(String, default="")            # userbot alternativo para notificaciones

    # === HERRAMIENTAS (TOOLS) ===
    # JSON string: {"buscar_catalogo": true, "enviar_notificacion": true, "obtener_hora": true}
    tools_config = Column(Text, default='{"buscar_catalogo": true, "enviar_notificacion": true, "obtener_hora": true}')

    owner = relationship("User", back_populates="bots")
