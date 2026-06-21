from pydantic import BaseModel
from typing import List, Optional

# ── Auth Schemas ─────────────────────────────────────────────────────

class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    role: str

    class Config:
        from_attributes = True

# ── Bot Schemas ──────────────────────────────────────────────────────

class BotConfigBase(BaseModel):
    userbot_identifier: str

    # API Key del cliente (se usa para Gemini y para los embeddings del RAG)
    apikey: Optional[str] = ""

    # IA / Prompt
    system_prompt: Optional[str] = ""
    ai_model: Optional[str] = "gemini-2.5-flash"
    thinking_budget: Optional[int] = -1
    thinking_level: Optional[str] = "HIGH"
    use_google_search: Optional[bool] = False
    use_google_maps: Optional[bool] = False
    is_active: Optional[bool] = True

    # Contexto regional
    pais: Optional[str] = "colombia"
    idioma: Optional[str] = "es"

    # Tiempos
    delay_seconds: Optional[float] = 0.0
    pause_timeout_minutes: Optional[int] = 30

    # Notificaciones
    activarnotificacion: Optional[bool] = False
    estado_notificacion: Optional[str] = ""
    lineaogruponotificacion: Optional[str] = ""
    es_grupo_notificacion: Optional[bool] = False
    activaruserbotopcional: Optional[bool] = False
    userbotopcional: Optional[str] = ""

    # Tools (JSON string)
    tools_config: Optional[str] = '{"buscar_catalogo": true, "enviar_notificacion": true, "obtener_hora": true}'

class BotConfigCreate(BotConfigBase):
    pass

class BotConfigUpdate(BaseModel):
    apikey: Optional[str] = None
    system_prompt: Optional[str] = None
    ai_model: Optional[str] = None
    thinking_budget: Optional[int] = None
    thinking_level: Optional[str] = None
    use_google_search: Optional[bool] = None
    use_google_maps: Optional[bool] = None
    is_active: Optional[bool] = None
    pais: Optional[str] = None
    idioma: Optional[str] = None
    delay_seconds: Optional[float] = None
    pause_timeout_minutes: Optional[int] = None
    activarnotificacion: Optional[bool] = None
    estado_notificacion: Optional[str] = None
    lineaogruponotificacion: Optional[str] = None
    es_grupo_notificacion: Optional[bool] = None
    activaruserbotopcional: Optional[bool] = None
    userbotopcional: Optional[str] = None
    tools_config: Optional[str] = None

class BotConfigResponse(BotConfigBase):
    id: int
    user_id: Optional[int] = None

    class Config:
        from_attributes = True

class BotKnowledgeUpdate(BaseModel):
    content: str

# ── Token Schemas ────────────────────────────────────────────────────

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
