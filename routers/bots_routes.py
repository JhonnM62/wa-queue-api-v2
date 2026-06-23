from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import json

from models import bot_models as models
from schemas import bot_schemas as schemas
from core import auth
from core.database import get_db
from services import rag_service

router = APIRouter(tags=["bots"])

# ── CRUD ─────────────────────────────────────────────────────────────

@router.post("")
@router.post("/", response_model=schemas.BotConfigResponse)
def create_bot(
    bot: schemas.BotConfigCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_active_user)
):
    db_bot = db.query(models.BotConfig).filter(
        models.BotConfig.userbot_identifier == bot.userbot_identifier
    ).first()
    if db_bot:
        if db_bot.user_id is None:
            # Reclamar bot huérfano
            db_bot.user_id = current_user.id
            for field, value in bot.model_dump(exclude_unset=True).items():
                setattr(db_bot, field, value)
            db.commit()
            db.refresh(db_bot)
            return db_bot
        else:
            raise HTTPException(status_code=400, detail="Bot identifier already registered")

    new_bot = models.BotConfig(user_id=current_user.id, **bot.model_dump())
    db.add(new_bot)
    db.commit()
    db.refresh(new_bot)
    return new_bot


@router.get("")
@router.get("/", response_model=List[schemas.BotConfigResponse])
def get_my_bots(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_active_user)
):
    return db.query(models.BotConfig).filter(models.BotConfig.user_id == current_user.id).all()


@router.get("/lookup/{userbot_identifier}", response_model=schemas.BotConfigResponse)
def lookup_bot_by_identifier(
    userbot_identifier: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_active_user)
):
    """
    Busca un bot por su userbot_identifier.
    Útil para autocompletar el formulario cuando el ID ya existe en la BD.
    """
    bot = db.query(models.BotConfig).filter(
        models.BotConfig.userbot_identifier == userbot_identifier
    ).first()
    if not bot:
        raise HTTPException(status_code=404, detail="No hay configuración guardada para ese ID de userbot")
    return bot


@router.put("/{bot_id}", response_model=schemas.BotConfigResponse)
def update_bot(
    bot_id: int,
    bot_update: schemas.BotConfigUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_active_user)
):
    bot = db.query(models.BotConfig).filter(models.BotConfig.id == bot_id).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
        
    if bot.user_id is None:
        bot.user_id = current_user.id
    elif current_user.role != "admin" and bot.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough privileges")

    for field, value in bot_update.model_dump(exclude_unset=True).items():
        setattr(bot, field, value)

    db.commit()
    db.refresh(bot)
    return bot


@router.delete("/{bot_id}")
def delete_bot(
    bot_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_active_user)
):
    bot = db.query(models.BotConfig).filter(models.BotConfig.id == bot_id).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    if current_user.role != "admin" and bot.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough privileges")

    # Eliminar el conocimiento RAG asociado
    rag_service.delete_bot_knowledge(bot.id, api_key=bot.apikey or "")

    db.delete(bot)
    db.commit()
    return {"message": "Bot deleted successfully"}


# ── SIMULADOR DE CHAT ────────────────────────────────────────────────
@router.post("/{bot_id}/simulate")
async def simulate_bot(
    bot_id: int,
    mensaje: str = Form(""),
    historial: str = Form("[]"),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_active_user)
):
    bot = db.query(models.BotConfig).filter(models.BotConfig.id == bot_id).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    if current_user.role != "admin" and bot.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough privileges")

    try:
        history_list = json.loads(historial)
    except:
        history_list = []

    final_content = mensaje.strip()
    
    # Simular el procesamiento multimedia
    if file and bot.apikey:
        try:
            file_bytes = await file.read()
            import google.generativeai as genai
            genai.configure(api_key=bot.apikey)
            model_name = bot.ai_model or "gemini-2.5-flash"
            model = genai.GenerativeModel(model_name)
            
            prompt = "Describe detalladamente el contenido de este archivo. Si es imagen describe qué hay, si es audio transcribe exactamente lo que dice, si es documento extrae su texto."
            resp = await model.generate_content_async([
                {"mime_type": file.content_type, "data": file_bytes},
                prompt
            ])
            final_content += f"\n[Adjunto simulado: {resp.text}]"
        except Exception as e:
            final_content += f"\n[Error simulando adjunto: {e}]"

    from services.graph import process_message_with_graph
    
    try:
        try:
            tools_cfg = json.loads(bot.tools_config or "{}")
        except:
            tools_cfg = {}
            
        config_dict = {
            "api_key": bot.apikey,
            "ai_model": bot.ai_model,
            "tools_config": tools_cfg,
            "bot_id": bot.id,
            "userbot_identifier": bot.userbot_identifier,
            "use_google_search": bot.use_google_search,
            "use_google_maps": bot.use_google_maps
        }
        ai_response = process_message_with_graph(
            bot_id=bot.id,
            phone="simulador",
            message=final_content,
            system_prompt=bot.system_prompt,
            config_dict=config_dict,
            json_history=history_list
        )
        try:
            parsed = json.loads(ai_response)
            return {"response": parsed}
        except:
            return {"response": {"respuesta_cliente": ai_response, "estado_conversacion": "procesando", "accion_interna": ""}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── KNOWLEDGE / RAG ──────────────────────────────────────────────────

@router.post("/{bot_id}/upload-knowledge")
async def upload_bot_knowledge(
    bot_id: int,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_active_user)
):
    bot = db.query(models.BotConfig).filter(models.BotConfig.id == bot_id).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    if current_user.role != "admin" and bot.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough privileges")
    
    results = []
    try:
        for file in files:
            if not file.filename.lower().endswith(('.pdf', '.txt', '.png', '.jpg', '.jpeg', '.webp')):
                raise HTTPException(status_code=400, detail=f"File {file.filename} not supported. Only PDF, TXT, PNG, JPG, and WEBP files are supported")
            result = await rag_service.process_and_store_document(bot.id, file, api_key=bot.apikey or "")
            results.append(result)
        return {"message": " / ".join(results)}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{bot_id}/knowledge")
def get_bot_knowledge(
    bot_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_active_user)
):
    bot = db.query(models.BotConfig).filter(models.BotConfig.id == bot_id).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    if current_user.role != "admin" and bot.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough privileges")
    
    knowledge = rag_service.get_all_knowledge(bot.id, api_key=bot.apikey or "")
    return {"knowledge": knowledge}

@router.put("/{bot_id}/knowledge/{doc_id}")
def update_bot_knowledge(
    bot_id: int,
    doc_id: str,
    update_data: schemas.BotKnowledgeUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_active_user)
):
    bot = db.query(models.BotConfig).filter(models.BotConfig.id == bot_id).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    if current_user.role != "admin" and bot.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough privileges")
    
    try:
        rag_service.update_single_knowledge(bot.id, doc_id, update_data.content, api_key=bot.apikey or "")
        return {"message": "Conocimiento actualizado exitosamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{bot_id}/knowledge/{doc_id}")
def delete_bot_knowledge_single(
    bot_id: int,
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_active_user)
):
    bot = db.query(models.BotConfig).filter(models.BotConfig.id == bot_id).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    if current_user.role != "admin" and bot.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough privileges")
    
    try:
        rag_service.delete_single_knowledge(bot.id, doc_id, api_key=bot.apikey or "")
        return {"message": "Conocimiento eliminado exitosamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
