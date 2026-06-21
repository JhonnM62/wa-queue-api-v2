"""
Script de creación del usuario admin.
Ejecutar con: python create_admin.py
"""
import sys
import os

# Asegurar que las rutas del proyecto estén disponibles
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from core.database import SessionLocal, engine
from models import bot_models as models
from core.auth import get_password_hash

# Crear tablas si no existen
models.Base.metadata.create_all(bind=engine)

EMAIL = "admin@autosystemprojects.site"
PASSWORD = "AdminVitalicio2024!"

db = SessionLocal()

try:
    # Verificar si ya existe
    existing = db.query(models.User).filter(models.User.email == EMAIL).first()
    if existing:
        print(f"[OK] El usuario '{EMAIL}' ya existe con rol: {existing.role}")
    else:
        total = db.query(models.User).count()
        role = "admin" if total == 0 else "client"
        
        hashed = get_password_hash(PASSWORD)
        new_user = models.User(email=EMAIL, hashed_password=hashed, role=role)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        print("[OK] Usuario creado exitosamente:")
        print(f"   Email: {new_user.email}")
        print(f"   Rol:   {new_user.role}")
        print(f"   ID:    {new_user.id}")
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
finally:
    db.close()
