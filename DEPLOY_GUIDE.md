# 🐳 Guía de Despliegue y Actualización — AutoSystem Panel

---

## ¿Qué hace este proyecto?

Un microservicio de chatbots de WhatsApp con:
- **Motor de IA autónomo** via LangGraph + Gemini (sin aprobaciones manuales)
- **Panel SaaS** en React para gestionar múltiples bots
- **RAG** (base de conocimiento por PDF/TXT por bot)
- **API REST** con autenticación JWT

---

## 1. Requisitos previos en el contenedor

El contenedor ya tiene Python instalado. Solo necesitas asegurarte de tener `git` y `pip`:

```bash
git --version
pip --version
```

---

## 2. Clonar el repositorio (primera vez)

```bash
git clone https://github.com/TU_USUARIO/TU_REPO.git /app
cd /app
```

---

## 3. Instalar dependencias Python

```bash
pip install -r requirements.txt
```

Si el `requirements.txt` no tiene los nuevos paquetes aún, instálalos manualmente:

```bash
pip install \
  langchain-chroma \
  langchain-text-splitters \
  langchain-community \
  langchain-google-genai \
  langgraph \
  sqlalchemy \
  "bcrypt==4.0.1" \
  python-jose[cryptography] \
  python-multipart \
  pypdf \
  fastapi \
  uvicorn
```

> **Nota sobre bcrypt**: Usa la versión `4.0.1` específicamente. Las versiones `5.x` tienen incompatibilidades con el sistema de autenticación.

---

## 4. Configurar variables de entorno

Crea o edita el archivo `.env` en la raíz del proyecto:

```env
# === YA EXISTENTES ===
GEMINI_FALLBACK_MODELS="gemini-2.0-flash,gemini-2.0-flash-exp,gemini-2.0-flash-001,gemini-1.5-flash-latest,gemini-1.5-flash,gemini-1.5-flash-002,gemini-1.5-pro"
REACTION_API_BASE_URL="http://TU_IP:8001"
MEDIA_API_URL="http://TU_IP:8011/get-last-media/"

# === NUEVA — OBLIGATORIA ===
SECRET_KEY="cadena_aleatoria_segura_de_64_caracteres"

# GOOGLE_API_KEY: ya NO es necesaria en el .env.
# Cada bot usa la apikey que envía el cliente en el campo `apikey` de la petición de WhatsApp.
# Esa clave se guarda automáticamente en la BD del bot y se reutiliza para el RAG (PDFs).
```

Para generar un `SECRET_KEY` seguro:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## 5. Sobre las bases de datos (sin acción manual)

**No necesitas crear ni migrar nada.** Las tres bases de datos se generan automáticamente:

| Base de datos | Archivo generado | ¿Cuándo se crea? |
|---|---|---|
| SQLite SaaS (usuarios y bots) | `saas_database.db` | Al primer `python main.py` |
| ChromaDB (RAG, vectores de PDFs) | `chroma_db/` | Al primer documento subido |
| LangGraph Memory (historial de chat) | `threads_memory.db` | Al primer mensaje procesado |

---

## 6. Levantar el servidor

```bash
python main.py
```

Deberías ver:
```
INFO:     Started server process [XXXX]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

El panel estará disponible en: `http://TU_IP:8000` → redirige automáticamente a `/panel`

---

## 7. Crear el usuario administrador (solo una vez)

Una vez el servidor esté corriendo, **en una terminal separada** ejecuta:

```bash
python create_admin.py
```

Salida esperada:
```
[OK] Usuario creado exitosamente:
   Email: admin@autosystemprojects.site
   Rol:   admin
   ID:    1
```

Las credenciales son:
- **Email**: `admin@autosystemprojects.site`
- **Password**: `AdminVitalicio2024!`

---

## 8. Acceder al panel

1. Abre `http://TU_IP:8000` en el navegador
2. Inicia sesión con las credenciales del admin
3. Crea tus bots con `+ Nuevo Bot`
4. Sube documentos de catálogo (PDF o TXT) desde la configuración de cada bot

---

## 9. Flujo de actualización (git pull)

```bash
# 1. Entrar al contenedor
docker exec -it <NOMBRE_O_ID> bash

# 2. Traer los cambios
cd /app
git pull origin main

# 3. Actualizar dependencias si cambiaron
pip install -r requirements.txt

# 4. Solo si cambiaron archivos en frontend/src/:
cd frontend && npm install && npm run build && cd ..

# 5. Reiniciar
kill $(pgrep -f "python main.py") && python main.py &
```

---

## 10. Volúmenes Docker — Preguntas frecuentes

### ¿Se pueden agregar volúmenes a un contenedor ya existente?

**No directamente.** Docker no permite modificar la configuración de volúmenes de un contenedor que ya fue creado con `docker run` sin volúmenes. Tienes dos opciones:

#### Opción A — Copiar datos y recrear (recomendada)

Antes de eliminar el contenedor, copia los archivos importantes al **host**:

```bash
# Copiar bases de datos del contenedor al host
docker cp <NOMBRE_CONTENEDOR>:/app/saas_database.db ./saas_database.db
docker cp <NOMBRE_CONTENEDOR>:/app/chroma_db ./chroma_db
docker cp <NOMBRE_CONTENEDOR>:/app/threads_memory.db ./threads_memory.db
docker cp <NOMBRE_CONTENEDOR>:/app/Download ./Download
docker cp <NOMBRE_CONTENEDOR>:/app/.env ./.env
```

Luego recrea el contenedor con volúmenes (ver sección 11).

#### Opción B — Usar `docker commit` para preservar el estado (sin recrear)

Si **no quieres recrear** el contenedor pero sí quieres un backup manual:

```bash
# Crear una imagen snapshot del contenedor actual con todo adentro
docker commit <NOMBRE_CONTENEDOR> autosystem-backup:$(date +%Y%m%d)

# Verificar que se creó
docker images | grep autosystem-backup
```

> Esto guarda el estado completo del contenedor como imagen, pero **no es lo mismo que volúmenes**. Los datos seguirán viviendo dentro del contenedor, no en el host.

---

## 11. Recrear el contenedor desde cero (con volúmenes correctos)

Sigue estos pasos en orden para no perder datos:

### Paso 1 — Respaldar datos del contenedor actual

```bash
# Ver el nombre del contenedor activo
docker ps

# Copiar todo lo importante al directorio del host donde tienes el proyecto
docker cp <NOMBRE_CONTENEDOR>:/app/saas_database.db ./saas_database.db
docker cp <NOMBRE_CONTENEDOR>:/app/threads_memory.db ./threads_memory.db
docker cp <NOMBRE_CONTENEDOR>:/app/chroma_db ./chroma_db
docker cp <NOMBRE_CONTENEDOR>:/app/Download ./Download
docker cp <NOMBRE_CONTENEDOR>:/app/.env ./.env
```

### Paso 2 — Detener y eliminar el contenedor viejo

```bash
docker stop <NOMBRE_CONTENEDOR>
docker rm <NOMBRE_CONTENEDOR>
```

### Paso 3 — Asegurarte de tener el `docker-compose.yml` correcto

Crea o actualiza el `docker-compose.yml` en la raíz del proyecto:

```yaml
version: "3.8"

services:
  wa-bot:
    build: .
    container_name: autosystem-wa-bot
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./saas_database.db:/app/saas_database.db
      - ./chroma_db:/app/chroma_db
      - ./threads_memory.db:/app/threads_memory.db
      - ./Download:/app/Download
    restart: unless-stopped
```

### Paso 4 — Actualizar el `requirements.txt`

Asegúrate de que el `requirements.txt` incluya los nuevos paquetes antes de reconstruir:

```text
langchain-chroma>=1.1.0
langchain-text-splitters>=1.1.0
langchain-community>=0.4.0
langchain-google-genai>=2.0.0
langgraph>=0.4.0
sqlalchemy>=2.0.0
bcrypt==4.0.1
python-jose[cryptography]>=3.3.0
python-multipart>=0.0.9
pypdf>=4.0.0
```

### Paso 5 — Reconstruir la imagen y levantar

```bash
# Reconstruir la imagen (toma los cambios de requirements.txt y el código)
docker-compose build --no-cache

# Levantar el contenedor nuevo
docker-compose up -d

# Verificar que está corriendo
docker ps
docker logs autosystem-wa-bot -f
```

### Paso 6 — Verificar datos restaurados

Entra al nuevo contenedor y confirma que los archivos existen:

```bash
docker exec -it autosystem-wa-bot bash
ls -lh saas_database.db threads_memory.db chroma_db/ Download/
```

### Paso 7 — Verificar el admin (solo si la DB fue recreada desde cero)

Si el `saas_database.db` fue copiado del contenedor anterior, el admin **ya existe** y no necesitas hacer nada.

Si empezaste con una DB vacía:

```bash
docker exec -it autosystem-wa-bot python create_admin.py
```

---

## 12. Rutas principales

| Ruta | Descripción |
|---|---|
| `http://IP:8000/` | Redirige automáticamente al panel |
| `http://IP:8000/panel` | Panel SaaS (React) |
| `http://IP:8000/docs` | Documentación Swagger de la API |
| `http://IP:8000/api/auth/login` | Iniciar sesión (devuelve JWT) |
| `http://IP:8000/api/bots/` | CRUD de configuraciones de bots |
| `http://IP:8000/wa/process` | Endpoint de recepción de mensajes WhatsApp |

---

## Resumen rápido (cheatsheet)

```bash
# ── PRIMERA VEZ ──────────────────────────────────────────────
git clone ... /app && cd /app
# Editar .env con GOOGLE_API_KEY y SECRET_KEY
docker-compose build --no-cache
docker-compose up -d
docker exec -it autosystem-wa-bot python create_admin.py

# ── ACTUALIZACIÓN SIMPLE (sin recrear) ───────────────────────
docker exec -it autosystem-wa-bot bash
git pull origin main
pip install -r requirements.txt
# Si cambió el frontend:
cd frontend && npm run build && cd ..
kill $(pgrep -f "python main.py") && python main.py &

# ── RECREAR CONTENEDOR (con backup) ──────────────────────────
docker cp <CONTENEDOR>:/app/saas_database.db .
docker cp <CONTENEDOR>:/app/chroma_db .
docker cp <CONTENEDOR>:/app/threads_memory.db .
docker cp <CONTENEDOR>:/app/Download .
docker cp <CONTENEDOR>:/app/.env .
docker stop <CONTENEDOR> && docker rm <CONTENEDOR>
docker-compose build --no-cache && docker-compose up -d
```
