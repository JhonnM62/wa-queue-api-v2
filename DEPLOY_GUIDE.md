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

## 11. Recrear el contenedor desde cero (Volumen de directorio completo)

Sigue estos pasos en tu VPS para actualizar o recrear el contenedor usando la técnica de volumen de directorio completo (Live Reload).

### Paso 1 — Detener y eliminar el contenedor viejo

```bash
docker rm -f wa-queue-api-v2.2
```

### Paso 2 — Obtener la última versión del código

Si ya tienes la carpeta, asegúrate de tener los últimos cambios de GitHub:

```bash
cd wa-queue-api-v2
git pull origin main
```

*(Si no la tenías, clónala con `git clone https://github.com/JhonnM62/wa-queue-api-v2.git`)*

### Paso 3 — Reconstruir la imagen

Construye la imagen para que instale las últimas librerías de `requirements.txt`:

```bash
docker build -t wa-queue-api-v2.2 .
```

### Paso 4 — Levantar el contenedor mapeando el directorio

Ejecuta este comando en la raíz de tu proyecto. El puerto expuesto será `8014` y se vinculará todo tu directorio actual (`$PWD`) al contenedor:

```bash
docker run -d --name wa-queue-api-v2.2 -p 8014:8000 -v "$PWD":/app wa-queue-api-v2.2
```

### Paso 5 — Verificar datos restaurados

Verifica que el contenedor esté corriendo correctamente:

```bash
docker ps
docker logs wa-queue-api-v2.2 -f
```

---

## 12. Rutas principales

| Ruta | Descripción |
|---|---|
| `http://IP:8014/` | Redirige automáticamente al panel |
| `http://IP:8014/panel` | Panel SaaS (React) |
| `http://IP:8014/docs` | Documentación Swagger de la API |
| `http://IP:8014/api/auth/login` | Iniciar sesión (devuelve JWT) |
| `http://IP:8014/api/bots/` | CRUD de configuraciones de bots |
| `http://IP:8014/wa/process` | Endpoint de recepción de mensajes WhatsApp |

---

## Resumen rápido (cheatsheet)

```bash
# ── PRIMERA VEZ / RECREAR CONTENEDOR ─────────────────────────
git clone https://github.com/JhonnM62/wa-queue-api-v2.git
cd wa-queue-api-v2
# Editar .env con tus secretos
docker build -t wa-queue-api-v2.2 .
docker run -d --name wa-queue-api-v2.2 -p 8014:8000 -v "$PWD":/app wa-queue-api-v2.2
docker exec -it wa-queue-api-v2.2 python create_admin.py

# ── ACTUALIZACIÓN RÁPIDA (Live Reload) ───────────────────────
cd wa-queue-api-v2
git pull origin main
docker restart wa-queue-api-v2.2
```

---

## 13. Transferir datos desde Windows al VPS

Si tienes archivos locales (como historiales de conversaciones) y necesitas subirlos al servidor, puedes usar uno de estos dos métodos:

### Método 1: Consola (Rápido, comando `scp`)

Dado que la carpeta de descargas no existe por defecto, primero **entra a la consola de tu VPS** y crea la estructura:

```bash
mkdir -p /root/wa-queue-api-v2/Download/AutoSystem/historial/
```

Luego, abre PowerShell en tu **Windows local** y envía el contenido directamente a esa carpeta:

```powershell
scp -r "C:\datos empleados\conversaciones_locales\*" root@TU_IP_VPS:/root/wa-queue-api-v2/Download/AutoSystem/historial/
```
*(Si usas CMD en lugar de PowerShell y el asterisco falla, transfiere la carpeta completa y luego renómbrala en el servidor).*

### Método 2: Interfaz visual (Recomendado)

1. Descarga e instala **WinSCP** en tu Windows.
2. Conéctate con la IP, usuario `root` y tu contraseña del VPS.
3. En la ventana derecha (VPS), ve a `/root/wa-queue-api-v2/` y crea las carpetas manualmente si no existen (`Download`, luego `AutoSystem`, luego `historial`).
4. En la ventana izquierda (Windows), entra a `C:\datos empleados\conversaciones_locales`.
5. Selecciona todo el contenido y arrástralo adentro de la carpeta `historial` en la ventana derecha.

*Nota: Gracias al volumen `$PWD:/app`, cualquier archivo que transfieras a la carpeta del VPS será detectado por el contenedor de Docker de forma instantánea.*
