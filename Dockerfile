# Usa una imagen base oficial de Python ligera
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia solo el archivo de requerimientos primero para aprovechar el cache de Docker
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de los archivos de tu proyecto al directorio de trabajo
COPY . .

# Crea los directorios de persistencia si no existen en la copia inicial
# Esto es una capa de seguridad si no confías en que la copia COPY . . los incluya vacíos
# O si se crean en tiempo de ejecución y necesitas que persistan
RUN mkdir -p Download/AutoSystem/conf_2 \
           Download/AutoSystem/historial \
           Download/AutoSystem/mensajeactual

# Expone el puerto en el que Uvicorn escuchará dentro del contenedor
# Usaremos el puerto 8000 por convención interna
EXPOSE 8000

# Define el comando por defecto para ejecutar la aplicación con Uvicorn
# --host 0.0.0.0 es crucial para que la API sea accesible desde fuera del contenedor
# --port 8000 especifica el puerto interno
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]