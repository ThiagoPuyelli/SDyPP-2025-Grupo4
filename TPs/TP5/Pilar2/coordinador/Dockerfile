# Imagen base oficial con Python 3.11
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos necesarios
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Exponer el puerto de FastAPI (por defecto 8000)
EXPOSE 8000

# Comando para iniciar FastAPI con hot reload (solo para desarrollo)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Comando para producción (sin reload)
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
