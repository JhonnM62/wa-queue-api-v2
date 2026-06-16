# 🐛 Solución al Problema de PayPal Client ID Undefined

## Problema Identificado

El SDK de PayPal falla en producción con el error:
```
client-id not recognized for either production or sandbox: undefined
```

**Causa:** La variable de entorno `NEXT_PUBLIC_PAYPAL_CLIENT_ID` no se está pasando correctamente al contenedor de producción.

## Análisis del Problema

### ✅ Funciona en Local
- URL: `https://www.paypal.com/sdk/js?client-id=AQyZVZ3rtLUqTBE4o-TW8QSwhBw8DFXHs_emPiveghITkT3mj5eslyItlVzUcWKLlTQMKi_1-px5K734&currency=USD`
- Status: 200 OK

### ❌ Falla en Producción
- URL: `https://www.paypal.com/sdk/js?client-id=undefined&currency=USD`
- Status: 400 Bad Request

## Solución Implementada

### 1. Sistema de Debug Agregado

**Frontend (`upgrade/page.tsx`):**
```javascript
// 🐛 DEBUG: Verificar variables de entorno de PayPal
const clientId = process.env.NEXT_PUBLIC_PAYPAL_CLIENT_ID
console.log('🐛 [DEBUG] PayPal Client ID:', clientId ? `${clientId.substring(0, 10)}...` : 'UNDEFINED')
console.log('🐛 [DEBUG] Todas las variables NEXT_PUBLIC:', Object.keys(process.env).filter(key => key.startsWith('NEXT_PUBLIC_')))
```

**Backend (`app.js`):**
```javascript
// 🐛 DEBUG: Mostrar variables de entorno importantes al iniciar
console.log('🐛 [DEBUG-BACKEND] PAYPAL_CLIENT_ID:', process.env.PAYPAL_CLIENT_ID ? `${process.env.PAYPAL_CLIENT_ID.substring(0, 10)}...` : 'NO CONFIGURADO')
```

### 2. Corrección en GitHub Actions

**Problema:** El workflow no pasaba `NEXT_PUBLIC_PAYPAL_CLIENT_ID` al contenedor.

**Solución:** Agregado en `.github/workflows/publish.yml`:
```yaml
--env NEXT_PUBLIC_PAYPAL_CLIENT_ID="${{ secrets.NEXT_PUBLIC_PAYPAL_CLIENT_ID }}" \
```

### 3. Documentación de Variables de Entorno

**Archivos creados:**
- `appboots/.env.example` - Variables del frontend
- `baileys-api/.env.example` - Variables del backend

## Configuración Requerida en GitHub

### Secrets and Variables > Actions > Variables

**Variables requeridas:**
- `NEXT_PUBLIC_PAYPAL_CLIENT_ID` - Client ID de PayPal para el frontend
- `NEXT_PUBLIC_API_URL` - URL del backend API

**Secrets requeridos:**
- `ENV_FILE_CONTENT` - Contenido del archivo .env
- `PAYPAL_CLIENT_SECRET` - Secret de PayPal para el backend

## Verificación Post-Deploy

1. **Revisar logs del contenedor frontend:**
   ```bash
   docker logs appboots-frontend
   ```
   Buscar líneas que empiecen con `🐛 [DEBUG]`

2. **Revisar logs del contenedor backend:**
   ```bash
   docker logs baileys-api
   ```
   Buscar líneas que empiecen con `🐛 [DEBUG-BACKEND]`

3. **Verificar en el navegador:**
   - Abrir DevTools > Console
   - Buscar logs de debug de PayPal
   - Verificar que el client-id no sea 'undefined'

## Próximos Pasos

1. Hacer commit y push de estos cambios
2. Verificar que `NEXT_PUBLIC_PAYPAL_CLIENT_ID` esté configurado en GitHub Actions
3. Hacer un nuevo deploy
4. Verificar los logs de debug para confirmar que las variables se cargan correctamente
5. Una vez confirmado el funcionamiento, remover los logs de debug

## Notas Importantes

- Las variables `NEXT_PUBLIC_*` en Next.js se embeben en el build del cliente
- Deben pasarse como variables de entorno del contenedor, no solo en el archivo .env
- El debug agregado es temporal y debe removerse una vez solucionado el problema