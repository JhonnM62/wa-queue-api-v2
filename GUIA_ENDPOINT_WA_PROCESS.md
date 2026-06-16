# Guía del Endpoint `/wa/process`

Esta guía documenta todos los parámetros disponibles para el endpoint `POST /wa/process` del sistema de procesamiento de mensajes de WhatsApp.

## 📋 Parámetros Requeridos

Los siguientes parámetros son **obligatorios** y deben incluirse en cada petición:

### `lineaWA`
- **Tipo**: `string`
- **Descripción**: Número de teléfono de WhatsApp (sin el símbolo +)
- **Ejemplo**: `"573001234567"`

### `mensaje_reciente`
- **Tipo**: `string`
- **Descripción**: El mensaje más reciente enviado por el usuario
- **Ejemplo**: `"Hola, necesito ayuda con mi pedido"`

### `userbot`
- **Tipo**: `string`
- **Descripción**: Identificador único del bot/usuario que procesará el mensaje
- **Ejemplo**: `"bot_ventas_01"`

### `apikey`
- **Tipo**: `string`
- **Descripción**: Clave API para autenticación del servicio
- **Ejemplo**: `"sk-1234567890abcdef"`

### `server`
- **Tipo**: `string`
- **Descripción**: URL del servidor de WhatsApp API
- **Ejemplo**: `"https://api.whatsapp.com"`

### `numerodemensajes`
- **Tipo**: `integer`
- **Descripción**: Número de mensajes del historial a considerar para el contexto
- **Ejemplo**: `10`

### `promt`
- **Tipo**: `string`
- **Descripción**: Prompt o instrucciones para la IA
- **Ejemplo**: `"Eres un asistente de ventas amigable y profesional"`

### `token`
- **Tipo**: `string`
- **Descripción**: Token de acceso para la API de WhatsApp
- **Ejemplo**: `"EAABwzLixnjYBO..."`

### `pais`
- **Tipo**: `string`
- **Descripción**: País del usuario para configuración de zona horaria
- **Valores soportados**: Ver sección de [Países Soportados](#países-soportados)
- **Ejemplo**: `"colombia"`

### `idioma`
- **Tipo**: `string`
- **Descripción**: Idioma para las respuestas de la IA
- **Ejemplo**: `"español"`

## ⚙️ Parámetros Opcionales

Los siguientes parámetros tienen valores por defecto y son opcionales:

### `delay_seconds`
- **Tipo**: `float`
- **Valor por defecto**: `7.0`
- **Descripción**: Tiempo de espera en segundos antes de procesar el mensaje
- **Ejemplo**: `5.5`

### `temperature`
- **Tipo**: `float`
- **Valor por defecto**: `0.5`
- **Rango**: `0.0 - 1.0`
- **Descripción**: Controla la creatividad de las respuestas de la IA (0 = más determinista, 1 = más creativo)
- **Ejemplo**: `0.7`

### `topP`
- **Tipo**: `float`
- **Valor por defecto**: `0.95`
- **Rango**: `0.0 - 1.0`
- **Descripción**: Controla la diversidad de las respuestas usando nucleus sampling
- **Ejemplo**: `0.9`

### `maxOutputTokens`
- **Tipo**: `integer`
- **Valor por defecto**: `4096`
- **Descripción**: Número máximo de tokens en la respuesta de la IA
- **Ejemplo**: `2048`

### `pause_timeout_minutes`
- **Tipo**: `integer`
- **Valor por defecto**: `0`
- **Descripción**: Tiempo en minutos para auto-despausar un contacto (0 = sin timeout)
- **Ejemplo**: `30`

### `ai_model`
- **Tipo**: `string`
- **Valor por defecto**: `"gemini-2.0-flash"`
- **Descripción**: Modelo de IA a utilizar
- **Modelos disponibles**: Ver sección de [Modelos de IA](#modelos-de-ia)
- **Ejemplo**: `"gemini-2.5-pro"`

### `thinking_budget`
- **Tipo**: `integer`
- **Valor por defecto**: `0`
- **Descripción**: Presupuesto de razonamiento para modelos Gemini 2.5 (-1 = activar, 0 = desactivar)
- **Valores válidos**: `-1`, `0`
- **Ejemplo**: `-1`

## 🧠 Parámetros de Estados de Conversación

Estos parámetros permiten enviar directamente el contenido de estados de conversación como strings:

### `estados_generales`
- **Tipo**: `string` (opcional)
- **Valor por defecto**: `null`
- **Descripción**: String con contenido de estados generales de conversación para incluir en el prompt
- **Formato**: Debe contener estados en formato `- "nombre_estado": descripción.`
- **Ejemplo**: 
```json
"estados_generales": "- \"bienvenida_inicial\": Al inicio de una nueva conversación. - \"identificando_necesidad_cliente\": Comprendiendo el problema/pregunta inicial del cliente."
```

### `estados_especificos`
- **Tipo**: `string` (opcional)
- **Valor por defecto**: `null`
- **Descripción**: String con contenido de estados específicos de flujo (venta, reserva, procesos con objetivo claro)
- **Formato**: Debe contener estados en formato `- "nombre_estado": descripción.`
- **Ejemplo**: 
```json
"estados_especificos": "- \"inicio_flujo_objetivo\": El cliente expresa interés o inicia un proceso con un objetivo claro. - \"calificando_lead_o_necesidad_flujo\": Evaluando la idoneidad o detalles específicos para el flujo."
```

**💡 Uso recomendado de Estados:**
- Usa `estados_generales` para conversaciones de soporte general o consultas
- Usa `estados_especificos` para procesos de venta, reservas o flujos con objetivos específicos
- Puedes enviar ambos para máxima flexibilidad en el manejo de estados
- El sistema organizará automáticamente el contenido si viene desordenado

## 🔔 Parámetros de Notificaciones

Estos parámetros controlan el sistema de notificaciones automáticas:

### `lineaogruponotificacion`
- **Tipo**: `string` (opcional)
- **Valor por defecto**: `null`
- **Descripción**: Línea de WhatsApp o grupo donde enviar notificaciones
- **Ejemplo**: `"573009876543"`

### `lineaogrupo`
- **Tipo**: `boolean` (opcional)
- **Valor por defecto**: `null`
- **Descripción**: Indica si el destino de notificación es un grupo (true) o línea individual (false)
- **Ejemplo**: `false`

### `estado`
- **Tipo**: `string` (opcional)
- **Valor por defecto**: `null`
- **Descripción**: Estado que debe coincidir con `estado_conversacion` de la IA para activar notificación
- **Ejemplo**: `"pedido_completado"`

### `activarnotificacion`
- **Tipo**: `boolean` (opcional)
- **Valor por defecto**: `false`
- **Descripción**: Activa o desactiva el sistema de notificaciones
- **Ejemplo**: `true`

## 📝 Ejemplo Completo

```json
{
  "lineaWA": "573001234567",
  "mensaje_reciente": "Hola, quiero hacer un pedido",
  "userbot": "bot_ventas_01",
  "apikey": "sk-1234567890abcdef",
  "server": "https://api.whatsapp.com",
  "numerodemensajes": 15,
  "promt": "Eres un asistente de ventas especializado en productos tecnológicos. Sé amigable, profesional y ayuda al cliente a encontrar lo que necesita.",
  "token": "EAABwzLixnjYBO...",
  "pais": "colombia",
  "idioma": "español",
  "delay_seconds": 5.0,
  "temperature": 0.7,
  "topP": 0.9,
  "maxOutputTokens": 2048,
  "pause_timeout_minutes": 30,
  "ai_model": "gemini-2.5-pro",
  "thinking_budget": -1,
  "estados_generales": "- \"bienvenida_inicial\": Al inicio de una nueva conversación. - \"identificando_necesidad_cliente\": Comprendiendo el problema/pregunta inicial del cliente. - \"recolectando_informacion_adicional\": Solicitando más detalles al cliente.",
  "estados_especificos": "- \"inicio_flujo_objetivo\": El cliente expresa interés o inicia un proceso con un objetivo claro. - \"calificando_lead_o_necesidad_flujo\": Evaluando la idoneidad o detalles específicos para el flujo.",
  "lineaogruponotificacion": "573009876543",
  "lineaogrupo": false,
  "estado": "pedido_completado",
  "activarnotificacion": true
}
```

## 🌍 Países Soportados

Los siguientes países están soportados para el parámetro `pais`:

| País | Código | Zona Horaria |
|------|--------|--------------|
| Argentina | `"argentina"` | America/Argentina/Buenos_Aires |
| Bolivia | `"bolivia"` | America/La_Paz |
| Brasil | `"brasil"` | America/Sao_Paulo |
| Chile | `"chile"` | America/Santiago |
| Colombia | `"colombia"` | America/Bogota |
| Costa Rica | `"costa rica"` | America/Costa_Rica |
| Cuba | `"cuba"` | America/Havana |
| Ecuador | `"ecuador"` | America/Guayaquil |
| El Salvador | `"el salvador"` | America/El_Salvador |
| Guatemala | `"guatemala"` | America/Guatemala |
| Honduras | `"honduras"` | America/Tegucigalpa |
| México | `"mexico"` | America/Mexico_City |
| Nicaragua | `"nicaragua"` | America/Managua |
| Panamá | `"panama"` | America/Panama |
| Paraguay | `"paraguay"` | America/Asuncion |
| Perú | `"peru"` | America/Lima |
| Puerto Rico | `"puerto rico"` | America/Puerto_Rico |
| República Dominicana | `"republica dominicana"` | America/Santo_Domingo |
| España | `"españa"` | Europe/Madrid |
| Uruguay | `"uruguay"` | America/Montevideo |
| Venezuela | `"venezuela"` | America/Caracas |
| Estados Unidos | `"usa"` | America/New_York |
| Canadá | `"canada"` | America/Toronto |
| Reino Unido | `"uk"` | Europe/London |
| Francia | `"france"` | Europe/Paris |
| Alemania | `"germany"` | Europe/Berlin |
| Italia | `"italy"` | Europe/Rome |
| Portugal | `"portugal"` | Europe/Lisbon |
| Australia | `"australia"` | Australia/Sydney |
| India | `"india"` | Asia/Kolkata |
| China | `"china"` | Asia/Shanghai |
| Japón | `"japan"` | Asia/Tokyo |

## 🤖 Modelos de IA

### Modelos Disponibles
- `gemini-2.0-flash` (por defecto)
- `gemini-2.0-flash-exp`
- `gemini-2.0-flash-001`
- `gemini-1.5-flash-latest`
- `gemini-1.5-flash`
- `gemini-1.5-flash-002`
- `gemini-1.5-pro`
- `gemini-2.5-flash`
- `gemini-2.5-pro`

### Modelos con Soporte para Thinking Config
Solo los siguientes modelos soportan el parámetro `thinking_budget`:
- `gemini-2.5-flash`
- `gemini-2.5-pro`

## ⚠️ Consideraciones Importantes

### Interdependencias de Parámetros

1. **Sistema de Notificaciones**: Para que funcione correctamente, se requiere:
   - `activarnotificacion: true`
   - `lineaogruponotificacion` debe tener un valor válido
   - `estado` debe coincidir con el `estado_conversacion` devuelto por la IA

2. **Thinking Budget**: Solo funciona con modelos Gemini 2.5:
   - Si usas `thinking_budget: -1` con otros modelos, será ignorado
   - Valores válidos: `-1` (activar) o `0` (desactivar)

### Límites y Validaciones

- **`lineaWA`**: Debe ser un número de teléfono válido sin el símbolo +
- **`temperature`**: Debe estar entre 0.0 y 1.0
- **`topP`**: Debe estar entre 0.0 y 1.0
- **`maxOutputTokens`**: Valor mínimo recomendado: 256, máximo: 8192
- **`numerodemensajes`**: Valor mínimo: 1, máximo recomendado: 50

### Formatos Especiales

- **Números de teléfono**: Sin espacios, guiones o símbolos. Solo dígitos.
- **URLs del servidor**: Deben incluir el protocolo (http:// o https://)
- **Tokens**: Cadenas alfanuméricas largas, mantener confidenciales

### Mejores Prácticas

1. **Seguridad**: Nunca expongas `apikey` o `token` en logs o interfaces públicas
2. **Performance**: Ajusta `numerodemensajes` según tus necesidades (más mensajes = más contexto pero mayor latencia)
3. **Costos**: Modelos más avanzados como `gemini-2.5-pro` consumen más recursos
4. **Notificaciones**: Úsalas con moderación para evitar spam

---

*Última actualización: Enero 2025*
*Versión del sistema: 1.2.30_gemini_thinking_config*