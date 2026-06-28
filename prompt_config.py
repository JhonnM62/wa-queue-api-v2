EXAMPLE_JSON_STRUCTURE = """{
    "estado_conversacion": "presentando_propuesta_venta",
    "1": {
        "tipo": "mensaje",
        "mensaje": "¡Excelente elección! El Paquete Premium incluye [característica A], [característica B] y acceso exclusivo a [beneficio C]. ¿Te gustaría proceder con este paquete?"
    },
    "2": {
        "tipo": "imagen",
        "ruta2": "URL_IMAGEN_PAQUETE_PREMIUM.jpg",
        "mensaje": "Así se ve nuestro Paquete Premium."
    }
}"""

# Estados de conversación disponibles
ESTADOS_GENERALES = """
    **Estados Generales de Conversación:**
    - "bienvenida_inicial": Al inicio de una nueva conversación.
    - "identificando_necesidad_cliente": Comprendiendo el problema/pregunta inicial del cliente.
    - "recolectando_informacion_adicional": Solicitando más detalles al cliente.
    - "procesando_solicitud_cliente": Trabajando internamente en la respuesta o acción.
    - "proporcionando_respuesta_solucion": Entregando la información o solución principal a una consulta.
    - "esperando_confirmacion_cliente": Tras dar una respuesta, esperando feedback o más preguntas.
    - "aclarando_dudas_adicionales": Respondiendo a preguntas de seguimiento del cliente.
    - "resolucion_consulta_confirmada": El cliente confirma que su consulta puntual está resuelta.
    - "ofreciendo_ayuda_adicional": Preguntando si hay algo más en lo que se pueda ayudar.
    - "despedida_conversacion_finalizada": La conversación está concluyendo.
    - "escalando_a_humano": Si la IA determina que necesita intervención de un agente humano.
    - "error_ia_gestionando": Si la IA encuentra un error interno pero intenta gestionarlo."""

ESTADOS_ESPECIFICOS = """
    **Estados Específicos de Flujo (ej. Venta, Reserva, Proceso con objetivo claro solo coloca los que te especifica aca solo coloca los que te especifica aca IMPORTANTE NO INVENTES OTROS ESTADOS QUE NO ESTEN EN ESTA LISTA):**
    - "inicio_flujo_objetivo": El cliente expresa interés o inicia un proceso con un objetivo claro (ej. "quiero comprar", "necesito reservar").
    - "calificando_lead_o_necesidad_flujo": Evaluando la idoneidad o detalles específicos para el flujo (ej. presupuesto para venta, fechas para reserva).
    - "presentando_propuesta_o_opciones": Mostrando productos, servicios, planes, opciones de reserva, etc.
    - "manejando_objeciones_flujo": Respondiendo a dudas o preocupaciones sobre la propuesta.
    - "negociando_terminos_flujo": Discutiendo precio, condiciones, personalizaciones.
    - "esperando_decision_final_cliente_flujo": El cliente está evaluando la propuesta final osea esta detallando el resumen del pedido antes de comprometerse o confirmar (ej. en el mensaje se envio lo que pidio el cliente con totales y se solicita la informacion para despachar el pedido) SOLO ENVIA UNA VEZ ESETE ESTADO O CUANDO SEA NESESARIO.
    - "flujo_exitoso_completado": El objetivo principal se ha logrado (ej. venta cerrada y pagada, reserva confirmada).
    - "flujo_no_exitoso_cancelado_o_perdido": El cliente declina, abandona el proceso, o no se cumplen condiciones.
    - "seguimiento_post_flujo": Interacción después de un flujo exitoso (ej. postventa, encuesta de satisfacción)."""

def get_prompt_template(estados_generales=None, estados_especificos=None):
    """
    Genera el template de prompt con estados opcionales según los parámetros.
    
    Args:
        estados_generales: String opcional con contenido de estados generales
        estados_especificos: String opcional con contenido de estados específicos
    """
    
    # Construir la sección de estados según los parámetros
    estados_section = "\nEstados de conversación sugeridos (elige el más apropiado o uno similar si es necesario, adaptándolo al contexto específico del proceso, solo coloca los que te especifica aca IMPORTANTE NO INVENTES OTROS ESTADOS QUE NO ESTEN EN ESTA LISTA):"
    
    if estados_generales or estados_especificos:
        # Si se envían estados personalizados, usar esos
        if estados_generales:
            # Organizar el string de estados generales si es necesario
            estados_generales_organizados = organizar_estados_string(estados_generales)
            if estados_generales_organizados:
                estados_section += "\n**Estados Generales de Conversación:**\n" + estados_generales_organizados
        
        if estados_especificos:
            # Organizar el string de estados específicos si es necesario
            estados_especificos_organizados = organizar_estados_string(estados_especificos)
            if estados_especificos_organizados:
                estados_section += "\n**Estados Específicos de Flujo:**\n" + estados_especificos_organizados
    else:
        # Si no se envían estados, usar los estados por defecto
        estados_section += "\n" + ESTADOS_GENERALES
        estados_section += "\n" + ESTADOS_ESPECIFICOS
    
    return f"""{{{"system_rules_base"}}}
_________________________________________
Historial de la conversación reciente (últimos {{{"num_mensajes_shown"}}} mensajes relevantes):
{{{"historial_texto"}}}
_________________________________________
Instrucciones de formato de respuesta JSON:
Genera SOLAMENTE un objeto JSON válido. La estructura debe ser un diccionario.
DEBE incluir una clave "estado_conversacion" a nivel superior, cuyo valor sea un string describiendo el estado actual de la conversación y del flujo principal que se está gestionando (ej. una venta, una consulta, una reserva) desde la perspectiva de la IA.
La IA debe inferir este estado basándose en el contexto de la conversación y las interacciones.{estados_section}

Además de "estado_conversacion", el JSON debe contener claves numéricas secuenciales como strings ("1", "2", "3", etc.) y los valores deben ser objetos que describen la acción a realizar para enviar al cliente.
Cada objeto de acción DEBE tener una clave "tipo". Tipos válidos y sus campos requeridos/opcionales:
    -   "mensaje":
        -   Requerido: "mensaje" (string con el texto a enviar).
    -   "imagen":
        -   Requerido: "ruta2" (string con la URL pública o ruta accesible del archivo de imagen).
        -   Opcional: "mensaje" (string, para el pie de foto o caption).
        -   Opcional: "nombrearchivo" (string, si es relevante para el envío).
    -   "video":
        -   Requerido: "ruta2" (string con la URL pública o ruta accesible del archivo de video).
        -   Opcional: "mensaje" (string, para el pie de foto o caption).
    -   "audio":
        -   Requerido: "ruta2" (string con la URL pública o ruta accesible del archivo de audio).
        -   Opcional: "ptt" (boolean, true si es Push-to-Talk).
    -   "pdf" (o "documento"):
        -   Requerido: "ruta2" (string con la URL pública o ruta accesible del archivo PDF/documento).
        -   Requerido: "nombrearchivo" (string, nombre del archivo con extensión, ej: "factura.pdf").
        -   Opcional: "mensaje" (string, para el pie de foto o caption).
    -   "ubicacion":
        -   Requerido: "lat" (float, latitud).
        -   Requerido: "long" (float, longitud).

Ejemplo de la estructura JSON de respuesta esperada:
{{{"example_json_structure_placeholder"}}}

Importante:
-   El campo "estado_conversacion" NO se envía al cliente, es para registro interno y toma de decisiones de la IA.
-   Condensa la respuesta al cliente tanto como sea posible en la primera acción ("1"), pero usa acciones adicionales ("2", "3", ...) si es necesario separar lógicamente los envíos.
-   NO incluyas explicaciones, comentarios o texto fuera del objeto JSON. Tu respuesta DEBE ser únicamente el objeto JSON.
-   Debes usar exclusivamente el campo 'ruta2' de las acciones para enviar archivos adjuntos cuando se proporcionen rutas o URLs específicas de imágenes, audios, videos, documentos o coordenadas (latitud y longitud). No incluyas estas rutas directamente en el mensaje, solo úsalas si la conversación lo requiere.
-   REGLA ESTRICTA DE MEMORIA: Revisa exhaustivamente el historial de conversación antes de hacer una pregunta. NO vuelvas a pedir un dato o información que el cliente ya haya proporcionado anteriormente.

Considera TODO el contexto (reglas, historial, mensaje actual) para generar la respuesta más adecuada y útil, infiriendo el estado de la conversación y del flujo principal según la interacción.

---
**INSTRUCCIÓN CRÍTICA DE IDIOMA:** Responde ÚNICAMENTE en el idioma especificado: **{{{"idioma_respuesta"}}}**. Todo el contenido textual dentro del JSON (mensajes, captions) DEBE estar en **{{{"idioma_respuesta"}}}**.
"""

# Template por defecto (mantiene compatibilidad con código existente)
PROMPT_TEMPLATE = get_prompt_template()

def organizar_estados_string(estados_string: str) -> str:
    """
    Organiza un string de estados que puede venir desordenado.
    Extrae los estados individuales y los organiza de manera estructurada.
    Soporta múltiples formatos:
    - Formato anterior: - "estado": descripción
    - Formato nuevo: 1. Estado\n2. Otro estado
    
    Args:
        estados_string: String que contiene estados, posiblemente desordenados
        
    Returns:
        String organizado con los estados estructurados
    """
    if not estados_string or not isinstance(estados_string, str):
        return ""
    
    import re
    
    # Limpiar el string de entrada
    estados_string = estados_string.strip()
    
    # Patrón para encontrar estados en formato anterior: - "estado": descripción
    patron_estado_anterior = r'-\s*["\']([^"\']+)["\']\s*:\s*([^-]+?)(?=\s*-\s*["\']|$)'
    
    # Encontrar todos los estados con el patrón anterior
    estados_encontrados = re.findall(patron_estado_anterior, estados_string, re.DOTALL | re.MULTILINE)
    
    if estados_encontrados:
        # Procesar formato anterior
        estados_organizados = []
        for nombre_estado, descripcion in estados_encontrados:
            # Limpiar nombre y descripción
            nombre_limpio = nombre_estado.strip()
            descripcion_limpia = descripcion.strip().rstrip('.')
            
            # Formatear el estado
            estado_formateado = f'     - "{nombre_limpio}": {descripcion_limpia}.'
            estados_organizados.append(estado_formateado)
        
        # Unir todos los estados organizados
        resultado = '\n'.join(estados_organizados)
        return resultado
    
    # Si no se encuentra el formato anterior, intentar con formato nuevo (números con saltos de línea)
    # Patrón para formato: 1. Estado\n2. Otro estado
    patron_estado_nuevo = r'(\d+)\.\s*([^\n\r]+)'
    
    # Encontrar todos los estados con el patrón nuevo
    estados_nuevos = re.findall(patron_estado_nuevo, estados_string)
    
    if estados_nuevos:
        # Procesar formato nuevo
        estados_organizados = []
        for numero, descripcion in estados_nuevos:
            # Limpiar descripción
            descripcion_limpia = descripcion.strip()
            
            # Formatear el estado manteniendo el número
            estado_formateado = f'{numero}. {descripcion_limpia}'
            estados_organizados.append(estado_formateado)
        
        # Unir todos los estados organizados
        resultado = '\n'.join(estados_organizados)
        return resultado
    
    # Si no se encuentra ningún patrón reconocido, intentar procesar después de limpiar escapes
    # Primero reemplazar \n con saltos de línea reales
    estados_con_saltos_reales = estados_string.replace('\\n', '\n')
    
    # Intentar nuevamente con el patrón nuevo después de limpiar escapes
    estados_nuevos_post_escape = re.findall(patron_estado_nuevo, estados_con_saltos_reales)
    
    if estados_nuevos_post_escape:
        # Procesar formato nuevo después de limpiar escapes
        estados_organizados = []
        for numero, descripcion in estados_nuevos_post_escape:
            # Limpiar descripción
            descripcion_limpia = descripcion.strip()
            
            # Formatear el estado manteniendo el número
            estado_formateado = f'{numero}. {descripcion_limpia}'
            estados_organizados.append(estado_formateado)
        
        # Unir todos los estados organizados
        resultado = '\n'.join(estados_organizados)
        return resultado
    
    # Si aún no se encuentra patrón, devolver limpio eliminando también \r
    return estados_con_saltos_reales.replace('\r', '').strip()