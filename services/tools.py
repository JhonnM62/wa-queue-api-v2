import os
import sys
import time
import json
import re
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import httpx
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from services.rag_service import search_knowledge
from core.phone_utils import (
    is_meta_identifier,
    clean_real_phone,
    resolve_order_phones,
    format_detalle_wa,
    format_money,
    canonical_order_signature,
    phones_match,
)


@tool
def buscar_catalogo_tool(query: str, config: RunnableConfig) -> str:
    """Busca información en el catálogo o base de conocimiento del chatbot (precios, productos, categorías).
    ES OBLIGATORIO USAR ESTA HERRAMIENTA SIEMPRE QUE VAYAS A ARMAR UN PEDIDO O DAR UN PRECIO.
    NUNCA asumas los precios ni la categoría de un producto; usa siempre esta herramienta para verificar.

    INSTRUCCIÓN CRÍTICA AL INTERPRETAR LOS RESULTADOS:
    - Cada producto en el catálogo tiene un campo CATEGORÍA. Léelo SIEMPRE antes de asignar precio o tipo.
    - Si CATEGORÍA dice 'Comida' → es hamburguesa/sánduche/perro/salchipapa. Usa su precio de comida. NO es granizado.
    - Si CATEGORÍA dice 'Granizado' → es bebida en vaso. Aplica precio de granizado.
    - Si CATEGORÍA dice 'Bebida' → es bebida embotellada/lata. No apliques cobro de tapa.
    - NUNCA confundas la categoría aunque el nombre suene similar a otra cosa.
    - Si el resultado no incluye campo CATEGORÍA, usa el precio EXACTO que aparezca en el resultado y clasifica
      según el contexto (los granizados siempre tienen 'oz' en su descripción o mencionan 'licor'/'sin licor').
    """
    bot_id = config["configurable"].get("bot_id")
    api_key = config["configurable"].get("config_dict", {}).get("api_key")
    return search_knowledge(bot_id=bot_id, query=query, api_key=api_key)


def dispatch_order_to_pos_webhooks(endpoints_json: str, payload: dict):
    """Despacha asíncronamente el pedido en segundo plano a todas las URLs de POS configuradas en paralelo."""
    if not endpoints_json:
        return

    def _send_single(ep):
        if not ep.get("activo", True) or not ep.get("url"):
            return
        url = str(ep["url"]).strip()
        if not url.startswith(("http://", "https://")):
            return
        token = ep.get("token", "")
        headers = {"Content-Type": "application/json"}
        if token:
            headers["x-webhook-token"] = token
        nombre = ep.get("nombre", url)
        try:
            with httpx.Client(timeout=15.0, verify=False) as client:
                resp = client.post(url, json=payload, headers=headers)
                print(f"[POS-WEBHOOK] Pedido {payload.get('orderIdWA')} enviado a {nombre} -> HTTP {resp.status_code}")
        except Exception as err:
            print(f"[POS-WEBHOOK] Error enviando pedido a {nombre}: {err}")

    def _send_all():
        try:
            endpoints = json.loads(endpoints_json)
            if not isinstance(endpoints, list):
                return
            for ep in endpoints:
                threading.Thread(target=_send_single, args=(ep,), daemon=True).start()
        except Exception as e:
            print(f"[POS-WEBHOOK] Error procesando pos_endpoints: {e}")

    threading.Thread(target=_send_all, daemon=True).start()



@tool
def enviar_notificacion_tool(
    resumen_general: str, 
    lista_productos: str, 
    total_a_cobrar: str, 
    direccion_envio: str, 
    metodo_pago: str, 
    telefono_cliente: str,
    nombre_cliente: str,
    config: RunnableConfig,
    es_pedido_nuevo_separado: bool = False
) -> str:
    """Envía una notificación al vendedor y guarda el pedido en el dashboard de ventas.
    Úsala SOLO cuando se haya concretado un pedido final o el cliente confirme/modifique un pedido en curso.
    Debes extraer de la conversación el resumen general, la lista de productos, el total a cobrar, la dirección, el método de pago, el teléfono de contacto del cliente (SOLO relevante cuando el remitente es un Meta ID largo; si el contacto ya viene con su número de WhatsApp normal, el sistema tomará automáticamente el número de la petición `lineaWA`), y el nombre del cliente (si lo dio, sino 'No especificado').

    PARÁMETRO 'es_pedido_nuevo_separado':
    - Pon False (por defecto) si el cliente está pidiendo por primera vez, o si está completando datos de pago/dirección, o si está cambiando o agregando productos al pedido actual en curso.
    - Pon True ÚNICAMENTE si el cliente indica de forma expresa y explícita que desea realizar un pedido completamente NUEVO y SEPARADO del anterior (por ejemplo: "quiero hacer otro pedido aparte", "este es otro pedido para otra persona").

    IMPORTANTE PARA LISTA DE PRODUCTOS (`lista_productos`):
    Escribe UNA SOLA LÍNEA concisa por cada producto pedido con el formato: `[Cantidad]x [Nombre del Producto] - $[Subtotal del ítem]`
    (Ejemplo: `1x Smirnoff de Lulo 12oz - $13.000\n2x Forloko Gold 16oz - $30.000`).
    PROHIBIDO incluir líneas separadas de "Precio Unidad", "Subtotal" o "Total sin el valor del domicilio".

    IMPORTANTE PARA DIRECCIÓN: La 'direccion_envio' DEBE ser SIEMPRE únicamente la dirección física literal y explícita del cliente (NO incluyas el número de teléfono dentro del campo dirección). Si el cliente dice "la misma dirección de siempre" o la dirección ya se mencionó en el historial, DEBES buscarla en el contexto y escribirla completa. NUNCA escribas "N/A" o "Ya registrada" si la dirección es conocida o se puede deducir del historial.
    Solo si algún dato realmente no existe o no aplica bajo ninguna circunstancia, pasa una cadena vacía.
    """
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    config_dict = configurable.get("config_dict", {})
    
    server_url = (config_dict.get("server") or "").strip()
    token = config_dict.get("token") or ""
    receiver = (config_dict.get("receiver") or config_dict.get("lineaogruponotificacion") or "").strip()
    is_group = config_dict.get("is_group", False)
    userbot = config_dict.get("userbot_identifier")
    user_phone = str(config_dict.get("user_phone") or configurable.get("phone") or "").strip()

    # Auto-detectar si el receiver es grupo por formato (@g.us o numérico de >15 dígitos)
    if not is_group and receiver:
        if "@g.us" in receiver or (receiver.isdigit() and len(receiver) > 15):
            is_group = True
    is_group = bool(is_group)

    # Determinar qué userbot enviará la notificación al grupo/línea
    notification_userbot = (
        config_dict.get("notification_userbot")
        or (config_dict.get("userbotopcional") if config_dict.get("activaruserbotopcional") else None)
        or config_dict.get("userbotopcional")
        or config_dict.get("userbot_identifier")
        or userbot
    )

    clean_server_url = (server_url or "").rstrip("/")
    if clean_server_url and not clean_server_url.startswith(("http://", "https://")):
        clean_server_url = f"http://{clean_server_url}"

    print(f"[tools/enviar_notificacion/{user_phone}] 🔔 [NOTIF-TOOL] Invocada por Gemini para {user_phone}:", flush=True)
    print(f"[tools/enviar_notificacion/{user_phone}]    - Cliente: '{nombre_cliente}' ({telefono_cliente})", flush=True)
    print(f"[tools/enviar_notificacion/{user_phone}]    - Total: {total_a_cobrar} | Pago: {metodo_pago}", flush=True)
    print(f"[tools/enviar_notificacion/{user_phone}]    - Dirección: {direccion_envio}", flush=True)
    print(f"[tools/enviar_notificacion/{user_phone}]    - Resumen: {resumen_general}", flush=True)
    print(f"[tools/enviar_notificacion/{user_phone}]    - Destino WhatsApp: receiver='{receiver}', is_group={is_group}", flush=True)
    print(f"[tools/enviar_notificacion/{user_phone}]    - Emisor WhatsApp: sender_bot='{notification_userbot}' (server: '{clean_server_url}')", flush=True)

    if not clean_server_url or not receiver or not notification_userbot:
        print(f"[tools/enviar_notificacion/{user_phone}] ❌ [NOTIF-TOOL] ABORTADO: Faltan datos (server_url='{clean_server_url}', receiver='{receiver}', notif_userbot='{notification_userbot}')", flush=True)
        return "Fallo: No se encontraron los datos del servidor, receptor o userbot para notificar."

    # 1. Guardar en el dashboard (orders.json)
    ORDERS_FILE_PATH = "./Download/AutoSystem/orders.json"
    order_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)

    detalle_completo = format_detalle_wa(lista_productos)
    total_a_cobrar = format_money(total_a_cobrar)

    phone_info = resolve_order_phones(
        api_linea_wa=user_phone,
        provided_phone=telefono_cliente,
        nombre_cliente=nombre_cliente
    )
    is_meta = phone_info["is_meta"]
    display_phone = phone_info["display_phone"]
    assigned_phone = phone_info["assigned_phone"]
    cliente_str = phone_info["cliente_str"]
    real_phone = phone_info["extracted_phone"] if is_meta else ""

    if not is_meta and telefono_cliente and telefono_cliente.strip():
        typed_digits = re.sub(r'\D', '', str(telefono_cliente))
        api_digits = re.sub(r'\D', '', str(assigned_phone))
        if typed_digits and len(typed_digits) >= 7 and typed_digits not in api_digits:
            real_local = api_digits[-10:] if len(api_digits) >= 10 else api_digits
            print(
                f"[tools/enviar_notificacion/{user_phone}] ⚠️ [PHONE-OVERRIDE] El cliente escribió '{telefono_cliente}', "
                f"pero NO es Meta ID. Se ignora el número escrito y se fuerza lineaWA='{assigned_phone}'.",
                flush=True
            )
            if direccion_envio and typed_digits in direccion_envio:
                direccion_envio = direccion_envio.replace(typed_digits, real_local)
            if resumen_general and typed_digits in resumen_general:
                resumen_general = resumen_general.replace(typed_digits, real_local)

    # --- Lógica de deduplicación y detección de modificaciones / nuevos pedidos ---
    # REGLAS INTELIGENTES:
    # 1. Ventana de modificación activa: 20 minutos (1.200.000 ms).
    #    - Si el cliente añade/cambia productos dentro de los 20 min ("aún estamos a tiempo"):
    #      Se actualiza el pedido activo in-place en el dashboard (mismo ID) y se notifica como MODIFICACIÓN / ADICIÓN.
    #    - Si el cliente pide productos adicionales después de 20 minutos:
    #      La cocina ya preparó o despachó la orden anterior; se genera un NUEVO PEDIDO independiente.
    # 2. Ventana de actualización de pago/datos: hasta 2 horas.
    #    - Si los productos y el total son IDÉNTICOS (el cliente solo aclaró método de pago, cambio o dirección),
    #      se actualiza in-place para no duplicar platos en cocina, incluso si tardó más de 20 minutos en pagar.
    # 3. Pedidos con status "Despachados": NUNCA se modifican, cualquier pedido posterior es siempre NUEVO.
    # 4. Si 'es_pedido_nuevo_separado' es True: se fuerza siempre la creación de un nuevo pedido independiente.
    MODIFICATION_WINDOW_MS = 20 * 60 * 1000        # 20 minutos ("aún estamos a tiempo")
    PAYMENT_UPDATE_WINDOW_MS = 2 * 3600 * 1000      # 2 horas para completar pago/datos de la misma orden
    is_modification = False
    existing_order_id = None
    existing_order = None
    same_items_and_total = False

    try:
        orders_data = {}
        if os.path.exists(ORDERS_FILE_PATH):
            with open(ORDERS_FILE_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    orders_data = json.loads(content)

        # Recorrer del pedido más reciente al más antiguo
        sorted_orders = sorted(
            orders_data.items(),
            key=lambda kv: kv[1].get("timestamp", 0) if isinstance(kv[1], dict) else 0,
            reverse=True
        )

        if not es_pedido_nuevo_separado:
            for oid, odata in sorted_orders:
                if not isinstance(odata, dict):
                    continue
                is_active_status = odata.get("status") in ["Recientes", "En proceso"]
                if not is_active_status:
                    continue

                matches_user = (
                    phones_match(odata.get("phone"), user_phone) 
                    or phones_match(odata.get("phone"), display_phone) 
                    or phones_match(odata.get("phone"), assigned_phone)
                    or (odata.get("meta_identifier") and (
                        phones_match(odata.get("meta_identifier"), user_phone)
                        or phones_match(odata.get("meta_identifier"), display_phone)
                    ))
                    or (is_meta and phone_info.get("extracted_phone") and phones_match(odata.get("extracted_phone"), phone_info["extracted_phone"]))
                )
                if not matches_user:
                    continue

                order_ts = odata.get("timestamp", 0)
                if not order_ts:
                    continue

                time_diff_ms = order_timestamp - order_ts

                # Comparar productos y total con el pedido activo
                old_sig = canonical_order_signature(
                    odata.get("detalle_completo", "") or odata.get("pedido", ""),
                    odata.get("total_a_cobrar", "")
                )
                new_sig = canonical_order_signature(detalle_completo, total_a_cobrar)
                texto_lower = (lista_productos or "").lower()
                is_generic_summary = "historial" in texto_lower or "productos por un total" in texto_lower
                items_match = is_generic_summary or (old_sig == new_sig)

                if items_match:
                    # Mismos productos (ej. aclarando pago, enviando comprobante o confirmando dirección)
                    if time_diff_ms < PAYMENT_UPDATE_WINDOW_MS:
                        is_modification = True
                        existing_order_id = oid
                        existing_order = odata
                        same_items_and_total = True
                        break
                else:
                    # Productos diferentes o adición de ítems ("pidió algo más")
                    if time_diff_ms < MODIFICATION_WINDOW_MS:
                        # Aún estamos a tiempo (< 20 min): modificar/adicionar al pedido activo in-place
                        is_modification = True
                        existing_order_id = oid
                        existing_order = odata
                        same_items_and_total = False
                        break
                    else:
                        # Pasaron más de 20 minutos: la cocina ya preparó o despachó la orden anterior.
                        # Debe tratarse como NUEVO pedido independiente.
                        print(
                            f"[tools/enviar_notificacion/{user_phone}] ⏱️ Pasaron {time_diff_ms / 60000:.1f} min desde el pedido {oid} (> 20 min). "
                            f"Como el cliente pidió productos adicionales/diferentes, se crea un NUEVO pedido independiente.",
                            flush=True
                        )
                        is_modification = False
                        existing_order_id = None
                        existing_order = None
                        break

        is_exact_duplicate = False
        if is_modification and existing_order_id and existing_order:
            texto_lower = (lista_productos or "").lower()
            is_generic_summary = "historial" in texto_lower or "productos por un total" in texto_lower

            # Preservar dirección anterior si la nueva es "N.A." o vacía
            old_addr = existing_order.get("direccion", "")
            new_addr_lower = (direccion_envio or "").lower().strip()
            if new_addr_lower in ["n/a", "n.a.", "n.a", "no aplica", "no especificada", "ya registrada", "n.a. (no aplica)"] or not new_addr_lower:
                if old_addr and old_addr.lower() not in ["n/a", "n.a.", "n.a", "no aplica", "no especificada", "ya registrada", "n.a. (no aplica)"]:
                    direccion_envio = old_addr

            # Comparar firma canónica de productos + total + dirección + método de pago
            old_addr_norm = "".join(c.lower() for c in str(old_addr) if c.isascii() and c.isalnum())
            new_addr_norm = "".join(c.lower() for c in str(direccion_envio or "") if c.isascii() and c.isalnum())
            old_mp_norm = "".join(c.lower() for c in str(existing_order.get("metodo_pago", "")) if c.isascii() and c.isalnum())
            new_mp_norm = "".join(c.lower() for c in str(metodo_pago or "") if c.isascii() and c.isalnum())

            same_address = (old_addr_norm == new_addr_norm) or not new_addr_norm
            same_payment = (old_mp_norm == new_mp_norm) or not new_mp_norm

            if same_items_and_total and same_address and same_payment:
                is_exact_duplicate = True

            # ACTUALIZAR in-place: preservar timestamp original, status e historial
            orders_data[existing_order_id]["summary"] = resumen_general
            
            if not is_generic_summary:
                orders_data[existing_order_id]["pedido"] = detalle_completo
                orders_data[existing_order_id]["detalle_completo"] = detalle_completo
            else:
                orders_data[existing_order_id]["detalle_completo"] = format_detalle_wa(
                    orders_data[existing_order_id].get("detalle_completo", "")
                )

            orders_data[existing_order_id]["total_a_cobrar"] = total_a_cobrar
            orders_data[existing_order_id]["direccion"] = direccion_envio
            orders_data[existing_order_id]["metodo_pago"] = metodo_pago
            orders_data[existing_order_id]["phone"] = assigned_phone
            if is_meta and phone_info["extracted_phone"]:
                orders_data[existing_order_id]["extracted_phone"] = phone_info["extracted_phone"]
            elif not is_meta:
                orders_data[existing_order_id]["extracted_phone"] = None
            if is_meta:
                orders_data[existing_order_id]["meta_identifier"] = user_phone
            if nombre_cliente and nombre_cliente.strip().lower() not in ["", "no especificado", "no aplica", "n/a", "na"]:
                orders_data[existing_order_id]["extracted_name"] = nombre_cliente.strip()
            order_id = existing_order_id
            
            # Si preservamos el detalle anterior, asegurarnos de usarlo para la notificación HTTP de WhatsApp
            if is_generic_summary:
                detalle_completo = orders_data[existing_order_id].get("detalle_completo", detalle_completo)
                
            print(f"[tools/enviar_notificacion/{user_phone}] Pedido actualizado in-place en dashboard (ID: {order_id}, same_items={same_items_and_total}, is_exact_duplicate={is_exact_duplicate}).", flush=True)
        else:
            # Pedido nuevo — usar prefijo ORDER_ original
            order_id = f"ORDER_{int(time.time())}_{display_phone[-4:]}"

            orders_data[order_id] = {
                "id": order_id,
                "phone": assigned_phone,
                "meta_identifier": user_phone if is_meta else None,
                "summary": resumen_general,
                "pedido": lista_productos,
                "detalle_completo": detalle_completo,
                "total_a_cobrar": total_a_cobrar,
                "direccion": direccion_envio,
                "metodo_pago": metodo_pago,
                "status": "Recientes",
                "timestamp": order_timestamp,
                "static_duration": 0,
                "history": [{"status": "Recientes", "timestamp": order_timestamp}],
                "origen": "tool_langgraph",
                "extracted_phone": phone_info["extracted_phone"],
                "extracted_name": nombre_cliente
            }
            print(f"[tools/enviar_notificacion/{user_phone}] Nuevo pedido guardado en dashboard (ID: {order_id}).", flush=True)

        with open(ORDERS_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(orders_data, f, ensure_ascii=False, indent=4)

        # 1.1 Despachar webhook a los sistemas POS ERP configurados
        pos_endpoints_cfg = config_dict.get("pos_endpoints", "[]")
        order_webhook_payload = {
            "orderIdWA": order_id,
            "clienteTelefono": display_phone,
            "clienteNombre": nombre_cliente if nombre_cliente and str(nombre_cliente).strip().lower() not in ["no especificado", "no aplica", "n/a", "na", ""] else "",
            "direccion": direccion_envio,
            "detalleTextoWA": detalle_completo or lista_productos,
            "resumen": resumen_general,
            "totalWA": total_a_cobrar,
            "metodoPagoWA": metodo_pago,
            "timestamp": order_timestamp,
            "isModification": is_modification,
            "userbot": config_dict.get("userbot_identifier") or userbot
        }
        dispatch_order_to_pos_webhooks(pos_endpoints_cfg, order_webhook_payload)

        # Registrar timestamp de notificación en orders_data para prevenir duplicados en ventana corta
        if order_id in orders_data:
            orders_data[order_id]["wa_notified_timestamp"] = order_timestamp
            with open(ORDERS_FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(orders_data, f, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"[tools/enviar_notificacion/{user_phone}] Error al guardar en dashboard: {e}", flush=True)

    # --- Omitir spam de WhatsApp si es un duplicado exacto ---
    if is_exact_duplicate:
        print(f"[tools/enviar_notificacion/{user_phone}] ⚠️ [NOTIF-TOOL] Omitiendo notificación por WhatsApp porque el detalle, total y método de pago no cambiaron recientemente (duplicado exacto).", flush=True)
        config_dict["notification_tool_executed"] = True
        return "Notificación omitida por duplicado exacto, pero pedido actualizado en dashboard."

    # 2. Enviar notificación HTTP a WhatsApp usando el userbot designado (opcional o principal)
    send_url = f"{clean_server_url}/chats/send?id={notification_userbot}"
    headers = {"Content-Type": "application/json", "x-access-token": token}
    
    link_message_text = phone_info["link_message_text"]
    
    header_title = "🛍️ *NUEVO PEDIDO* 🛍️"
    if is_modification:
        if not same_items_and_total:
            header_title = "🔄 *PEDIDO MODIFICADO / ADICIÓN* 🔄"
        else:
            header_title = "🔄 *PEDIDO ACTUALIZADO (PAGO/DATOS)* 🔄"

    detail_message_text = (
        f"{header_title}\n\n"
        f"👤 *Cliente:* {cliente_str}\n\n"
        f"📝 *Detalle del Pedido:*\n{detalle_completo}\n\n"
        f"💵 *Total a Cobrar:* {total_a_cobrar}\n"
        f"💳 *Método de Pago:* {metodo_pago}\n"
        f"📍 *Dirección de Entrega:* {direccion_envio}\n\n"
        f"🤖 *Resumen IA:* {resumen_general}"
    )
    combined_message_text = f"{detail_message_text}\n\n{link_message_text}"

    combined_payload = {
        "receiver": receiver,
        "isGroup": bool(is_group),
        "message": {
            "text": combined_message_text
        }
    }

    def _send_wa_notification_async():
        try:
            print(f"[tools/enviar_notificacion/{user_phone}] 🚀 [NOTIF-THREAD] Enviando a WhatsApp: url={send_url}, receiver={receiver}, isGroup={bool(is_group)}, sender_bot={notification_userbot}", flush=True)
            with httpx.Client(timeout=25.0, verify=False) as client:
                resp = client.post(send_url, headers=headers, json=combined_payload)
                if resp.status_code in [200, 201]:
                    print(f"[tools/enviar_notificacion/{user_phone}] ✅ [NOTIF-THREAD] ¡ÉXITO! Notificación enviada al grupo/línea -> HTTP {resp.status_code}: {resp.text[:120]}", flush=True)
                else:
                    print(f"[tools/enviar_notificacion/{user_phone}] ❌ [NOTIF-THREAD] ERROR respuesta WhatsApp ({send_url}) -> HTTP {resp.status_code}: {resp.text}", flush=True)
        except Exception as e:
            print(f"[tools/enviar_notificacion/{user_phone}] 💥 [NOTIF-THREAD] Excepción enviando notificación: {e}", flush=True)

    threading.Thread(target=_send_wa_notification_async, daemon=True).start()
    config_dict["notification_tool_executed"] = True
    print(f"[tools/enviar_notificacion/{user_phone}] 📌 [NOTIF-TOOL] Hilo de envío a WhatsApp iniciado (notification_tool_executed=True).", flush=True)
    return "Notificación encolada y pedido registrado en dashboard exitosamente."
