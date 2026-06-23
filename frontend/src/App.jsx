import React, { useState, useEffect, useCallback } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import {
  Bot, LogIn, LayoutDashboard, Settings, Plus, Trash2, Save,
  LogOut, AlertCircle, Loader, Search, Bell, Wrench, ChevronDown, ChevronUp,
  MessageSquare, Paperclip, Send, X, UploadCloud, FileText, Database, Edit
} from 'lucide-react';

// ─── API Helper ────────────────────────────────────────────────────────
async function apiFetch(path, options = {}) {
  const token = localStorage.getItem('token');
  const headers = { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) };
  const res = await fetch(path, { ...options, headers });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Error desconocido');
  }
  return res.json();
}

// ─── Toast ─────────────────────────────────────────────────────────────
function Toast({ msg, type, onClose }) {
  useEffect(() => { const t = setTimeout(onClose, 4000); return () => clearTimeout(t); }, [onClose]);
  const colors = { error: '#ef4444', success: '#10b981', info: '#3b82f6', warn: '#f59e0b' };
  return (
    <div style={{
      position: 'fixed', bottom: '24px', right: '24px', zIndex: 9999,
      background: colors[type] || colors.info, color: '#fff',
      padding: '14px 20px', borderRadius: '10px', maxWidth: '400px',
      boxShadow: '0 8px 30px rgba(0,0,0,0.35)', display: 'flex', alignItems: 'center', gap: '10px',
      animation: 'slideIn 0.3s ease',
    }}>
      <AlertCircle size={18} /><span style={{ fontSize: '14px' }}>{msg}</span>
    </div>
  );
}

// ─── Section Component ─────────────────────────────────────────────────
function Section({ title, icon, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={{ border: '1px solid var(--panel-border)', borderRadius: '10px', overflow: 'hidden', marginBottom: '16px' }}>
      <button onClick={() => setOpen(o => !o)} style={{
        width: '100%', background: 'rgba(255,255,255,0.05)', border: 'none', padding: '12px 16px',
        display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', color: 'var(--text-primary)', fontWeight: 600, fontSize: '14px'
      }}>
        {icon}{title}
        <span style={{ marginLeft: 'auto' }}>{open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}</span>
      </button>
      {open && <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '14px' }}>{children}</div>}
    </div>
  );
}

// ─── Field Helper ──────────────────────────────────────────────────────
function Field({ label, hint, children }) {
  return (
    <div>
      <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '5px', fontWeight: 600, letterSpacing: '0.03em' }}>
        {label} {hint && <span style={{ color: '#64748b', fontWeight: 400, fontStyle: 'italic' }}>{hint}</span>}
      </label>
      {children}
    </div>
  );
}

// ─── DEFAULT BOT STATE ─────────────────────────────────────────────────
const DEFAULT_BOT = {
  userbot_identifier: '',
  apikey: '',
  system_prompt: '',
  ai_model: 'gemini-2.5-flash',
  thinking_budget: 0,
  thinking_level: "HIGH",
  use_google_search: false,
  use_google_maps: false,
  is_active: true,
  pais: 'colombia',
  idioma: 'es',
  delay_seconds: 0,
  pause_timeout_minutes: 30,
  activarnotificacion: false,
  estado_notificacion: '',
  lineaogruponotificacion: '',
  es_grupo_notificacion: false,
  activaruserbotopcional: false,
  userbotopcional: '',
  tools_config: '{"buscar_catalogo": true, "enviar_notificacion": true, "obtener_hora": true}',
};

function parseTools(jsonStr) {
  try { return JSON.parse(jsonStr); }
  catch { return { buscar_catalogo: true, enviar_notificacion: true, obtener_hora: true }; }
}

// ─── Bot Modal ─────────────────────────────────────────────────────────
function BotModal({ bot, onClose, onSave, toast }) {
  const [form, setForm] = useState(bot ? { ...bot } : { ...DEFAULT_BOT });
  const [tools, setTools] = useState(parseTools(bot?.tools_config || DEFAULT_BOT.tools_config));
  const [saving, setSaving] = useState(false);
  const [lookingUp, setLookingUp] = useState(false);
  const [lookupDone, setLookupDone] = useState(!!bot);

  const set = (key, val) => setForm(f => ({ ...f, [key]: val }));

  // Auto-lookup cuando se escribe el ID (con debounce)
  const handleLookup = async () => {
    const id = form.userbot_identifier.trim();
    if (!id) { toast('Ingresa un ID de userbot primero', 'warn'); return; }
    setLookingUp(true);
    try {
      const found = await apiFetch(`/api/bots/lookup/${encodeURIComponent(id)}`);
      setForm({ ...found });
      setTools(parseTools(found.tools_config));
      setLookupDone(true);
      toast('Configuracion encontrada y cargada', 'success');
    } catch (e) {
      setLookupDone(true);
      toast('No hay configuracion guardada para ese ID. Puedes crear una nueva.', 'info');
    } finally {
      setLookingUp(false);
    }
  };

  const handleSave = async () => {
    if (!form.userbot_identifier.trim()) { toast('El ID del userbot es obligatorio', 'error'); return; }
    setSaving(true);
    const payload = { ...form, tools_config: JSON.stringify(tools) };
    try {
      if (bot || form.id) {
        const id = form.id || bot.id;
        await apiFetch(`/api/bots/${id}`, { method: 'PUT', body: JSON.stringify(payload) });
      } else {
        await apiFetch('/api/bots/', { method: 'POST', body: JSON.stringify(payload) });
      }
      toast(bot ? 'Bot actualizado' : 'Bot creado', 'success');
      onSave();
      onClose();
    } catch (e) {
      toast(e.message, 'error');
    } finally {
      setSaving(false);
    }
  };

  const AI_MODELS = [
    'gemini-3.5-flash', 'gemini-3.1-pro-preview', 'gemini-3-flash-preview',
    'gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash',
    'gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-1.5-pro'
  ];

  const TOOLS_INFO = {
    buscar_catalogo: { label: 'Buscar en catálogo (RAG)', desc: 'Busca en PDFs y TXTs subidos al bot. Responde preguntas sobre productos, precios y servicios.' },
    enviar_notificacion: { label: 'Enviar notificación WhatsApp', desc: 'Envía un mensaje push al número de notificación cuando se alcanza el estado configurado.' },
    obtener_hora: { label: 'Obtener hora actual', desc: 'Permite al bot responder qué hora es según el país configurado.' },
  };

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(6px)', display: 'flex', alignItems: 'flex-start', justifyContent: 'center', zIndex: 1000, padding: '24px', overflowY: 'auto' }}>
      <div className="glass-panel" style={{ width: '100%', maxWidth: '680px', marginBottom: '24px' }}>
        <h2 className="text-gradient" style={{ marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Bot size={24} />{bot ? 'Configurar Bot' : 'Nuevo Bot'}
        </h2>

        {/* ── ID + Lookup ─────────────────────────────────── */}
        <div style={{ marginBottom: '20px' }}>
          <Field label="ID DEL USERBOT" hint="(el valor que envías en req.userbot)">
            <div style={{ display: 'flex', gap: '8px' }}>
              <input
                id="bot-identifier"
                type="text"
                placeholder="ej: mi_bot_ventas_01"
                value={form.userbot_identifier}
                onChange={e => { set('userbot_identifier', e.target.value); setLookupDone(false); }}
                disabled={!!bot}
                style={bot ? { opacity: 0.5, cursor: 'not-allowed', flex: 1 } : { flex: 1 }}
              />
              {!bot && (
                <button
                  id="bot-lookup-btn"
                  onClick={handleLookup}
                  disabled={lookingUp}
                  title="Verificar si ya existe configuración para este ID"
                  style={{ padding: '12px 16px', background: 'rgba(59,130,246,0.2)', border: '1px solid #3b82f6', color: '#60a5fa', whiteSpace: 'nowrap', minWidth: '130px' }}
                >
                  {lookingUp ? <Loader size={16} style={{ animation: 'spin 1s linear infinite' }} /> : <><Search size={14} /> Verificar ID</>}
                </button>
              )}
            </div>
            {lookupDone && !bot && (
              <p style={{ fontSize: '11px', color: '#64748b', marginTop: '4px', marginBottom: 0 }}>
                Si el ID ya existe en la BD, los campos se autocompletarán.
              </p>
            )}
          </Field>
        </div>

        {/* ── Sección: IA ─────────────────────────────────── */}
        <Section title="Inteligencia Artificial" icon={<Bot size={16} style={{ color: '#60a5fa' }} />}>
          <Field label="GOOGLE API KEY" hint="(se autoguarda desde la primera petición de WhatsApp)">
            <input
              id="bot-apikey"
              type="password"
              placeholder="AIza..."
              value={form.apikey || ''}
              onChange={e => set('apikey', e.target.value)}
              autoComplete="off"
            />
            <p style={{ fontSize: '11px', color: '#64748b', margin: '4px 0 0' }}>
              Requerida para el RAG (subida de PDFs). Se rellena automáticamente cuando llega la primera petición de WhatsApp con ese userbot.
            </p>
          </Field>
          <Field label="MODELO DE GEMINI">
            <select id="bot-model" value={form.ai_model} onChange={e => set('ai_model', e.target.value)}>
              {AI_MODELS.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
          </Field>
          {!form.ai_model?.includes('gemini-3') && (
            <Field label="MODO DE RAZONAMIENTO (thinking_budget)" hint="(0=respuesta directa, >0=presupuesto tokens)">
              <input id="bot-thinking-budget" type="number" value={form.thinking_budget} onChange={e => set('thinking_budget', parseInt(e.target.value))} />
            </Field>
          )}
          {form.ai_model?.includes('gemini-3') && (
            <Field label="NIVEL DE RAZONAMIENTO (thinking_level)" hint="(Para modelos 3.0+)">
              <select id="bot-thinking-level" value={form.thinking_level || 'HIGH'} onChange={e => set('thinking_level', e.target.value)}>
                <option value="HIGH">ALTO (High)</option>
                <option value="MEDIUM">MEDIO (Medium)</option>
                <option value="LOW">BAJO (Low)</option>
                <option value="MINIMAL">MÍNIMO (Minimal)</option>
              </select>
            </Field>
          )}
          <Field label="PROMPT DEL SISTEMA">
            <textarea
              id="bot-prompt"
              placeholder="Eres un asistente de ventas de [Empresa]. Responde siempre en español..."
              value={form.system_prompt}
              onChange={e => set('system_prompt', e.target.value)}
              rows={7}
              style={{ resize: 'vertical', fontFamily: 'monospace', fontSize: '13px', lineHeight: '1.6' }}
            />
          </Field>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <input id="bot-active" type="checkbox" checked={form.is_active} onChange={e => set('is_active', e.target.checked)} style={{ width: '16px', height: '16px', cursor: 'pointer', accentColor: '#3b82f6' }} />
            <label htmlFor="bot-active" style={{ cursor: 'pointer', fontSize: '14px' }}>Bot activo (procesa mensajes vía LangGraph)</label>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '10px' }}>
            <input id="bot-google-search" type="checkbox" checked={form.use_google_search || false} onChange={e => set('use_google_search', e.target.checked)} style={{ width: '16px', height: '16px', cursor: 'pointer', accentColor: '#3b82f6' }} />
            <label htmlFor="bot-google-search" style={{ cursor: 'pointer', fontSize: '14px' }}>Habilitar Búsqueda en Google (Gemini Tools)</label>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '10px' }}>
            <input id="bot-google-maps" type="checkbox" checked={form.use_google_maps || false} onChange={e => set('use_google_maps', e.target.checked)} style={{ width: '16px', height: '16px', cursor: 'pointer', accentColor: '#3b82f6' }} />
            <label htmlFor="bot-google-maps" style={{ cursor: 'pointer', fontSize: '14px' }}>Habilitar Google Maps (Gemini Tools)</label>
          </div>
        </Section>

        {/* ── Sección: Regional ───────────────────────────── */}
        <Section title="Región e Idioma" icon={<span style={{ fontSize: '16px' }}>🌎</span>} defaultOpen={false}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <Field label="PAÍS" hint="(para zona horaria)">
              <input id="bot-pais" type="text" placeholder="colombia" value={form.pais} onChange={e => set('pais', e.target.value)} />
            </Field>
            <Field label="IDIOMA">
              <input id="bot-idioma" type="text" placeholder="es" value={form.idioma} onChange={e => set('idioma', e.target.value)} />
            </Field>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <Field label="DELAY ENTRE MENSAJES (seg)">
              <input id="bot-delay" type="number" min="0" step="0.5" value={form.delay_seconds} onChange={e => set('delay_seconds', parseFloat(e.target.value))} />
            </Field>
            <Field label="TIMEOUT DE PAUSA (min)">
              <input id="bot-pause" type="number" min="1" value={form.pause_timeout_minutes} onChange={e => set('pause_timeout_minutes', parseInt(e.target.value))} />
            </Field>
          </div>
        </Section>

        {/* ── Sección: Notificaciones ─────────────────────── */}
        <Section title="Notificaciones" icon={<Bell size={16} style={{ color: '#f59e0b' }} />} defaultOpen={false}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <input id="bot-notif" type="checkbox" checked={form.activarnotificacion} onChange={e => set('activarnotificacion', e.target.checked)} style={{ width: '16px', height: '16px', cursor: 'pointer', accentColor: '#f59e0b' }} />
            <label htmlFor="bot-notif" style={{ cursor: 'pointer', fontSize: '14px' }}>Activar notificaciones push</label>
          </div>

          {form.activarnotificacion && (
            <>
              <Field label="ESTADO QUE DISPARA LA NOTIFICACIÓN" hint="(ej: Pedido, Cierre)">
                <input id="bot-estado" type="text" placeholder="Pedido" value={form.estado_notificacion} onChange={e => set('estado_notificacion', e.target.value)} />
              </Field>
              <Field label="NÚMERO / GRUPO DESTINO">
                <input id="bot-linea-notif" type="text" placeholder="573001234567@c.us o ID del grupo" value={form.lineaogruponotificacion} onChange={e => set('lineaogruponotificacion', e.target.value)} />
              </Field>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <input id="bot-es-grupo" type="checkbox" checked={form.es_grupo_notificacion} onChange={e => set('es_grupo_notificacion', e.target.checked)} style={{ width: '16px', height: '16px', cursor: 'pointer', accentColor: '#3b82f6' }} />
                <label htmlFor="bot-es-grupo" style={{ cursor: 'pointer', fontSize: '14px' }}>El destino es un grupo</label>
              </div>
              <hr style={{ border: 'none', borderTop: '1px solid var(--panel-border)' }} />
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <input id="bot-userbot-opc" type="checkbox" checked={form.activaruserbotopcional} onChange={e => set('activaruserbotopcional', e.target.checked)} style={{ width: '16px', height: '16px', cursor: 'pointer', accentColor: '#3b82f6' }} />
                <label htmlFor="bot-userbot-opc" style={{ cursor: 'pointer', fontSize: '14px' }}>Usar userbot alternativo para notificaciones</label>
              </div>
              {form.activaruserbotopcional && (
                <Field label="USERBOT ALTERNATIVO">
                  <input id="bot-userbot-alt" type="text" placeholder="ID del userbot alternativo" value={form.userbotopcional} onChange={e => set('userbotopcional', e.target.value)} />
                </Field>
              )}
            </>
          )}
        </Section>

        {/* ── Sección: Herramientas (Tools) ───────────────── */}
        <Section title="Herramientas del Agente (Tools)" icon={<Wrench size={16} style={{ color: '#a78bfa' }} />} defaultOpen={true}>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)', margin: 0 }}>
            Selecciona qué herramientas puede usar el agente autónomamente durante la conversación.
          </p>
          {Object.entries(TOOLS_INFO).map(([key, info]) => (
            <div key={key} style={{ display: 'flex', gap: '14px', alignItems: 'flex-start', background: 'rgba(0,0,0,0.2)', padding: '12px', borderRadius: '8px' }}>
              <input
                id={`tool-${key}`}
                type="checkbox"
                checked={tools[key] ?? true}
                onChange={e => setTools(t => ({ ...t, [key]: e.target.checked }))}
                style={{ width: '18px', height: '18px', cursor: 'pointer', accentColor: '#a78bfa', marginTop: '2px', flexShrink: 0 }}
              />
              <div>
                <label htmlFor={`tool-${key}`} style={{ cursor: 'pointer', fontWeight: 600, fontSize: '14px', display: 'block' }}>{info.label}</label>
                <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{info.desc}</span>
              </div>
            </div>
          ))}
        </Section>

        {/* ── Acciones ────────────────────────────────────── */}
        <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end', marginTop: '8px' }}>
          <button id="bot-cancel-btn" onClick={onClose} style={{ background: 'transparent', border: '1px solid var(--panel-border)', color: 'var(--text-secondary)' }}>
            Cancelar
          </button>
          <button id="bot-save-btn" onClick={handleSave} disabled={saving} style={{ opacity: saving ? 0.7 : 1 }}>
            {saving ? <><Loader size={16} style={{ animation: 'spin 1s linear infinite' }} /> Guardando...</> : <><Save size={16} /> Guardar</>}
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Auth ──────────────────────────────────────────────────────────────
function useAuth() {
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [user, setUser] = useState(null);

  const login = async (email, password) => {
    const form = new URLSearchParams();
    form.append('username', email);
    form.append('password', password);
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: form,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Credenciales incorrectas' }));
      throw new Error(err.detail || 'Error al iniciar sesión');
    }
    const data = await res.json();
    localStorage.setItem('token', data.access_token);
    setToken(data.access_token);
  };

  const logout = () => { localStorage.removeItem('token'); setToken(null); setUser(null); };

  useEffect(() => {
    if (token) apiFetch('/api/auth/me').then(setUser).catch(logout);
  }, [token]);

  const register = async (email, password) => {
    const res = await fetch('/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Error al registrar' }));
      throw new Error(err.detail || 'Error al registrar');
    }
    await login(email, password);
  };

  return { token, user, login, register, logout };
}

// ─── Login ─────────────────────────────────────────────────────────────
function LoginPage({ auth }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try { await auth.login(email, password); navigate('/panel'); }
    catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };

  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh', padding: '24px' }}>
      <div className="glass-panel" style={{ width: '100%', maxWidth: '420px' }}>
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <div style={{ fontSize: '48px', marginBottom: '12px' }}>🤖</div>
          <h1 className="text-gradient" style={{ fontSize: '28px', margin: 0 }}>AutoSystem Panel</h1>
          <p style={{ color: 'var(--text-secondary)', margin: '8px 0 0', fontSize: '14px' }}>Gestiona tus chatbots de WhatsApp</p>
        </div>
        {error && (
          <div style={{ background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.4)', borderRadius: '8px', padding: '12px 16px', marginBottom: '20px', color: '#fca5a5', fontSize: '14px', display: 'flex', gap: '8px', alignItems: 'center' }}>
            <AlertCircle size={16} />{error}
          </div>
        )}
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div><label style={{ display: 'block', fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '6px' }}>Email</label><input id="login-email" type="email" placeholder="admin@tuempresa.com" value={email} onChange={e => setEmail(e.target.value)} required /></div>
          <div><label style={{ display: 'block', fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '6px' }}>Contraseña</label><input id="login-password" type="password" placeholder="••••••••" value={password} onChange={e => setPassword(e.target.value)} required /></div>
          <button id="login-btn" type="submit" disabled={loading} style={{ marginTop: '8px', opacity: loading ? 0.7 : 1 }}>
            {loading ? <><Loader size={16} style={{ animation: 'spin 1s linear infinite' }} /> Iniciando...</> : <><LogIn size={16} /> Iniciar Sesión</>}
          </button>
        </form>
        <p style={{ textAlign: 'center', marginTop: '20px', fontSize: '13px', color: 'var(--text-secondary)' }}>
          ¿No tienes cuenta? <button onClick={() => navigate('/register')} style={{ background: 'none', border: 'none', color: '#3b82f6', cursor: 'pointer', padding: 0, fontWeight: 600 }}>Regístrate aquí</button>
        </p>
      </div>
    </div>
  );
}

// ─── Register ──────────────────────────────────────────────────────────
function RegisterPage({ auth }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try { await auth.register(email, password); navigate('/panel'); }
    catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };

  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh', padding: '24px' }}>
      <div className="glass-panel" style={{ width: '100%', maxWidth: '420px' }}>
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <div style={{ fontSize: '48px', marginBottom: '12px' }}>🤖</div>
          <h1 className="text-gradient" style={{ fontSize: '28px', margin: 0 }}>Crear Cuenta</h1>
          <p style={{ color: 'var(--text-secondary)', margin: '8px 0 0', fontSize: '14px' }}>Regístrate para gestionar tus bots</p>
        </div>
        {error && (
          <div style={{ background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.4)', borderRadius: '8px', padding: '12px 16px', marginBottom: '20px', color: '#fca5a5', fontSize: '14px', display: 'flex', gap: '8px', alignItems: 'center' }}>
            <AlertCircle size={16} />{error}
          </div>
        )}
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div><label style={{ display: 'block', fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '6px' }}>Email</label><input id="reg-email" type="email" placeholder="tu@email.com" value={email} onChange={e => setEmail(e.target.value)} required /></div>
          <div><label style={{ display: 'block', fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '6px' }}>Contraseña</label><input id="reg-password" type="password" placeholder="••••••••" value={password} onChange={e => setPassword(e.target.value)} required minLength={6} /></div>
          <button id="reg-btn" type="submit" disabled={loading} style={{ marginTop: '8px', opacity: loading ? 0.7 : 1 }}>
            {loading ? <><Loader size={16} style={{ animation: 'spin 1s linear infinite' }} /> Registrando...</> : <><Plus size={16} /> Crear Cuenta</>}
          </button>
        </form>
        <p style={{ textAlign: 'center', marginTop: '20px', fontSize: '13px', color: 'var(--text-secondary)' }}>
          ¿Ya tienes cuenta? <button onClick={() => navigate('/login')} style={{ background: 'none', border: 'none', color: '#3b82f6', cursor: 'pointer', padding: 0, fontWeight: 600 }}>Inicia sesión</button>
        </p>
      </div>
    </div>
  );
}

// ─── Chat Simulator ────────────────────────────────────────────────────
function ChatSimulatorModal({ bot, onClose, toast }) {
  const [history, setHistory] = useState([]);
  const [msg, setMsg] = useState('');
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = React.useRef(null);
  const messagesEndRef = React.useRef(null);

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [history]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!msg.trim() && !file) return;

    const newHistory = [...history, { role: 'cliente', mensaje: msg, file: file?.name }];
    setHistory(newHistory);
    setMsg('');
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('mensaje', msg);
      formData.append('historial', JSON.stringify(history));
      if (file) formData.append('file', file);

      const token = localStorage.getItem('token');
      const res = await fetch(`/api/bots/${bot.id}/simulate`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData
      });

      if (!res.ok) {
        const errorText = await res.text();
        let parsedErr;
        try { parsedErr = JSON.parse(errorText).detail; } catch { parsedErr = errorText; }
        throw new Error(parsedErr);
      }
      const data = await res.json();
      
      const aiResponse = data.response;
      setHistory([...newHistory, { role: 'asistente', mensaje: aiResponse.respuesta_cliente, raw: aiResponse }]);
    } catch (err) {
      toast('Error en simulador: ' + err.message, 'error');
    } finally {
      setLoading(false);
      setFile(null);
    }
  };

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(6px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000, padding: '24px' }}>
      <div className="glass-panel" style={{ width: '100%', maxWidth: '500px', height: '80vh', display: 'flex', flexDirection: 'column', padding: 0, overflow: 'hidden' }}>
        
        {/* Header */}
        <div style={{ padding: '16px 20px', borderBottom: '1px solid rgba(255,255,255,0.1)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(0,0,0,0.2)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ width: '32px', height: '32px', borderRadius: '50%', background: '#3b82f6', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Bot size={18} color="white" />
            </div>
            <div>
              <h3 style={{ margin: 0, fontSize: '15px' }}>{bot.userbot_identifier}</h3>
              <p style={{ margin: 0, fontSize: '11px', color: '#94a3b8' }}>Simulador LangGraph</p>
            </div>
          </div>
          <button onClick={onClose} style={{ background: 'transparent', border: 'none', color: '#94a3b8', cursor: 'pointer' }}><X size={20} /></button>
        </div>

        {/* Messages */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {history.length === 0 && (
            <div style={{ textAlign: 'center', color: '#64748b', marginTop: '40px', fontSize: '13px' }}>
              Escribe un mensaje para empezar la simulación.
            </div>
          )}
          {history.map((h, i) => (
            <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: h.role === 'cliente' ? 'flex-end' : 'flex-start' }}>
              <div style={{ 
                maxWidth: '85%', 
                padding: '10px 14px', 
                borderRadius: '12px',
                background: h.role === 'cliente' ? '#3b82f6' : 'rgba(255,255,255,0.1)',
                color: 'white',
                fontSize: '14px',
                lineHeight: '1.4',
                whiteSpace: 'pre-wrap'
              }}>
                {h.file && <div style={{ fontSize: '11px', opacity: 0.8, marginBottom: '4px' }}>📎 {h.file}</div>}
                {h.mensaje || '(Vacio)'}
              </div>
              {h.raw && <div style={{ fontSize: '10px', color: '#64748b', marginTop: '4px' }}>{h.raw.estado_conversacion} | {h.raw.accion_interna}</div>}
            </div>
          ))}
          {loading && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#94a3b8', fontSize: '12px' }}>
              <Loader size={12} style={{ animation: 'spin 1s linear infinite' }} /> Escribiendo...
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div style={{ padding: '16px', borderTop: '1px solid rgba(255,255,255,0.1)', background: 'rgba(0,0,0,0.2)' }}>
          {file && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px', fontSize: '12px', color: '#60a5fa', background: 'rgba(59,130,246,0.1)', padding: '6px 12px', borderRadius: '6px' }}>
              <Paperclip size={14} /> {file.name} 
              <button onClick={() => setFile(null)} style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', marginLeft: 'auto' }}><X size={14} /></button>
            </div>
          )}
          <form onSubmit={handleSend} style={{ display: 'flex', gap: '8px' }}>
            <input type="file" ref={fileInputRef} style={{ display: 'none' }} onChange={e => setFile(e.target.files[0])} />
            <button type="button" onClick={() => fileInputRef.current.click()} style={{ padding: '10px', background: 'rgba(255,255,255,0.1)', border: 'none', color: 'white', borderRadius: '8px', cursor: 'pointer' }} title="Adjuntar archivo">
              <Paperclip size={18} />
            </button>
            <input 
              type="text" 
              value={msg} 
              onChange={e => setMsg(e.target.value)} 
              placeholder="Escribe un mensaje..." 
              style={{ flex: 1, margin: 0 }}
              disabled={loading}
            />
            <button type="submit" disabled={loading || (!msg.trim() && !file)} style={{ padding: '10px 16px', margin: 0, opacity: (loading || (!msg.trim() && !file)) ? 0.5 : 1 }}>
              <Send size={18} />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

// ─── Knowledge Upload Modal ───────────────────────────────────────────
function KnowledgeUploadModal({ bot, onClose, toast }) {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const fileInputRef = React.useRef(null);

  const handleUpload = async (e) => {
    e.preventDefault();
    if (files.length === 0) {
      if (fileInputRef.current) fileInputRef.current.click();
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      files.forEach(f => formData.append('files', f));

      const token = localStorage.getItem('token');
      const res = await fetch(`/api/bots/${bot.id}/upload-knowledge`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData
      });

      if (!res.ok) {
        const err = await res.text();
        throw new Error(JSON.parse(err).detail || 'Error subiendo archivo');
      }
      
      toast('Archivo procesado y guardado en la base de datos RAG exitosamente', 'success');
      onClose();
    } catch (e) {
      toast(e.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(6px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000, padding: '24px' }}>
      <div className="glass-panel" style={{ width: '100%', maxWidth: '400px' }}>
        <h2 className="text-gradient" style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <UploadCloud size={24} /> Subir Conocimiento (RAG)
        </h2>
        <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '20px' }}>
          Sube un archivo <strong>.pdf</strong>, <strong>.txt</strong> o una <strong>imagen</strong> (catálogo/menú) para alimentar la base de datos de tu bot.
          Las imágenes serán leídas automáticamente por Gemini para extraer productos y precios.
        </p>
        <form onSubmit={handleUpload}>
          <div style={{ border: '2px dashed rgba(255,255,255,0.1)', borderRadius: '12px', padding: '30px 20px', textAlign: 'center', marginBottom: '20px', background: 'rgba(0,0,0,0.2)', position: 'relative' }}>
            <input type="file" multiple accept=".pdf,.txt,.png,.jpg,.jpeg,.webp" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', opacity: 0, cursor: 'pointer' }} ref={fileInputRef} onChange={e => setFiles(Array.from(e.target.files))} />
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px', pointerEvents: 'none' }}>
              <div style={{ width: '48px', height: '48px', borderRadius: '50%', background: 'rgba(59,130,246,0.1)', color: '#60a5fa', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <FileText size={24} />
              </div>
              <span style={{ color: files.length > 0 ? 'white' : 'var(--text-secondary)', fontSize: '14px', fontWeight: files.length > 0 ? 600 : 400 }}>
                {files.length > 0 ? files.map(f => f.name).join(', ') : 'Haz clic aquí para seleccionar archivos (PDF/TXT/Imágenes)'}
              </span>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
            <button type="button" onClick={onClose} style={{ background: 'transparent', border: '1px solid rgba(255,255,255,0.1)', color: 'white' }}>Cancelar</button>
            <button type="submit" disabled={loading} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              {loading ? <Loader size={16} style={{ animation: 'spin 1s linear infinite' }} /> : <UploadCloud size={16} />}
              {loading ? 'Procesando RAG...' : (files.length > 0 ? 'Guardar y Procesar' : 'Seleccionar Archivo')}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ─── Knowledge Viewer Modal ───────────────────────────────────────────
function KnowledgeViewerModal({ bot, onClose, toast }) {
  const [knowledge, setKnowledge] = useState([]);
  const [loading, setLoading] = useState(true);
  const [editingId, setEditingId] = useState(null);
  const [editContent, setEditContent] = useState('');
  const [savingId, setSavingId] = useState(null);
  const [deletingId, setDeletingId] = useState(null);

  const loadKnowledge = useCallback(async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem('token');
      const res = await fetch(`/api/bots/${bot.id}/knowledge`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (!res.ok) throw new Error('Error al cargar la base de datos');
      const data = await res.json();
      setKnowledge(data.knowledge || []);
    } catch (e) {
      toast(e.message, 'error');
    } finally {
      setLoading(false);
    }
  }, [bot.id, toast]);

  React.useEffect(() => {
    loadKnowledge();
  }, [loadKnowledge]);

  const handleEdit = (item) => {
    setEditingId(item.id);
    setEditContent(item.content);
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditContent('');
  };

  const handleSave = async (id) => {
    try {
      setSavingId(id);
      const token = localStorage.getItem('token');
      const res = await fetch(`/api/bots/${bot.id}/knowledge/${id}`, {
        method: 'PUT',
        headers: { 
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}` 
        },
        body: JSON.stringify({ content: editContent })
      });
      if (!res.ok) throw new Error('Error al guardar el fragmento');
      toast('Fragmento guardado correctamente', 'success');
      setEditingId(null);
      loadKnowledge();
    } catch (e) {
      toast(e.message, 'error');
    } finally {
      setSavingId(null);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('¿Seguro que deseas eliminar este fragmento?')) return;
    try {
      setDeletingId(id);
      const token = localStorage.getItem('token');
      const res = await fetch(`/api/bots/${bot.id}/knowledge/${id}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` }
      });
      if (!res.ok) throw new Error('Error al eliminar el fragmento');
      toast('Fragmento eliminado', 'success');
      loadKnowledge();
    } catch (e) {
      toast(e.message, 'error');
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(6px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000, padding: '24px' }}>
      <div className="glass-panel" style={{ width: '100%', maxWidth: '700px', maxHeight: '85vh', display: 'flex', flexDirection: 'column' }}>
        <h2 className="text-gradient" style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Database size={24} /> Base de Datos (RAG)
        </h2>
        
        <div style={{ flex: 1, overflowY: 'auto', marginBottom: '20px', paddingRight: '10px' }}>
          {loading ? (
            <div style={{ display: 'flex', justifyContent: 'center', padding: '40px' }}><Loader size={24} style={{ animation: 'spin 1s linear infinite', color: '#60a5fa' }} /></div>
          ) : knowledge.length === 0 ? (
            <p style={{ textAlign: 'center', color: 'var(--text-secondary)', padding: '40px 0' }}>No hay conocimiento guardado para este bot aún.</p>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {knowledge.map((item, i) => (
                <div key={item.id} style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '8px', padding: '16px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px', fontSize: '12px', color: '#60a5fa', alignItems: 'center' }}>
                    <div>
                      <span style={{ fontWeight: 'bold' }}>Fragmento #{i + 1}</span>
                      <span style={{ color: 'var(--text-secondary)', marginLeft: '10px' }}>Fuente: {item.source}</span>
                    </div>
                    <div style={{ display: 'flex', gap: '8px' }}>
                      {editingId === item.id ? (
                        <>
                          <button 
                            onClick={() => handleSave(item.id)} 
                            disabled={savingId === item.id}
                            style={{ padding: '4px 8px', fontSize: '12px', background: 'rgba(16, 185, 129, 0.2)', color: '#34d399', border: '1px solid rgba(16, 185, 129, 0.3)', borderRadius: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}
                          >
                            {savingId === item.id ? <Loader size={12} className="spin" /> : <Save size={12} />} Guardar
                          </button>
                          <button 
                            onClick={handleCancelEdit}
                            style={{ padding: '4px 8px', fontSize: '12px', background: 'rgba(255, 255, 255, 0.1)', color: '#fff', border: 'none', borderRadius: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}
                          >
                            <X size={12} /> Cancelar
                          </button>
                        </>
                      ) : (
                        <>
                          <button 
                            onClick={() => handleEdit(item)} 
                            style={{ padding: '4px 8px', fontSize: '12px', background: 'rgba(59, 130, 246, 0.1)', color: '#60a5fa', border: '1px solid rgba(59, 130, 246, 0.2)', borderRadius: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}
                          >
                            <Edit size={12} /> Editar
                          </button>
                          <button 
                            onClick={() => handleDelete(item.id)} 
                            disabled={deletingId === item.id}
                            style={{ padding: '4px 8px', fontSize: '12px', background: 'rgba(239, 68, 68, 0.1)', color: '#f87171', border: '1px solid rgba(239, 68, 68, 0.2)', borderRadius: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}
                          >
                            {deletingId === item.id ? <Loader size={12} className="spin" /> : <Trash2 size={12} />} Eliminar
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                  
                  {editingId === item.id ? (
                    <textarea
                      value={editContent}
                      onChange={(e) => setEditContent(e.target.value)}
                      style={{ width: '100%', minHeight: '150px', background: 'rgba(0,0,0,0.5)', color: 'white', border: '1px solid #3b82f6', borderRadius: '4px', padding: '10px', fontSize: '13px', lineHeight: '1.5', resize: 'vertical' }}
                    />
                  ) : (
                    <div style={{ fontSize: '13px', color: 'white', whiteSpace: 'pre-wrap', lineHeight: '1.5' }}>
                      {item.content}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
        
        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <button onClick={onClose} style={{ background: 'rgba(255,255,255,0.1)', color: 'white', padding: '8px 16px', borderRadius: '6px' }}>Cerrar Visor</button>
        </div>
      </div>
    </div>
  );
}

// ─── Dashboard ─────────────────────────────────────────────────────────
function DashboardPage({ auth }) {
  const [bots, setBots] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [editingBot, setEditingBot] = useState(null);
  const [simulatingBot, setSimulatingBot] = useState(null);
  const [uploadingBot, setUploadingBot] = useState(null);
  const [viewingKnowledgeBot, setViewingKnowledgeBot] = useState(null);
  const [toastMsg, setToastMsg] = useState(null);
  const navigate = useNavigate();

  const showToast = useCallback((msg, type = 'info') => setToastMsg({ msg, type }), []);

  const loadBots = useCallback(async () => {
    setLoading(true);
    try { setBots(await apiFetch('/api/bots/')); }
    catch (e) { showToast(e.message, 'error'); }
    finally { setLoading(false); }
  }, [showToast]);

  useEffect(() => { loadBots(); }, [loadBots]);

  const handleDelete = async (bot) => {
    if (!confirm(`¿Eliminar "${bot.userbot_identifier}"? No se puede deshacer.`)) return;
    try { await apiFetch(`/api/bots/${bot.id}`, { method: 'DELETE' }); showToast('Bot eliminado', 'success'); loadBots(); }
    catch (e) { showToast(e.message, 'error'); }
  };

  return (
    <div className="container">
      <div className="flex-between" style={{ marginBottom: '8px' }}>
        <h1 className="text-gradient" style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '12px' }}>
          <LayoutDashboard size={30} /> Mis Chatbots
        </h1>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          {auth.user && <span style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>{auth.user.email} <span style={{ color: '#3b82f6', fontWeight: 600 }}>({auth.user.role})</span></span>}
          <button id="new-bot-btn" onClick={() => { setEditingBot(null); setShowModal(true); }}><Plus size={16} /> Nuevo Bot</button>
          <button id="logout-btn" onClick={() => { auth.logout(); navigate('/login'); }} className="danger" style={{ padding: '12px 14px' }}><LogOut size={16} /></button>
        </div>
      </div>
      <p style={{ color: 'var(--text-secondary)', fontSize: '13px', marginBottom: '28px' }}>
        Un bot se activa automáticamente cuando llega una petición con el mismo <code style={{ background: 'rgba(255,255,255,0.08)', padding: '1px 6px', borderRadius: '4px' }}>userbot_identifier</code>.
      </p>

      {loading ? (
        <div style={{ textAlign: 'center', padding: '80px', color: 'var(--text-secondary)' }}>
          <Loader size={40} style={{ animation: 'spin 1s linear infinite' }} />
          <p style={{ marginTop: '16px' }}>Cargando bots...</p>
        </div>
      ) : bots.length === 0 ? (
        <div className="glass-panel" style={{ textAlign: 'center', padding: '80px' }}>
          <Bot size={60} style={{ color: 'var(--text-secondary)', marginBottom: '16px' }} />
          <h2 style={{ color: 'var(--text-secondary)' }}>No tienes bots registrados</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>Crea tu primer bot para empezar. Si ya enviaste peticiones, usa "Verificar ID" para cargar la configuración automáticamente.</p>
          <button style={{ marginTop: '20px' }} onClick={() => { setEditingBot(null); setShowModal(true); }}><Plus size={16} /> Crear Primer Bot</button>
        </div>
      ) : (
        <div className="grid-cards">
          {bots.map(bot => {
            const t = parseTools(bot.tools_config);
            const activeTools = Object.entries(t).filter(([, v]) => v).map(([k]) => k);
            return (
              <div key={bot.id} className="glass-panel">
                <div className="flex-between" style={{ marginBottom: '10px' }}>
                  <Bot size={22} style={{ color: '#60a5fa' }} />
                  <span style={{ display: 'inline-block', padding: '3px 10px', background: bot.is_active ? 'rgba(16,185,129,0.2)' : 'rgba(148,163,184,0.15)', color: bot.is_active ? '#10b981' : '#94a3b8', borderRadius: '12px', fontSize: '12px', fontWeight: 600 }}>
                    {bot.is_active ? 'Activo' : 'Inactivo'}
                  </span>
                </div>
                <h3 style={{ margin: '0 0 2px', wordBreak: 'break-all', fontSize: '16px' }}>{bot.userbot_identifier}</h3>
                <p style={{ color: '#64748b', fontSize: '11px', margin: '0 0 10px' }}>#{bot.id} · {bot.ai_model} · {bot.pais}</p>

                {bot.system_prompt && (
                  <p style={{ fontSize: '12px', color: 'var(--text-secondary)', background: 'rgba(0,0,0,0.2)', padding: '8px 10px', borderRadius: '6px', margin: '0 0 12px', overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', lineHeight: 1.5 }}>
                    {bot.system_prompt}
                  </p>
                )}

                <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginBottom: '12px' }}>
                  {activeTools.map(k => (
                    <span key={k} style={{ fontSize: '10px', padding: '2px 8px', background: 'rgba(167,139,250,0.15)', color: '#a78bfa', borderRadius: '10px', border: '1px solid rgba(167,139,250,0.3)' }}>
                      {k.replace('_', ' ')}
                    </span>
                  ))}
                  {bot.activarnotificacion && (
                    <span style={{ fontSize: '10px', padding: '2px 8px', background: 'rgba(245,158,11,0.15)', color: '#f59e0b', borderRadius: '10px', border: '1px solid rgba(245,158,11,0.3)' }}>
                      notif: {bot.estado_notificacion || '?'}
                    </span>
                  )}
                </div>

                <div style={{ display: 'flex', gap: '6px', marginTop: '12px' }}>
                  <button onClick={() => setSimulatingBot(bot)} style={{ flex: 1, background: 'rgba(16,185,129,0.15)', border: '1px solid rgba(16,185,129,0.3)', color: '#10b981', padding: '10px 4px', fontSize: '13px' }} title="Probar simulador LangGraph">
                    <MessageSquare size={14} /> Probar
                  </button>
                  <button onClick={() => setUploadingBot(bot)} style={{ flex: 1, background: 'rgba(59,130,246,0.15)', border: '1px solid rgba(59,130,246,0.3)', color: '#60a5fa', padding: '10px 4px', fontSize: '13px' }} title="Subir base de conocimiento (PDF/TXT) para el bot">
                    <UploadCloud size={14} /> BD
                  </button>
                  <button onClick={() => setViewingKnowledgeBot(bot)} style={{ flex: 1, background: 'rgba(168, 85, 247, 0.1)', color: '#c084fc', border: '1px solid rgba(168, 85, 247, 0.2)', padding: '10px 4px', fontSize: '13px' }}>
                    <Database size={14} /> Ver RAG
                  </button>
                  <button id={`configure-${bot.id}`} onClick={() => { setEditingBot(bot); setShowModal(true); }} style={{ flex: 1, background: 'transparent', border: '1px solid var(--accent-primary)', color: 'var(--accent-primary)', padding: '10px 4px', fontSize: '13px' }}>
                    <Settings size={14} /> Editar
                  </button>
                  <button id={`delete-${bot.id}`} onClick={() => handleDelete(bot)} className="danger" style={{ padding: '10px 12px', width: 'auto' }}>
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {showModal && <BotModal bot={editingBot} onClose={() => { setShowModal(false); setEditingBot(null); }} onSave={loadBots} toast={showToast} />}
      {simulatingBot && <ChatSimulatorModal bot={simulatingBot} onClose={() => setSimulatingBot(null)} toast={showToast} />}
      {uploadingBot && <KnowledgeUploadModal bot={uploadingBot} onClose={() => setUploadingBot(null)} toast={showToast} />}
      {viewingKnowledgeBot && <KnowledgeViewerModal bot={viewingKnowledgeBot} onClose={() => setViewingKnowledgeBot(null)} toast={showToast} />}
      
      {toastMsg && <Toast msg={toastMsg.msg} type={toastMsg.type} onClose={() => setToastMsg(null)} />}
    </div>
  );
}

// ─── App Root ───────────────────────────────────────────────────────────
export default function App() {
  const auth = useAuth();
  return (
    <BrowserRouter>
      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes slideIn { from { transform: translateX(40px); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
        select { appearance: none; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 12px center; padding-right: 36px; }
      `}</style>
      <Routes>
        <Route path="/login" element={!auth.token ? <LoginPage auth={auth} /> : <Navigate to="/panel" />} />
        <Route path="/register" element={!auth.token ? <RegisterPage auth={auth} /> : <Navigate to="/panel" />} />
        <Route path="/panel" element={auth.token ? <DashboardPage auth={auth} /> : <Navigate to="/login" />} />
        <Route path="*" element={<Navigate to={auth.token ? '/panel' : '/login'} />} />
      </Routes>
    </BrowserRouter>
  );
}
