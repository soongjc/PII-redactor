const BASE = '/api';

async function jsonOrThrow(res) {
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) msg = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch {}
    throw new Error(msg);
  }
  return res.json();
}

export async function uploadDocument(file) {
  const fd = new FormData();
  fd.append('file', file);
  return jsonOrThrow(await fetch(`${BASE}/documents`, { method: 'POST', body: fd }));
}

export async function getDocument(id) {
  return jsonOrThrow(await fetch(`${BASE}/documents/${id}`));
}

/**
 * Stream NDJSON stage events from /detect. Calls `onEvent(event)` for each
 * parsed event. Pass an `AbortSignal` via opts.signal to cancel.
 */
export async function detectPageStream(id, n, onEvent, opts = {}) {
  const res = await fetch(`${BASE}/documents/${id}/pages/${n}/detect`, {
    method: 'POST',
    signal: opts.signal
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) msg = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch {}
    throw new Error(msg);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf('\n')) !== -1) {
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (!line) continue;
      try {
        onEvent(JSON.parse(line));
      } catch (e) {
        console.warn('Bad NDJSON line:', line);
      }
    }
  }
  if (buf.trim()) {
    try { onEvent(JSON.parse(buf.trim())); } catch {}
  }
}

export async function confirmPage(id, n, entities) {
  return jsonOrThrow(
    await fetch(`${BASE}/documents/${id}/pages/${n}/confirm`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ entities })
    })
  );
}

export async function finalizeDocument(id) {
  return jsonOrThrow(await fetch(`${BASE}/documents/${id}/finalize`, { method: 'POST' }));
}

export function pageImageUrl(id, n) {
  return `${BASE}/documents/${id}/pages/${n}/image`;
}

export function maskedImageUrl(id, n) {
  return `${BASE}/documents/${id}/pages/${n}/masked`;
}

export function maskedPdfUrl(id) {
  return `${BASE}/documents/${id}/masked.pdf`;
}
