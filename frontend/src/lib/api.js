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

export async function detectPage(id, n) {
  return jsonOrThrow(await fetch(`${BASE}/documents/${id}/pages/${n}/detect`, { method: 'POST' }));
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
