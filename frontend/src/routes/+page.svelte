<script>
  import { goto } from '$app/navigation';
  import { uploadDocument } from '$lib/api.js';

  let file = $state(null);
  let busy = $state(false);
  let error = $state('');

  async function onSubmit(e) {
    e.preventDefault();
    if (!file) return;
    busy = true;
    error = '';
    try {
      const doc = await uploadDocument(file);
      goto(`/documents/${doc.id}/pages/1`);
    } catch (err) {
      error = err.message;
    } finally {
      busy = false;
    }
  }
</script>

<h1>Upload a PDF</h1>
<p>Pages are reviewed one at a time. Detected PII is masked when you confirm each page.</p>

<form onsubmit={onSubmit}>
  <input
    type="file"
    accept="application/pdf"
    onchange={(e) => (file = e.currentTarget.files?.[0] ?? null)}
  />
  <button type="submit" disabled={!file || busy}>
    {busy ? 'Uploading…' : 'Upload'}
  </button>
</form>

{#if error}<p class="err">{error}</p>{/if}

<style>
  form {
    display: flex;
    gap: 12px;
    align-items: center;
    margin-top: 16px;
  }
  button {
    background: #2563eb;
    border: 0;
    color: white;
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
  }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .err { color: #f87171; }
</style>
