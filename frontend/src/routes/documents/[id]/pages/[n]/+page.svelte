<script>
  import { page as pageStore } from '$app/stores';
  import { goto } from '$app/navigation';
  import {
    getDocument,
    detectPageStream,
    confirmPage,
    finalizeDocument,
    pageImageUrl,
    maskedPdfUrl
  } from '$lib/api.js';

  const PII_TYPES = [
    'NAME', 'EMAIL', 'PHONE', 'ADDRESS', 'SSN_NRIC',
    'DOB', 'CREDIT_CARD', 'BANK_ACCOUNT', 'IP'
  ];

  let docId = $derived(Number($pageStore.params.id));
  let pageNum = $derived(Number($pageStore.params.n));

  let doc = $state(null);
  let pageMeta = $state(null);
  let entities = $state([]);
  let confirming = $state(false);
  let finalizing = $state(false);
  let error = $state('');

  // Detection stream state
  let detecting = $state(false);
  let stageLog = $state([]); // [{stage, status, ...}]
  let abortCtrl = $state(null);

  // Drawing state
  let imgEl;
  let scale = $state(1);
  let drawing = $state(false);
  let drawStart = $state(null);
  let drawCurrent = $state(null);

  // Re-load metadata when doc/page changes but DO NOT auto-detect.
  $effect(() => {
    if (!docId || !pageNum) return;
    void loadMeta();
  });

  async function loadMeta() {
    error = '';
    entities = [];
    stageLog = [];
    pageMeta = null;
    try {
      doc = await getDocument(docId);
      pageMeta = doc.pages.find((p) => p.page_number === pageNum) ?? null;
      if (!pageMeta) {
        error = `Page ${pageNum} does not exist (document has ${doc.page_count}).`;
      }
    } catch (err) {
      error = err.message;
    }
  }

  async function runDetect() {
    if (detecting) return;
    detecting = true;
    error = '';
    entities = [];
    stageLog = [{ stage: 'starting', status: 'start', ts: Date.now() }];
    abortCtrl = new AbortController();
    try {
      await detectPageStream(
        docId,
        pageNum,
        (evt) => {
          stageLog = [...stageLog, { ...evt, ts: Date.now() }];
          if (evt.stage === 'done' && Array.isArray(evt.entities)) {
            entities = evt.entities.map((e) => ({ ...e, _enabled: true }));
          }
        },
        { signal: abortCtrl.signal }
      );
    } catch (err) {
      if (err.name === 'AbortError') {
        stageLog = [...stageLog, { stage: 'cancelled', status: 'done', ts: Date.now() }];
      } else {
        error = err.message;
      }
    } finally {
      detecting = false;
      abortCtrl = null;
    }
  }

  function cancelDetect() {
    abortCtrl?.abort();
  }

  function onImgLoad() { recomputeScale(); }
  function recomputeScale() {
    if (!imgEl || !pageMeta) return;
    scale = imgEl.clientWidth / pageMeta.width;
  }

  function toImageCoords(evt) {
    const rect = imgEl.getBoundingClientRect();
    const x = (evt.clientX - rect.left) / scale;
    const y = (evt.clientY - rect.top) / scale;
    return { x: Math.max(0, x), y: Math.max(0, y) };
  }
  function startDraw(evt) {
    if (!imgEl || evt.button !== 0) return;
    evt.preventDefault();
    drawing = true;
    drawStart = toImageCoords(evt);
    drawCurrent = drawStart;
  }
  function moveDraw(evt) {
    if (!drawing) return;
    drawCurrent = toImageCoords(evt);
  }
  function endDraw() {
    if (!drawing || !drawStart || !drawCurrent) {
      drawing = false; drawStart = drawCurrent = null; return;
    }
    const x = Math.round(Math.min(drawStart.x, drawCurrent.x));
    const y = Math.round(Math.min(drawStart.y, drawCurrent.y));
    const w = Math.round(Math.abs(drawCurrent.x - drawStart.x));
    const h = Math.round(Math.abs(drawCurrent.y - drawStart.y));
    drawing = false; drawStart = drawCurrent = null;
    if (w > 4 && h > 4) {
      entities = [...entities, { type: 'NAME', text: '(manual)', x, y, w, h, _enabled: true }];
    }
  }

  function removeEntity(i) { entities = entities.filter((_, idx) => idx !== i); }
  function toggleEntity(i) {
    entities = entities.map((e, idx) => (idx === i ? { ...e, _enabled: !e._enabled } : e));
  }
  function updateEntityType(i, value) {
    entities = entities.map((e, idx) => (idx === i ? { ...e, type: value } : e));
  }

  async function onConfirm() {
    confirming = true;
    error = '';
    try {
      const payload = entities.filter((e) => e._enabled)
        .map(({ _enabled, id, ...rest }) => rest);
      await confirmPage(docId, pageNum, payload);
      doc = await getDocument(docId);
      if (pageNum < doc.page_count) {
        goto(`/documents/${docId}/pages/${pageNum + 1}`);
      } else {
        pageMeta = doc.pages.find((p) => p.page_number === pageNum) ?? pageMeta;
      }
    } catch (err) {
      error = err.message;
    } finally {
      confirming = false;
    }
  }

  async function onFinalize() {
    finalizing = true;
    error = '';
    try {
      await finalizeDocument(docId);
      doc = await getDocument(docId);
    } catch (err) {
      error = err.message;
    } finally {
      finalizing = false;
    }
  }

  function gotoPage(n) {
    if (n < 1 || (doc && n > doc.page_count)) return;
    if (detecting) cancelDetect();
    goto(`/documents/${docId}/pages/${n}`);
  }

  let allConfirmed = $derived(doc && doc.pages.every((p) => p.confirmed));
  let isFinalized = $derived(doc?.status === 'finalized');

  function formatStage(evt) {
    const ms = evt.elapsed_ms != null ? ` (${(evt.elapsed_ms / 1000).toFixed(1)}s)` : '';
    const count = evt.count != null ? ` — ${evt.count}` : '';
    if (evt.stage === 'vl_ocr') {
      if (evt.status === 'start') return 'Running Qwen2.5-VL (OCR)…';
      if (evt.status === 'done') return `VL done${count} chunks${ms}`;
    }
    if (evt.stage === 'pii_tag') {
      if (evt.status === 'start') return 'Running Qwen3 (PII tagging)…';
      if (evt.status === 'done') return `PII done${count} entities${ms}`;
    }
    if (evt.stage === 'done') return `Matched ${evt.matched ?? 0}/${evt.pii_found ?? 0} to bboxes`;
    if (evt.stage === 'cancelled') return 'Cancelled';
    if (evt.stage === 'starting') return 'Starting…';
    return `${evt.stage} ${evt.status ?? ''}`;
  }
</script>

<svelte:window onresize={recomputeScale} />

{#if error}<p class="err">{error}</p>{/if}

{#if doc && pageMeta}
  <div class="topbar">
    <div>
      <strong>{doc.filename}</strong>
      <span class="muted"> · page {pageNum} of {doc.page_count}</span>
      <span class="badge {pageMeta.confirmed ? 'ok' : 'pending'}">
        {pageMeta.confirmed ? 'confirmed' : 'pending'}
      </span>
    </div>
    <div class="nav">
      <button onclick={() => gotoPage(pageNum - 1)} disabled={pageNum <= 1}>‹ Prev</button>
      <button onclick={() => gotoPage(pageNum + 1)} disabled={pageNum >= doc.page_count}>Next ›</button>
      {#if detecting}
        <button onclick={cancelDetect}>Cancel detect</button>
      {:else}
        <button onclick={runDetect}>{entities.length ? 'Re-run detect' : 'Detect PII'}</button>
      {/if}
      <button class="primary" onclick={onConfirm} disabled={confirming || detecting}>
        {confirming ? 'Saving…' : pageNum < doc.page_count ? 'Confirm & next' : 'Confirm page'}
      </button>
      {#if allConfirmed && !isFinalized}
        <button class="primary" onclick={onFinalize} disabled={finalizing}>
          {finalizing ? 'Finalizing…' : 'Finalize document'}
        </button>
      {/if}
      {#if isFinalized}
        <a class="primary btn" href={maskedPdfUrl(docId)} target="_blank" rel="noopener">Download masked PDF</a>
      {/if}
    </div>
  </div>

  {#if stageLog.length > 0}
    <div class="progress">
      <strong>Pipeline:</strong>
      <ul>
        {#each stageLog as evt, i (i)}
          <li>{formatStage(evt)}</li>
        {/each}
      </ul>
    </div>
  {/if}

  <div class="layout">
    <div class="canvas">
      <div
        class="img-wrap"
        onmousedown={startDraw}
        onmousemove={moveDraw}
        onmouseup={endDraw}
        onmouseleave={endDraw}
        role="presentation"
      >
        <img
          bind:this={imgEl}
          src={pageImageUrl(docId, pageNum)}
          alt={`Page ${pageNum}`}
          onload={onImgLoad}
          draggable="false"
        />
        {#if pageMeta}
          <svg
            class="overlay"
            viewBox={`0 0 ${pageMeta.width} ${pageMeta.height}`}
            preserveAspectRatio="none"
          >
            {#each entities as e, i (i)}
              {#if e._enabled}
                <rect
                  x={e.x}
                  y={e.y}
                  width={e.w}
                  height={e.h}
                  fill="rgba(255,0,0,0.25)"
                  stroke="#f87171"
                  stroke-width="2"
                  vector-effect="non-scaling-stroke"
                />
              {/if}
            {/each}
            {#if drawing && drawStart && drawCurrent}
              <rect
                x={Math.min(drawStart.x, drawCurrent.x)}
                y={Math.min(drawStart.y, drawCurrent.y)}
                width={Math.abs(drawCurrent.x - drawStart.x)}
                height={Math.abs(drawCurrent.y - drawStart.y)}
                fill="rgba(96,165,250,0.2)"
                stroke="#60a5fa"
                stroke-width="2"
                vector-effect="non-scaling-stroke"
              />
            {/if}
          </svg>
        {/if}
      </div>
      <p class="hint">Drag on the image to add a manual redaction box. Detection only runs when you click "Detect PII".</p>
    </div>

    <aside class="sidebar">
      <h3>Detections ({entities.filter((e) => e._enabled).length})</h3>
      {#if detecting}
        <p class="muted">Running… click Cancel to stop.</p>
      {:else if entities.length === 0}
        <p class="muted">No entities yet. Click "Detect PII" or draw boxes manually.</p>
      {:else}
        <ul>
          {#each entities as e, i (i)}
            <li class:disabled={!e._enabled}>
              <label class="row">
                <input
                  type="checkbox"
                  checked={e._enabled}
                  onchange={() => toggleEntity(i)}
                />
                <select value={e.type} onchange={(ev) => updateEntityType(i, ev.currentTarget.value)}>
                  {#each PII_TYPES as t}
                    <option value={t}>{t}</option>
                  {/each}
                </select>
                <span class="text" title={e.text}>{e.text}</span>
                <button class="x" onclick={() => removeEntity(i)} aria-label="Remove">✕</button>
              </label>
            </li>
          {/each}
        </ul>
      {/if}
    </aside>
  </div>
{:else if !error}
  <p>Loading…</p>
{/if}

<style>
  .err { color: #f87171; }
  .muted { color: #888; }
  .topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 12px;
  }
  .nav { display: flex; gap: 8px; flex-wrap: wrap; }
  button, .btn {
    background: #2a2f3a;
    color: #e6e6e6;
    border: 1px solid #3a4150;
    border-radius: 6px;
    padding: 6px 12px;
    cursor: pointer;
    text-decoration: none;
    font: inherit;
  }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .primary { background: #2563eb; border-color: #2563eb; }
  .badge {
    display: inline-block;
    margin-left: 8px;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 12px;
  }
  .badge.ok { background: #14532d; color: #bbf7d0; }
  .badge.pending { background: #422006; color: #fed7aa; }
  .progress {
    background: #16181d;
    border: 1px solid #2a2f3a;
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 12px;
    font-size: 13px;
  }
  .progress ul { margin: 4px 0 0; padding-left: 20px; }
  .layout {
    display: grid;
    grid-template-columns: 1fr 320px;
    gap: 16px;
  }
  @media (max-width: 900px) {
    .layout { grid-template-columns: 1fr; }
  }
  .canvas { min-width: 0; }
  .img-wrap {
    position: relative;
    user-select: none;
    cursor: crosshair;
    background: #1a1d24;
    border: 1px solid #2a2f3a;
    border-radius: 6px;
    overflow: hidden;
  }
  .img-wrap img { display: block; width: 100%; height: auto; }
  .overlay {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
  }
  .hint { color: #888; font-size: 12px; }
  .sidebar {
    background: #16181d;
    border: 1px solid #2a2f3a;
    border-radius: 6px;
    padding: 12px;
    max-height: 80vh;
    overflow: auto;
  }
  .sidebar h3 { margin-top: 0; }
  .sidebar ul { list-style: none; padding: 0; margin: 0; }
  .sidebar li { padding: 4px 0; border-bottom: 1px solid #232732; }
  .sidebar li.disabled { opacity: 0.5; }
  .row { display: flex; gap: 6px; align-items: center; }
  .row select {
    background: #0f1115; color: #e6e6e6; border: 1px solid #3a4150; border-radius: 4px; padding: 2px 4px; font: inherit;
  }
  .row .text {
    flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px;
  }
  .row .x {
    background: transparent; border: 0; color: #f87171; padding: 0 6px; cursor: pointer;
  }
</style>
