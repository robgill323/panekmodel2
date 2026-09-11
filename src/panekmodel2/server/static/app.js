/* Throughline v2 — frontend.
   Implements artifacts/design/throughline-v2-design.html against the live
   pipeline API. No build step, no external runtime dependencies. */
'use strict';

// ── state ───────────────────────────────────────────────────────────
const S = {
  screen: 'run',
  theme: localStorage.getItem('tl-theme') || 'light',
  urlText: '',
  adv: false,
  /* Only the settings this UI actually exposes a control for. chunk_max_words
     is deliberately absent: chunk size is chosen in seconds, and the server
     derives the paired word cap. Sending a concrete word cap here would
     suppress that derivation and cut a "120 s" chunk short at ~77 s. */
  settings: {
    chunk_max_seconds: 60,
    embedding_model: 'all-mpnet-base-v2',
    sentiment_model: 'cardiffnlp/twitter-roberta-base-sentiment-latest',
    topic_reduce_to: 10,
    use_whisper_fallback: false,
    detect_people: true,
  },
  jobId: null,
  progress: null,
  results: null,
  runError: '',
  starting: false,
  topicId: null,
  videoId: null,
  t: 0,
  playing: false,
  dim: null,
  hoverIndex: null,
  sort: 'share',
  exportKind: 'combined',
  excerptMode: 'representative',
  outlierVideo: 'all',
  // Auto-follow is on until the reader scrolls the transcript themselves.
  followTranscript: true,
};

const CHUNK_OPTS = [15, 30, 60, 120];
const EMBED_OPTS = [
  ['all-MiniLM-L6-v2', 'fast, 384-dim'],
  ['all-mpnet-base-v2', 'stronger clusters, slower · default'],
  ['BAAI/bge-base-en-v1.5', 'best recall, GPU recommended'],
];
const SENT_OPTS = [
  ['cardiffnlp/twitter-roberta-base-sentiment-latest', 'three-way with a real neutral class · tuned on speech-like text · default'],
  ['siebert/sentiment-roberta-large-english', 'binary, no neutral — every chunk is forced positive or negative'],
  ['distilbert-base-uncased-finetuned-sst-2-english', 'binary, faster, coarser'],
];
// Below this assignment probability a chunk sits at the edge of its topic; the
// UI marks it rather than presenting the assignment as settled.
const LOW_CONFIDENCE = 0.5;
const OUTLIER_COLOR = 'var(--line2)';

// ── small helpers ───────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);

function esc(value) {
  return String(value == null ? '' : value).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}
function fmtT(seconds) {
  const s = Math.max(0, Math.round(seconds || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  const mm = h ? String(m).padStart(2, '0') : String(m);
  return (h ? h + ':' : '') + mm + ':' + String(r).padStart(2, '0');
}
function fmtDur(seconds) {
  const s = Math.max(0, Math.round(seconds || 0));
  const h = Math.floor(s / 3600);
  const m = Math.round((s % 3600) / 60);
  return h ? h + 'h ' + m + 'm' : m + ' min';
}
function sig(v) {
  const n = Number(v) || 0;
  return (n >= 0 ? '+' : '−') + Math.abs(n).toFixed(2);
}
function pct(v, digits) {
  return ((Number(v) || 0) * 100).toFixed(digits == null ? 1 : digits) + '%';
}
function sentHex(v) {
  const dark = S.theme === 'dark';
  if (v > 0.08) return dark ? '#3AAFA2' : '#14746F';
  if (v < -0.08) return dark ? '#E0705C' : '#B4402F';
  return '#A29C91';
}
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// ── derived lookups ─────────────────────────────────────────────────
function topicIndex() {
  const map = new Map();
  const topics = (S.results && S.results.topics) || [];
  topics.forEach((t, rank) => map.set(t.topic_id, Object.assign({ rank }, t)));
  return map;
}
function topicInfo(id) {
  const t = topicIndex().get(id);
  if (t) return t;
  return { topic_id: -1, rank: -1, label: 'Outlier bin', color: OUTLIER_COLOR, keywords: [] };
}
function excerptsFor(topic) {
  return (S.excerptMode === 'polarized' ? topic.excerpts_polarized : topic.excerpts) || [];
}

/* Assignment confidence, stated honestly. A chunk that outlier reduction moved
   into this topic has no probability for it, so we say so instead of showing a
   number that describes a topic it is no longer in. */
function confidenceLabel(chunk) {
  if (chunk.topic_reassigned) return 'reassigned from outliers';
  if (chunk.topic_prob == null) return 'confidence n/a';
  const label = 'assignment ' + Math.round(chunk.topic_prob * 100) + '%';
  return chunk.topic_prob < LOW_CONFIDENCE ? label + ' · weak' : label;
}

function isLowConfidence(chunk) {
  return chunk.topic_reassigned || chunk.topic_prob == null || chunk.topic_prob < LOW_CONFIDENCE;
}

function currentVideo() {
  if (!S.results) return null;
  return S.results.videos.find((v) => v.video_id === S.videoId) || S.results.videos[0] || null;
}
function currentTopic() {
  if (!S.results || !S.results.topics.length) return null;
  return S.results.topics.find((t) => t.topic_id === S.topicId) || S.results.topics[0];
}
function hasResults() { return !!(S.results && S.results.videos.length); }

// ── api ─────────────────────────────────────────────────────────────
async function api(path, options) {
  const res = await fetch(path, Object.assign({ headers: { 'Content-Type': 'application/json' } }, options));
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch (_) { body = text; }
  if (!res.ok) {
    const detail = body && body.detail ? body.detail : res.statusText;
    throw new Error(detail);
  }
  return body;
}

// ── swatches & patterns ─────────────────────────────────────────────
let patternSeq = 0;

function patternMarkup(id, color, kind) {
  if (kind === 0) return '';
  const kids = kind === 1
    ? '<path d="M0,6 l6,-6 M-1.5,1.5 l3,-3 M4.5,7.5 l3,-3" stroke="rgba(255,255,255,.55)" stroke-width="1.4"/>'
    : kind === 2
      ? '<circle cx="3" cy="3" r="1.1" fill="rgba(255,255,255,.6)"/>'
      : '<path d="M3,0 l0,6" stroke="rgba(255,255,255,.5)" stroke-width="1.4"/>';
  return `<pattern id="${id}" width="6" height="6" patternUnits="userSpaceOnUse">
    <rect width="6" height="6" fill="${color}"/>${kids}</pattern>`;
}

/* Every topic colour also carries a fill pattern, so hue is never the only
   encoding (design requirement, and a deuteranopia safeguard). */
function fillFor(prefix, rank, color) {
  return rank % 4 === 0 ? color : `url(#${prefix}-p${rank})`;
}

function swatch(id, size) {
  const t = topicInfo(id);
  const s = size || 14;
  if (t.rank < 0) {
    return `<span class="hatch" style="display:block;width:${s}px;height:${s}px;border-radius:2px"></span>`;
  }
  const pre = 'sw' + (patternSeq++);
  return `<svg width="${s}" height="${s}" viewBox="0 0 6 6" style="display:block;border-radius:2px" aria-hidden="true">
    <defs>${patternMarkup(pre + '-p' + t.rank, t.color, t.rank % 4)}</defs>
    <rect width="6" height="6" fill="${fillFor(pre, t.rank, t.color)}"/></svg>`;
}

function valMini(v) {
  const W = 54, H = 10;
  const val = clamp(Number(v) || 0, -1, 1);
  const x = val < 0 ? W / 2 + val * (W / 2) : W / 2;
  return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="display:block" aria-hidden="true">
    <rect x="0" y="4" width="${W}" height="2" fill="var(--line)"/>
    <line x1="${W / 2}" y1="0" x2="${W / 2}" y2="${H}" stroke="var(--line2)" stroke-width="1"/>
    <rect x="${x}" y="2" width="${Math.max(1.5, Math.abs(val) * (W / 2))}" height="6" fill="${sentHex(val)}" rx="1"/></svg>`;
}

function valChip(v) {
  return `<span style="display:inline-flex;align-items:center;gap:7px;justify-content:flex-end">
    ${valMini(v)}<span style="font:500 12px var(--mono);color:${sentHex(v)};min-width:44px;text-align:right">${sig(v)}</span></span>`;
}

function stackBar(mix, height) {
  const h = height || 18;
  const parts = mix.map((m) => {
    const t = topicInfo(m.topic_id);
    if (t.rank < 0) return `<span class="hatch" style="width:${m.share * 100}%"></span>`;
    const pre = 'sb' + (patternSeq++);
    const inner = t.rank % 4 === 0 ? '' :
      `<svg width="100%" height="100%" style="display:block"><defs>${patternMarkup(pre, t.color, t.rank % 4)}</defs><rect width="100%" height="100%" fill="url(#${pre})"/></svg>`;
    return `<span title="${esc(t.label)}" style="width:${m.share * 100}%;background:${t.color}">${inner}</span>`;
  }).join('');
  return `<div class="stack-bar" style="height:${h}px">${parts}</div>`;
}

// ── the throughline ─────────────────────────────────────────────────
const TL_CFG = {
  full: { lane: 26, gap: 9, rib: 104, axis: 24, px: 163 },
  med: { lane: 13, gap: 6, rib: 45, axis: 0, px: 64 },
  micro: { lane: 6, gap: 3, rib: 17, axis: 0, px: 26 },
};

function smoothPath(points) {
  let d = 'M' + points[0][0] + ',' + points[0][1];
  for (let i = 1; i < points.length; i++) {
    const p = points[i - 1], c = points[i], mx = (p[0] + c[0]) / 2;
    d += ' C' + mx + ',' + p[1] + ' ' + mx + ',' + c[1] + ' ' + c[0] + ',' + c[1];
  }
  return d;
}

/* Topic lane on top (contiguous chunks, chapter-like) with the sentiment
   ribbon beneath: above the centerline positive, below negative, height =
   |valence|. Interactive at full size, static at med/micro. */
function throughline(video, size, opts) {
  opts = opts || {};
  const cfg = TL_CFG[size];
  const W = 1000;
  const H = cfg.lane + cfg.gap + cfg.rib + cfg.axis;
  const dur = video.duration_s || 1;
  const chunks = video.chunks || [];
  if (!chunks.length) return '<div class="mono-note">no chunks</div>';

  const pre = 'tl' + (patternSeq++);
  const mid = cfg.lane + cfg.gap + cfg.rib / 2;
  const amp = cfg.rib / 2 - 2;
  const x = (t) => clamp(t / dur, 0, 1) * W;

  // contiguous same-topic runs become one lane segment
  const segs = [];
  chunks.forEach((c) => {
    const last = segs[segs.length - 1];
    if (last && last.topic_id === c.topic_id) last.end = c.end;
    else segs.push({ topic_id: c.topic_id, start: c.start, end: c.end });
  });

  const ranks = new Set(segs.map((s) => topicInfo(s.topic_id).rank).filter((r) => r >= 0));
  const defs = [...ranks].map((r) => {
    const t = (S.results.topics || [])[r];
    return t ? patternMarkup(pre + '-p' + r, t.color, r % 4) : '';
  }).join('') +
    `<clipPath id="${pre}-cu"><rect x="0" y="${cfg.lane + cfg.gap}" width="${W}" height="${cfg.rib / 2}"/></clipPath>` +
    `<clipPath id="${pre}-cd"><rect x="0" y="${mid}" width="${W}" height="${cfg.rib / 2}"/></clipPath>`;

  const lane = segs.map((s) => {
    const t = topicInfo(s.topic_id);
    const w = Math.max(1, x(s.end) - x(s.start) - (size === 'micro' ? 0.6 : 1.2));
    const fill = t.rank < 0 ? 'var(--line2)' : fillFor(pre, t.rank, t.color);
    const dimmed = S.dim != null && S.dim !== s.topic_id ? 0.16 : 1;
    return `<rect x="${x(s.start)}" y="0" width="${w}" height="${cfg.lane}" fill="${fill}" rx="${size === 'micro' ? 1 : 2}" opacity="${dimmed}"/>`;
  }).join('');

  const pts = chunks.map((c) => [x((c.start + c.end) / 2), mid - clamp(c.valence, -1, 1) * amp]);
  pts.unshift([0, mid - clamp(chunks[0].valence, -1, 1) * amp * 0.6]);
  pts.push([W, mid - clamp(chunks[chunks.length - 1].valence, -1, 1) * amp * 0.6]);
  const area = smoothPath(pts) + ' L' + W + ',' + mid + ' L0,' + mid + ' Z';
  const posC = S.theme === 'dark' ? '#3AAFA2' : '#14746F';
  const negC = S.theme === 'dark' ? '#E0705C' : '#B4402F';

  const ribbon =
    `<g clip-path="url(#${pre}-cu)"><path d="${area}" fill="${posC}" fill-opacity=".82"/></g>` +
    `<g clip-path="url(#${pre}-cd)"><path d="${area}" fill="${negC}" fill-opacity=".82"/></g>` +
    `<line x1="0" y1="${mid}" x2="${W}" y2="${mid}" stroke="currentColor" stroke-opacity=".28" stroke-width="${size === 'micro' ? 0.8 : 1}"/>`;

  const plotH = cfg.lane + cfg.gap + cfg.rib;
  let extras = '';
  if (cfg.axis) {
    const step = dur / 6;
    for (let i = 0; i <= 6; i++) {
      extras += `<line x1="${x(i * step)}" y1="${plotH - 5}" x2="${x(i * step)}" y2="${plotH}" stroke="currentColor" stroke-opacity=".35"/>`;
    }
  }
  if (opts.interactive) {
    extras += `<line class="tl-hover" x1="-10" y1="0" x2="-10" y2="${plotH}" stroke="currentColor" stroke-opacity=".4" stroke-dasharray="3 3"/>`;
    extras += `<g class="tl-playhead" transform="translate(0,0)"><line x1="0" y1="0" x2="0" y2="${plotH}" stroke="currentColor" stroke-width="1.6"/><circle cx="0" cy="0" r="4" fill="currentColor"/></g>`;
  }

  /* The lane and ribbon are meant to stretch to the container width, which is
     what preserveAspectRatio="none" does — but that same non-uniform scale
     squashes or stretches any text inside the SVG depending on viewport width.
     Labels therefore live in HTML beside it, at natural size. */
  const svg = `<svg viewBox="0 0 ${W} ${plotH}" preserveAspectRatio="none"
    style="width:100%;height:${cfg.px - cfg.axis}px;display:block;color:var(--ink);cursor:${opts.interactive ? 'crosshair' : 'default'}"
    ${opts.interactive ? 'data-tl-interactive="1" role="slider" tabindex="0" aria-label="Seek video" aria-valuemin="0" aria-valuemax="' + Math.round(dur) + '" aria-valuenow="' + Math.round(S.t) + '"' : 'aria-hidden="true"'}>
    <defs>${defs}</defs><g>${lane}</g><g>${ribbon}</g><g>${extras}</g></svg>`;

  if (!cfg.axis) return svg;

  const ticks = Array.from({ length: 7 }, (_, i) => {
    const pctX = (i / 6) * 100;
    const align = i === 0 ? 'left:0;text-align:left'
      : i === 6 ? 'right:0;text-align:right'
        : `left:${pctX}%;transform:translateX(-50%)`;
    return `<span class="tl-tick" style="${align}">${fmtT((i / 6) * dur)}</span>`;
  }).join('');

  // Percentages are of the plot box, which matches the SVG exactly — the axis
  // row sits outside it, so it cannot push the ±1 markers off the ribbon.
  const ribTop = ((cfg.lane + cfg.gap) / plotH) * 100;
  const ribBottom = 100 - ribTop - (cfg.rib / plotH) * 100;
  return `<div class="tl-frame">
    <div class="tl-plot">
      ${svg}
      <span class="tl-ybound" style="top:${ribTop}%;color:${posC}">+1</span>
      <span class="tl-ybound" style="bottom:${ribBottom}%;color:${negC}">−1</span>
    </div>
    <div class="tl-axis" style="height:${cfg.axis}px">${ticks}</div>
  </div>`;
}

// ── nav & chrome ────────────────────────────────────────────────────
const NAV = [
  ['run', 'New run', () => true],
  ['prog', 'Run progress', () => !!S.jobId],
  ['batch', 'Batch overview', () => hasResults()],
  ['topics', 'Topics', () => hasResults()],
  ['tsent', 'Topic sentiment', () => hasResults()],
  ['video', 'Video page', () => hasResults()],
  ['export', 'Export', () => hasResults()],
  // Style guide, not a research view — separated at the bottom of the nav.
  ['found', 'Foundations', () => true, 'utility'],
];

function navTag(key) {
  if (key === 'prog' && S.progress) {
    return S.progress.status === 'running' ? '◷' : '';
  }
  if (key === 'batch' && hasResults()) return String(S.results.totals.n_analyzed);
  if (key === 'topics' && hasResults()) return String(S.results.topics.length);
  return '';
}

function renderNav() {
  $('#nav').innerHTML = NAV.map(([key, label, enabled, kind]) => {
    const on = enabled();
    return `<button type="button" data-go="${key}" class="${kind === 'utility' ? 'nav-utility' : ''}"
      ${on ? '' : 'disabled'} ${S.screen === key ? 'aria-current="page"' : ''}>
      <span class="nav-dot"></span><span class="nav-label">${esc(label)}</span>
      <span class="nav-tag">${esc(navTag(key))}</span></button>`;
  }).join('');
}

function renderSessionCard() {
  const st = S.settings;
  let body;
  if (hasResults()) {
    const T = S.results.totals;
    // The settings that produced these results — not whatever the form shows now.
    const rs = S.results.settings;
    body = `Run ${runClock()} · ${T.n_analyzed} video${T.n_analyzed === 1 ? '' : 's'}<br>
      ${rs.chunk_max_seconds} s chunks · ${esc(shortModel(rs.embedding_model))}<br>
      ${T.n_topics} topics + outlier bin`;
  } else {
    body = `No run yet.<br>${st.chunk_max_seconds} s chunks · ${esc(shortModel(st.embedding_model))}`;
  }
  $('#session-card').innerHTML = `<div class="kicker">THIS SESSION</div>
    <div class="body">${body}</div>
    <div class="note">Results are held in memory. Closing the app discards them — export before you quit.</div>`;
}

function shortModel(name) {
  return String(name || '').split('/').pop();
}
function runClock() {
  const started = S.results && S.results.run && S.results.run.started_at;
  if (!started) return '—';
  return new Date(started * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

const CRUMBS = {
  run: [['Throughline', 'run'], ['New run', 'run']],
  prog: [['Throughline', 'run'], ['Run', 'prog']],
  batch: [['Run', 'prog'], ['Batch overview', 'batch']],
  topics: [['Run', 'prog'], ['Batch overview', 'batch'], ['Topics', 'topics']],
  tsent: [['Run', 'prog'], ['Batch overview', 'batch'], ['Topic sentiment', 'tsent']],
  video: [['Run', 'prog'], ['Batch overview', 'batch'], ['Video', 'video']],
  export: [['Run', 'prog'], ['Batch overview', 'batch'], ['Export', 'export']],
  outliers: [['Batch overview', 'batch'], ['Topics', 'topics'], ['Outlier bin', 'outliers']],
  found: [['Throughline', 'run'], ['Foundations', 'found']],
};

function renderChrome() {
  const crumbs = CRUMBS[S.screen] || CRUMBS.run;
  $('#crumbs').innerHTML = crumbs.map(([label, target], i) => {
    const last = i === crumbs.length - 1;
    return `<span style="display:flex;align-items:center;gap:8px">
      <button type="button" data-go="${target}" class="${last ? 'current' : ''}">${esc(label)}</button>
      ${last ? '' : '<span class="sep">/</span>'}</span>`;
  }).join('');

  const tag = $('#run-tag');
  let cls = 'run-tag', text = 'no run yet';
  if (S.progress && S.progress.status === 'running') { cls += ' working'; text = 'running' + ' · ' + S.progress.counts.ok + '/' + S.progress.counts.total; }
  else if (hasResults()) { cls += ' live'; text = 'run ' + runClock() + ' · session only'; }
  else if (S.progress && S.progress.status === 'failed') { text = 'run failed'; }
  tag.className = cls;
  tag.innerHTML = `<span class="dot"></span><span>${esc(text)}</span>`;
  $('#theme-label').textContent = S.theme === 'light' ? 'Light' : 'Dark';
  renderNav();
  renderSessionCard();
}

// ── screen: new run ─────────────────────────────────────────────────
function screenRun() {
  const urls = parseUrls(S.urlText);
  const st = S.settings;
  const est = Math.max(1, Math.round(urls.length * (st.use_whisper_fallback ? 4.2 : 0.6)));
  const summary = [
    ['URLs parsed', String(urls.length)],
    ['Duplicates removed', String(countRawLines(S.urlText) - urls.length)],
    ['Chunk size', st.chunk_max_seconds + ' s'],
    ['Embeddings', shortModel(st.embedding_model)],
    ['Sentiment', shortModel(st.sentiment_model)],
    ['Whisper fallback', st.use_whisper_fallback ? 'on · slow' : 'off'],
    ['Est. run time', urls.length ? '~' + est + ' min' : '—'],
  ];

  return `<div class="run-grid">
    <div class="run-left">
      <div>
        <div class="kicker">NEW ANALYSIS RUN</div>
        <h1 class="run-title">Paste the videos you want read.</h1>
        <p class="lede">One URL per line — watch, Shorts and Live forms all work. One video is a
          perfectly good run: the topic model just fits over that single transcript.</p>
      </div>

      <div class="url-box">
        <div class="url-box-head">
          <span class="label">VIDEO URLS</span>
          <span class="mono-note">${urls.length} URL${urls.length === 1 ? '' : 's'} · ${urls.length} unique</span>
        </div>
        <textarea id="url-input" spellcheck="false"
          placeholder="https://www.youtube.com/watch?v=…&#10;https://youtu.be/…&#10;https://www.youtube.com/shorts/…">${esc(S.urlText)}</textarea>
        <div class="url-box-foot">
          <label class="btn-dashed" for="csv-input">
            <span style="font-family:var(--mono);font-size:11px">CSV</span><span>Merge a URL column</span>
          </label>
          <input id="csv-input" type="file" accept=".csv,text/csv" class="sr-only">
          <span class="hint">Uploaded rows are appended to the list above and de-duplicated — nothing is replaced.</span>
        </div>
      </div>

      <div class="card">
        <button type="button" class="adv-toggle" data-action="toggle-adv">
          <span style="display:flex;flex-direction:column;gap:3px">
            <span style="font-size:13.5px;font-weight:600">Advanced settings</span>
            <span class="mono-note">${st.chunk_max_seconds} s chunks · ${esc(shortModel(st.embedding_model))} · ${esc(shortModel(st.sentiment_model))}${st.use_whisper_fallback ? ' · Whisper on' : ''}</span>
          </span>
          <span class="mono-note">${S.adv ? '▲' : '▼'}</span>
        </button>
        ${S.adv ? advancedPanel() : ''}
      </div>
    </div>

    <div class="run-side">
      <div class="card card-pad">
        <div class="kicker" style="margin-bottom:14px">RUN SUMMARY</div>
        <div style="display:flex;flex-direction:column;gap:9px;margin-bottom:18px">
          ${summary.map(([k, v]) => `<div class="summary-row"><span>${esc(k)}</span><span>${esc(v)}</span></div>`).join('')}
        </div>
        <button type="button" class="btn-primary btn-block" data-action="start-run" ${urls.length && !S.starting ? '' : 'disabled'}>
          ${S.starting ? 'Starting…' : 'Run analysis'}</button>
        <div class="hint" style="margin-top:10px;text-align:center">Nothing leaves this machine except the transcript fetch.</div>
        ${S.runError ? `<div class="banner error" style="margin-top:14px">${esc(S.runError)}</div>` : ''}
      </div>
      ${hasResults() ? '' : `<div class="dashed-card">
        <div class="kicker" style="margin-bottom:8px">EMPTY STATE · NOTHING RUN YET</div>
        <div style="font-family:var(--serif);font-size:16px;line-height:1.45;margin-bottom:8px">The results views stay visible but inert until a run finishes.</div>
        <div style="font-size:12.5px;line-height:1.55;color:var(--ink2)">Topics, Sentiment and Export are listed and explain what they will contain, so the shape of the output is legible before an hour of GPU time is spent.</div>
      </div>`}
    </div>
  </div>`;
}

function advancedPanel() {
  const st = S.settings;
  const picker = (field, opts) => opts.map(([id, note]) => `
    <button type="button" class="picker" data-field="${field}" data-value="${esc(id)}" aria-pressed="${st[field] === id}">
      <span class="id">${esc(shortModel(id))}</span><span class="note">${esc(note)}</span></button>`).join('');

  return `<div class="adv-body">
    <div style="padding-top:16px">
      <div style="display:flex;align-items:baseline;justify-content:space-between;margin-bottom:9px">
        <span class="field-label">Chunk size</span>
        <span style="font:400 12px var(--mono);color:var(--ink2)">${st.chunk_max_seconds} s ≈ ${Math.round(st.chunk_max_seconds * 2.6)} words</span>
      </div>
      <div class="seg-row">
        ${CHUNK_OPTS.map((c) => `<button type="button" class="seg" data-field="chunk_max_seconds" data-value="${c}" aria-pressed="${st.chunk_max_seconds === c}">${c} s</button>`).join('')}
      </div>
      <div class="hint" style="margin-top:8px">Shorter chunks find sharper topic boundaries; longer chunks give the sentiment model more context.</div>
    </div>
    <div class="picker-grid">
      <div>
        <div class="field-label" style="margin-bottom:8px">Embedding model</div>
        <div style="display:flex;flex-direction:column;gap:6px">${picker('embedding_model', EMBED_OPTS)}</div>
      </div>
      <div>
        <div class="field-label" style="margin-bottom:8px">Sentiment model</div>
        <div style="display:flex;flex-direction:column;gap:6px">${picker('sentiment_model', SENT_OPTS)}</div>
      </div>
    </div>
    <div class="switch-row">
      <button type="button" class="switch" role="switch" data-action="toggle-whisper" aria-checked="${st.use_whisper_fallback}"><span class="knob"></span></button>
      <div>
        <div class="field-label" style="margin-bottom:4px">Whisper audio fallback</div>
        <div style="font-size:12.5px;line-height:1.55;color:var(--ink2);max-width:52ch">When a video has no caption track,
          transcribe the audio locally. This recovers most skips — and it is
          <strong style="color:var(--warn)">much slower</strong>: roughly 6–10× the run time on this machine.</div>
      </div>
    </div>
    <div class="switch-row">
      <button type="button" class="switch" role="switch" data-action="toggle-people" aria-checked="${st.detect_people}"><span class="knob"></span></button>
      <div>
        <div class="field-label" style="margin-bottom:4px">Detect people &amp; entities</div>
        <div style="font-size:12.5px;line-height:1.55;color:var(--ink2);max-width:52ch">Named-entity recognition over each chunk, shown on the video page. Adds a pass over the transcript.</div>
      </div>
    </div>
  </div>`;
}

function parseUrls(raw) {
  const seen = new Set();
  const out = [];
  String(raw || '').split(/[\n,]/).forEach((line) => {
    const v = line.trim();
    if (v && !seen.has(v)) { seen.add(v); out.push(v); }
  });
  return out;
}
function countRawLines(raw) {
  return String(raw || '').split(/[\n,]/).map((l) => l.trim()).filter(Boolean).length;
}

// ── screen: progress ────────────────────────────────────────────────
function screenProgress() {
  const p = S.progress;
  if (!p) return `<div class="empty-note"><h2>No run started yet</h2>
    <p class="lede">Paste some URLs on the New run screen; progress appears here per stage and per URL.</p></div>`;

  const started = p.started_at ? new Date(p.started_at * 1000).toLocaleTimeString() : '—';
  const heading = p.status === 'done' ? 'Run complete'
    : p.status === 'failed' ? 'Run failed'
      : p.waiting ? 'Waiting for the pipeline'
        : `Reading ${p.counts.total} video${p.counts.total === 1 ? '' : 's'}`;

  return `<div class="prog-grid">
    <div style="display:flex;flex-direction:column;gap:20px">
      <div>
        <div class="kicker">RUN ${esc(p.status.toUpperCase())} · STARTED ${esc(started)}</div>
        <h1 style="margin:12px 0 8px">${esc(heading)}</h1>
        <p class="lede">The topic model is fitted once, over the whole batch, after every transcript is in —
          so topic IDs mean the same thing in every video.</p>
      </div>
      <div class="card card-pad">
        <div class="card-head">
          <h2>Pipeline</h2>
          <span class="mono-note">elapsed ${fmtT(p.elapsed_s)}</span>
        </div>
        <div style="display:flex;flex-direction:column;gap:16px">
          ${p.stages.map((s) => `<div class="stage" data-status="${s.status}">
            <div class="stage-head">
              <span class="stage-name"><span class="sdot"></span>${esc(s.name)}</span>
              <span class="mono-note">${esc(s.note)}</span>
            </div>
            <div class="bar"><span style="width:${Math.round(s.progress * 100)}%"></span></div>
          </div>`).join('')}
        </div>
        ${p.status === 'failed' ? `<div class="banner error" style="margin-top:20px"><strong>Run failed.</strong> ${esc(p.error)}</div>` : ''}
        ${p.waiting ? `<div class="banner" style="margin-top:20px">
          <strong>Queued.</strong> Another run is using the pipeline. One runs at a time — the topic model
          is shared, so overlapping runs would corrupt each other's results. This one starts automatically.</div>` : ''}
        ${p.status === 'running' && !p.waiting ? `<div class="banner" style="margin-top:20px">
          <strong>First run on this machine downloads model weights.</strong>
          Embedding and sentiment models are fetched once (several hundred MB) and cached;
          later runs start straight at the fetch stage.</div>` : ''}
        ${p.status === 'done' ? `<div style="margin-top:20px"><button type="button" class="btn-primary" data-go="batch">See batch overview</button></div>` : ''}
      </div>
      ${p.log.length ? `<details class="card card-pad"><summary class="mono-note" style="cursor:pointer">RUN LOG · ${p.log.length} lines</summary>
        <pre style="font:400 11.5px/1.7 var(--mono);color:var(--ink2);white-space:pre-wrap;margin:12px 0 0">${esc(p.log.join('\n'))}</pre></details>` : ''}
    </div>
    <div class="card" style="position:sticky;top:76px">
      <div style="display:flex;align-items:baseline;justify-content:space-between;padding:16px 18px;border-bottom:1px solid var(--line)">
        <h2 style="font-size:17px">Per-URL status</h2>
        <span class="mono-note">${p.counts.ok} ok · ${p.counts.skipped} skipped · ${p.counts.queued} queued</span>
      </div>
      <div class="url-list">
        ${p.urls.map((u) => `<div class="url-row" data-status="${u.status}">
          <span class="udot"></span>
          <span style="min-width:0">
            <span class="utitle">${esc(u.video_id || u.url)}</span>
            <span class="ustate">${esc(urlStateText(u))}</span>
          </span></div>`).join('')}
      </div>
    </div>
  </div>`;
}

function urlStateText(u) {
  if (u.status === 'analyzed') return u.reason || 'transcript ok';
  if (u.status === 'skipped') return 'skipped · ' + (u.reason || 'unknown reason');
  if (u.status === 'running') return 'working…';
  return 'queued';
}

// ── screen: batch overview ──────────────────────────────────────────
function screenBatch() {
  if (!hasResults()) return emptyResults('Batch overview', 'Videos analyzed, skips with reasons, batch sentiment, and the video list.');
  const R = S.results, T = R.totals;
  const stats = [
    ['VIDEOS ANALYZED', String(T.n_analyzed), `of ${T.n_submitted} submitted · ${T.n_skipped} skipped`],
    ['DURATION READ', (T.duration_s / 3600).toFixed(1) + ' h', 'transcript time, excluding skips'],
    ['CHUNKS', T.n_chunks.toLocaleString(), `${R.settings.chunk_max_seconds} s each · ${R.outlier.n_chunks} in the outlier bin`],
    ['TOPICS FOUND', String(T.n_topics), 'one shared model · plus outlier bin'],
    ['BATCH VALENCE', sig(T.mean_valence), 'mean of chunk scores, unweighted'],
  ];

  return `<div class="stack" style="max-width:1220px">
    <div style="display:flex;align-items:flex-end;justify-content:space-between;gap:24px;flex-wrap:wrap">
      <div>
        <div class="kicker">RUN ${esc(runClock())} · SESSION ONLY</div>
        <h1 style="font-size:38px;margin:12px 0 8px">Batch overview</h1>
        <p class="lede">${R.settings.chunk_max_seconds} s chunks · ${esc(shortModel(R.settings.embedding_model))} embeddings ·
          ${esc(shortModel(R.settings.sentiment_model))} sentiment · Whisper fallback ${R.settings.use_whisper_fallback ? 'on' : 'off'}.
          These settings are stamped on every export.</p>
      </div>
      <button type="button" class="btn-primary" data-go="export">Export this run</button>
    </div>

    <div class="stat-strip">
      ${stats.map(([label, v, sub]) => `<div class="stat">
        <div class="label">${esc(label)}</div><div class="v">${esc(v)}</div><div class="sub">${esc(sub)}</div></div>`).join('')}
    </div>

    <div class="two-col">
      <section class="card card-pad">
        <div class="card-head">
          <h2>Batch sentiment</h2>
          <span class="mono-note">n = ${T.n_chunks.toLocaleString()} chunks · −1 … +1</span>
        </div>
        ${histogram(T.histogram)}
        ${neutralNote(R)}
        <div style="display:flex;gap:22px;flex-wrap:wrap;padding-top:14px;border-top:1px solid var(--line);margin-top:14px">
          ${[['mean valence', sig(T.mean_valence), sentHex(T.mean_valence)],
            ['SD (controversy)', T.sd_valence.toFixed(2), 'var(--ink)'],
            ['negative chunks', pct(T.pct_negative, 0), 'var(--neg)'],
            ['positive chunks', pct(T.pct_positive, 0), 'var(--pos)']]
            .map(([k, v, c]) => `<div><div class="mono-note">${esc(k)}</div>
              <div style="font-family:var(--serif);font-weight:600;font-size:20px;margin-top:4px;color:${c}">${esc(v)}</div></div>`).join('')}
        </div>
      </section>

      <section class="card card-pad">
        <h2 style="margin-bottom:4px">${R.skipped.length} video${R.skipped.length === 1 ? '' : 's'} skipped</h2>
        <p style="margin:0 0 16px;font-size:13px;line-height:1.55;color:var(--ink2)">The batch ran anyway. Skipped videos are
          excluded from every statistic and named here so the denominator is never a mystery.</p>
        <div style="display:flex;flex-direction:column;gap:10px">
          ${R.skipped.length ? R.skipped.map((s) => `<div class="skip-card">
            <div class="t">${esc(s.video_id || 'Unrecognized URL')}</div>
            <div class="u">${esc(s.url)}</div>
            <div class="w">${esc(s.reason)}</div></div>`).join('')
            : '<div class="hint">Every submitted URL was analyzed.</div>'}
        </div>
      </section>
    </div>

    <section>
      <div class="section-head">
        <h2>Videos in this run</h2>
        <span class="mono-note">micro-throughline: topic lane + sentiment ribbon over full runtime</span>
      </div>
      <div class="card" style="overflow:hidden">
        <div class="video-table-head">
          <div>VIDEO</div><div>THROUGHLINE</div><div style="text-align:right">RUNTIME</div>
          <div style="text-align:right">MEAN VALENCE</div><div style="text-align:right">TOPICS</div>
        </div>
        ${R.videos.map((v) => `<button type="button" class="video-row" data-video="${esc(v.video_id)}" data-seek="0">
          <div style="min-width:0">
            <div class="vt">${esc(v.title)}</div>
            <div class="vs">${esc(v.channel || v.video_id)} · ${esc(topicInfo(v.topic_mix[0] ? v.topic_mix[0].topic_id : -1).label)}</div>
          </div>
          <div>${throughline(v, 'micro', {})}</div>
          <div class="num">${esc(fmtDur(v.duration_s))}</div>
          <div style="text-align:right">${valChip(v.mean_valence)}</div>
          <div class="num">${v.topic_mix.length}</div>
        </button>`).join('')}
      </div>
    </section>
  </div>`;
}

/* Models with no neutral class at all. Everything else gets described by what
   it actually did in this run — a 3-class model can simply produce no neutral
   chunks, and calling it binary on that basis would be a claim about the model
   we have not earned. */
const BINARY_MODELS = [
  'siebert/sentiment-roberta-large-english',
  'distilbert-base-uncased-finetuned-sst-2-english',
];

function neutralNote(R) {
  if (R.settings.model_produces_neutral) return '';
  const model = R.settings.sentiment_model;
  const reason = BINARY_MODELS.includes(model)
    ? `${esc(shortModel(model))} is a binary classifier — it has no neutral class, so every chunk is
       forced positive or negative.`
    : `No chunk in this run was scored neutral: ${esc(shortModel(model))} returned only positive and
       negative labels for this batch.`;
  return `<div class="hint" style="margin-top:10px">${reason}
    Values near zero mean low model confidence, not a neutral reading.
    ${BINARY_MODELS.includes(model) ? 'Pick a three-way model in Advanced settings if you need one.' : ''}</div>`;
}

function histogram(bins) {
  const W = 480, H = 160;
  const max = Math.max(1, ...bins.map((b) => b.count));
  const body = bins.map((b, i) => {
    const x = 14 + i * 42;
    const h = (b.count / max) * 104;
    return `<rect x="${x}" y="${118 - h}" width="34" height="${Math.max(1, h)}" fill="${sentHex(b.value)}" rx="1"/>
      <text x="${x + 17}" y="133" text-anchor="middle" font-size="10" font-family="var(--mono)" fill="var(--ink3)">${b.value.toFixed(1)}</text>
      <text x="${x + 17}" y="${118 - h - 5}" text-anchor="middle" font-size="9.5" font-family="var(--mono)" fill="var(--ink3)">${b.count}</text>`;
  }).join('');
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block;margin:18px 0 6px" role="img"
    aria-label="Histogram of chunk valence from minus one to plus one">
    ${body}
    <line x1="8" y1="118.5" x2="472" y2="118.5" stroke="var(--line2)"/>
    <text x="240" y="152" text-anchor="middle" font-size="10.5" font-family="var(--mono)" fill="var(--ink3)">chunk valence (−1 negative … +1 positive) · y = chunk count</text>
  </svg>`;
}

function emptyResults(title, what) {
  return `<div class="empty-note">
    <div class="kicker" style="margin-bottom:8px">EMPTY · NOTHING RUN YET</div>
    <h2>${esc(title)}</h2>
    <p class="lede">${esc(what)}</p>
    <p style="margin-top:16px"><button type="button" class="btn" data-go="run">Start a run</button></p>
  </div>`;
}

// ── screen: topics ──────────────────────────────────────────────────
function screenTopics() {
  if (!hasResults()) return emptyResults('Topics', 'The shared topic space: labels, keyword chips, prevalence, valence and representative excerpts.');
  const R = S.results;
  if (!R.topics.length) {
    return `<div class="empty-note"><h2>No topics were found</h2>
      <p class="lede">Every chunk landed in the outlier bin (${R.outlier.n_chunks} chunks). This usually means the batch
      is too small or too uniform for HDBSCAN to find structure. Try a shorter chunk size, or add more videos.</p></div>`;
  }
  const topic = currentTopic();
  const maxShare = Math.max(...R.topics.map((t) => t.share));

  return `<div class="stack">
    <div>
      <h1>Topics in this batch</h1>
      <p class="lede">One model fitted across all ${R.totals.n_analyzed} transcript${R.totals.n_analyzed === 1 ? '' : 's'}.
        Labels are generated from each topic's top terms — the keyword chips are the ground truth.</p>
    </div>

    <section class="card card-pad">
      <div class="card-head">
        <h2 style="font-size:17px">Topic share of all chunks</h2>
        <span class="mono-note">n = ${R.totals.n_chunks.toLocaleString()} chunks · outlier bin shown, not hidden</span>
      </div>
      <div class="share-bar">
        ${R.topics.map((t) => `<span title="${esc(t.label)}" style="width:${t.share * 100}%;background:${t.color}"></span>`).join('')}
        <span class="hatch" title="Outlier bin" style="width:${R.outlier.share * 100}%"></span>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:7px" class="mono-note">
        <span>0%</span><span>${R.topics.length} topics · outlier bin ${pct(R.outlier.share)} at the right end</span><span>100%</span>
      </div>
      <div class="legend-row">
        ${R.topics.slice(0, 6).map((t) => `<span class="legend-item">${swatch(t.topic_id, 12)}
          <span>${esc(t.label)}</span><span style="font-family:var(--mono);color:var(--ink3)">${pct(t.share)}</span></span>`).join('')}
      </div>
    </section>

    <div class="topic-grid">
      ${R.topics.map((t, i) => `<button type="button" class="topic-card ${i < 2 ? 'lead' : ''}"
        data-topic="${t.topic_id}" style="border-top-color:${t.color}">
        <div class="tl-head">
          ${swatch(t.topic_id, i < 2 ? 18 : 14)}
          <span style="display:flex;flex-direction:column;gap:3px">
            <span class="tname">${esc(t.label)}</span>
            <span class="tid">T${String(t.topic_id).padStart(2, '0')} · ${t.n_videos} video${t.n_videos === 1 ? '' : 's'}</span>
          </span>
        </div>
        <div class="kw-row">${t.keywords.slice(0, i < 2 ? 5 : 3).map((k) => `<span class="chip">${esc(k)}</span>`).join('')}</div>
        <div class="card-foot">
          <div>
            <div class="metric-row"><span>share of content</span><span class="v">${pct(t.share)}</span></div>
            <div class="thin-bar"><span style="width:${(t.share / maxShare) * 100}%;background:${t.color}"></span></div>
          </div>
          <div class="val-row">
            <span class="mono-note">valence ${sig(t.mean_valence)}</span>${valMini(t.mean_valence)}
          </div>
          <div style="display:flex;align-items:center;gap:7px" class="mono-note">
            <span class="ctrl-tag ${t.controversy === 'high' ? 'high' : ''}">controversy ${esc(t.controversy)}</span>
            <span>σ ${t.sd_valence.toFixed(2)}</span>
          </div>
        </div>
      </button>`).join('')}
      <div class="outlier-card">
        <div class="tl-head">
          <span class="hatch" style="width:18px;height:18px;flex:none;margin-top:2px;border-radius:2px"></span>
          <span class="tname" style="font-family:var(--serif);font-weight:600;font-size:16px">Outlier bin (topic −1)</span>
        </div>
        <div style="font-size:12.5px;line-height:1.55;color:var(--ink2)">Chunks the model refused to cluster: intros, ad reads,
          crosstalk, one-off asides. They stay in the denominator of every share on this page.</div>
        <div style="margin-top:auto">
          <div class="metric-row"><span>share of content</span><span class="v">${pct(R.outlier.share)}</span></div>
          <div class="thin-bar"><span class="hatch" style="width:${R.outlier.share * 100}%"></span></div>
          <button type="button" class="btn" style="margin-top:12px" data-go="outliers"
            ${R.outlier.n_chunks ? '' : 'disabled'}>Inspect ${R.outlier.n_chunks} chunks</button>
        </div>
      </div>
    </div>

    <section>
      <div class="section-head">
        <h2>${S.excerptMode === 'polarized' ? 'Most polarized' : 'Representative'} excerpts · ${esc(topic.label)}</h2>
        <div style="display:flex;align-items:center;gap:8px">
          <span class="mono-note">SHOW</span>
          ${[['representative', 'Most representative'], ['polarized', 'Most polarized']].map(([k, label]) =>
            `<button type="button" class="sort-btn" data-excerpt-mode="${k}" aria-pressed="${S.excerptMode === k}">${label}</button>`).join('')}
        </div>
      </div>
      <p class="mono-note" style="margin:-4px 0 12px">${S.excerptMode === 'polarized'
        ? 'Strongest feeling either way — these are the extremes, not the typical case.'
        : 'Highest topic-assignment probability — the chunks the model considers most typical of this topic.'}
        · play → video page, player cued to the timestamp</p>
      <div class="quote-grid">
        ${excerptsFor(topic).map((q) => `<div class="quote-card">
          <div style="display:flex;align-items:center;gap:8px">
            <span class="sent-swatch" style="background:${sentHex(q.valence)}"></span>
            <span class="mono-note">valence ${sig(q.valence)}</span>
            <span class="mono-note" style="margin-left:auto">${confidenceLabel(q)}</span>
          </div>
          <blockquote>“${esc(q.text)}”</blockquote>
          <div class="quote-foot">
            <div style="min-width:0">
              <div class="qt">${esc(q.video_title)}</div>
              <div class="qs">${esc(fmtT(q.start))} · ${esc(q.channel || q.video_id)}</div>
            </div>
            <button type="button" class="btn-play" data-video="${esc(q.video_id)}" data-seek="${q.start}">
              <span style="font-size:9px">▶</span><span>Play</span></button>
          </div>
        </div>`).join('')}
      </div>
    </section>
  </div>`;
}

// ── screen: outlier inspector ───────────────────────────────────────
/* Reachable from the outlier card on Topics. A plain browsable list: the point
   is that the chunks the model refused to cluster are inspectable, not hidden
   behind a percentage. */
function screenOutliers() {
  if (!hasResults()) return emptyResults('Outlier bin', 'The chunks the topic model refused to cluster, listed so you can read them.');
  const R = S.results;
  const rows = [];
  R.videos.forEach((v) => {
    v.chunks.forEach((c) => {
      if (c.topic_id === -1) rows.push({ chunk: c, video: v });
    });
  });
  const filtered = S.outlierVideo === 'all'
    ? rows
    : rows.filter((r) => r.video.video_id === S.outlierVideo);
  const withOutliers = R.videos.filter((v) => v.chunks.some((c) => c.topic_id === -1));

  return `<div class="stack">
    <div>
      <div class="kicker">TOPIC −1 · ${R.outlier.n_chunks} CHUNKS · ${pct(R.outlier.share)} OF THE BATCH</div>
      <h1 style="margin:12px 0 8px">The outlier bin</h1>
      <p class="lede">Chunks HDBSCAN would not put in any cluster — intros, ad reads, crosstalk, one-off asides.
        They stay in the denominator of every share in this run. Read them here to judge whether the model
        is discarding something you care about.</p>
    </div>

    <div class="card" style="overflow:hidden">
      <div style="display:flex;align-items:center;gap:10px;padding:14px 20px;border-bottom:1px solid var(--line);flex-wrap:wrap">
        <span class="mono-note">VIDEO</span>
        <button type="button" class="sort-btn" data-outlier-video="all" aria-pressed="${S.outlierVideo === 'all'}">All (${rows.length})</button>
        ${withOutliers.map((v) => {
          const n = v.chunks.filter((c) => c.topic_id === -1).length;
          const short = v.title.length > 28 ? v.title.slice(0, 27) + '…' : v.title;
          return `<button type="button" class="sort-btn" data-outlier-video="${esc(v.video_id)}"
            aria-pressed="${S.outlierVideo === v.video_id}" title="${esc(v.title)}">${esc(short)} (${n})</button>`;
        }).join('')}
      </div>
      <div class="outlier-head">
        <div>TIMESTAMP</div><div>VIDEO</div><div>CHUNK</div><div style="text-align:right">VALENCE</div>
      </div>
      <div style="max-height:640px;overflow:auto">
        ${filtered.length ? filtered.map(({ chunk, video }) => `<button type="button" class="outlier-row"
          data-video="${esc(video.video_id)}" data-seek="${chunk.start}">
          <div class="stamp mono-note">${fmtT(chunk.start)}</div>
          <div class="ov">${esc(video.title)}</div>
          <div class="otext">${esc(chunk.text)}</div>
          <div style="display:flex;align-items:center;gap:7px;justify-content:flex-end">
            ${valMini(chunk.valence)}<span class="mono-note">${sig(chunk.valence)}</span></div>
        </button>`).join('')
          : '<div style="padding:20px" class="hint">No outlier chunks in this video.</div>'}
      </div>
    </div>
    <p class="mono-note">click a row → video page, player cued to that moment</p>
  </div>`;
}

// ── screen: topic sentiment ─────────────────────────────────────────
function screenTopicSentiment() {
  if (!hasResults()) return emptyResults('Topic sentiment', 'Per-topic valence with controversy range, a sortable table, and the video × topic heatmap.');
  const R = S.results;
  if (!R.topics.length) return `<div class="empty-note"><h2>No topics to compare</h2><p class="lede">Every chunk landed in the outlier bin.</p></div>`;

  const ordered = R.topics.slice().sort((a, b) => (
    S.sort === 'share' ? b.share - a.share
      : S.sort === 'mean' ? a.mean_valence - b.mean_valence
        : b.sd_valence - a.sd_valence
  ));

  return `<div class="stack">
    <div>
      <h1>Topic sentiment</h1>
      <p class="lede">How each topic is talked about, and how much the batch disagrees about it.
        Chart and table are the same numbers in the same order — sorting one sorts both.</p>
    </div>

    <section class="card card-pad">
      <div class="card-head">
        <h2 style="font-size:18px">Mean valence by topic, with controversy range</h2>
        <span class="mono-note">dot = mean · bar = ±1 SD · n per topic in table</span>
      </div>
      ${divergingChart(ordered)}
      ${neutralNote(R)}
    </section>

    <section class="card" style="overflow:hidden">
      <div style="display:flex;align-items:center;gap:14px;padding:14px 20px;border-bottom:1px solid var(--line);flex-wrap:wrap">
        <h2 style="font-size:18px;flex:1">Detail table</h2>
        <span class="mono-note">SORT</span>
        ${[['share', 'Share'], ['mean', 'Valence'], ['sd', 'Controversy']].map(([k, label]) =>
          `<button type="button" class="sort-btn" data-sort="${k}" aria-pressed="${S.sort === k}">${label}</button>`).join('')}
      </div>
      <div style="overflow:auto">
        <div class="sent-table-head">
          <div>TOPIC</div><div style="text-align:right">CHUNKS</div><div style="text-align:right">SHARE</div>
          <div style="text-align:right">MEAN</div><div style="text-align:right">SD</div><div style="text-align:right">CONTROVERSY</div>
        </div>
        ${ordered.map((t) => `<button type="button" class="sent-row" data-topic="${t.topic_id}">
          <div style="display:flex;align-items:center;gap:9px;min-width:0">${swatch(t.topic_id, 13)}
            <span class="tlabel">${esc(t.label)}</span></div>
          <div style="text-align:right">${t.n_chunks.toLocaleString()}</div>
          <div style="text-align:right">${pct(t.share)}</div>
          <div style="text-align:right;color:${sentHex(t.mean_valence)};font-weight:500">${sig(t.mean_valence)}</div>
          <div style="text-align:right">${t.sd_valence.toFixed(2)}</div>
          <div style="display:flex;justify-content:flex-end;align-items:center;gap:7px">
            <span class="ctrl-mini ${t.controversy === 'high' ? 'high' : ''}"><span style="width:${Math.min(100, (t.sd_valence / 0.7) * 100)}%"></span></span>
            <span class="mono-note">${esc(t.controversy === 'moderate' ? 'mod' : t.controversy)}</span>
          </div>
        </button>`).join('')}
      </div>
    </section>

    <section class="card card-pad">
      <div class="card-head">
        <h2 style="font-size:18px">Video × topic heatmap</h2>
        <span class="mono-note">cell = share of that video's chunks</span>
      </div>
      <p style="margin:6px 0 18px;font-size:13px;line-height:1.55;color:var(--ink2);max-width:70ch">Rows are videos, columns are topics.
        Comparable because the topic space is shared. A dense column is a topic the whole batch keeps returning to;
        a lone dark cell is a video that owns a subject by itself.</p>
      ${heatmap(R.topics)}
    </section>
  </div>`;
}

function divergingChart(ordered) {
  const rowH = 30, cw = 1000, lw = 260, aw = cw - lw - 40;
  const chartH = ordered.length * rowH + 46;
  const cx = lw + aw / 2;
  const rows = ordered.map((t, i) => {
    const y = i * rowH + 22;
    const lo = clamp(t.mean_valence - t.sd_valence, -1, 1);
    const hi = clamp(t.mean_valence + t.sd_valence, -1, 1);
    const label = t.label.length > 30 ? t.label.slice(0, 29) + '…' : t.label;
    return `<g>
      <rect x="6" y="${y - 6}" width="11" height="11" rx="2" fill="${t.color}"/>
      <text x="24" y="${y + 5}" font-size="13" font-family="var(--sans)" fill="var(--ink)">${esc(label)}</text>
      <line x1="${cx + lo * (aw / 2)}" y1="${y}" x2="${cx + hi * (aw / 2)}" y2="${y}"
        stroke="${sentHex(t.mean_valence)}" stroke-opacity=".3" stroke-width="11" stroke-linecap="round"/>
      <circle cx="${cx + clamp(t.mean_valence, -1, 1) * (aw / 2)}" cy="${y}" r="5.5" fill="${sentHex(t.mean_valence)}"/>
      <text x="${cw - 6}" y="${y + 5}" text-anchor="end" font-size="12" font-family="var(--mono)" fill="${sentHex(t.mean_valence)}">${sig(t.mean_valence)}</text>
    </g>`;
  }).join('');
  const axisY = ordered.length * rowH + 16;
  return `<svg viewBox="0 0 ${cw} ${chartH}" style="width:100%;height:auto;display:block" role="img"
    aria-label="Mean valence per topic with plus or minus one standard deviation">
    ${rows}
    <line x1="${cx}" y1="6" x2="${cx}" y2="${axisY}" stroke="var(--line2)"/>
    <line x1="${lw}" y1="${axisY}" x2="${lw + aw}" y2="${axisY}" stroke="var(--line)"/>
    <text x="${lw}" y="${axisY + 18}" font-size="11" font-family="var(--mono)" fill="var(--ink3)">−1.0 negative</text>
    <text x="${cx}" y="${axisY + 18}" text-anchor="middle" font-size="11" font-family="var(--mono)" fill="var(--ink3)">0 neutral</text>
    <text x="${lw + aw}" y="${axisY + 18}" text-anchor="end" font-size="11" font-family="var(--mono)" fill="var(--ink3)">+1.0 positive</text>
  </svg>`;
}

function heatmap(topics) {
  const R = S.results;
  const cellW = 58, labelW = 250, rowH = 26;
  const MAX_LABEL = 22;
  const labels = topics.map((t) => (t.label.length > MAX_LABEL ? t.label.slice(0, MAX_LABEL - 1) + '…' : t.label));
  // Headers are rotated 52°, so their vertical rise is length × sin(52°).
  // Reserve exactly that much headroom instead of clipping them.
  const longest = Math.max(0, ...labels.map((l) => l.length)) * 6.2;
  const headroom = Math.ceil(longest * 0.79) + 22;
  const swatchY = headroom + 6;
  const firstRowY = headroom + 30;
  const H = R.videos.length * rowH + firstRowY + 16;
  const W = labelW + topics.length * cellW + 20;
  const head = topics.map((t, ci) => {
    const cx = labelW + ci * cellW + cellW / 2;
    return `<rect x="${cx - 6}" y="${swatchY}" width="12" height="12" rx="2" fill="${t.color}"/>
      <text x="${cx}" y="${headroom}" text-anchor="start" font-size="11.5" font-family="var(--sans)" fill="var(--ink2)"
        transform="rotate(-52 ${cx} ${headroom})">${esc(labels[ci])}</text>`;
  }).join('');
  const rows = R.videos.map((v, ri) => {
    const y = firstRowY + ri * rowH;
    const shares = new Map(v.topic_mix.map((m) => [m.topic_id, m.share]));
    const cells = topics.map((t, ci) => {
      const s = shares.get(t.topic_id) || 0;
      const x = labelW + ci * cellW + 3;
      const text = s ? `<text x="${x + (cellW - 6) / 2}" y="${y + 15}" text-anchor="middle" font-size="10.5"
        font-family="var(--mono)" fill="${s > 0.45 ? '#FFFFFF' : 'var(--ink2)'}">${Math.round(s * 100)}</text>` : '';
      return `<rect x="${x}" y="${y}" width="${cellW - 6}" height="22" rx="2" fill="${s ? t.color : 'var(--sunk)'}"
        fill-opacity="${s ? 0.18 + s * 0.82 : 1}" stroke="var(--line)" stroke-width="0.5"/>${text}`;
    }).join('');
    const title = v.title.length > 34 ? v.title.slice(0, 33) + '…' : v.title;
    return `<g data-video="${esc(v.video_id)}" data-seek="0" style="cursor:pointer">
      <text x="${labelW - 12}" y="${y + 15}" text-anchor="end" font-size="12" font-family="var(--sans)" fill="var(--ink)">${esc(title)}</text>
      ${cells}</g>`;
  }).join('');
  // max-width stops a small batch's heatmap from being upscaled to the
  // container width, which would blow the label type up with it.
  return `<div style="overflow-x:auto"><svg viewBox="0 0 ${W} ${H}"
    style="width:100%;min-width:${labelW + topics.length * cellW}px;max-width:${W}px;height:auto;display:block">
    ${head}${rows}</svg>
    <div class="mono-note" style="margin-top:10px">cell value = % of that video's chunks · blank = topic absent ·
      n = ${R.videos.length} video${R.videos.length === 1 ? '' : 's'} × ${topics.length} topics</div></div>`;
}

// ── screen: video page ──────────────────────────────────────────────
function screenVideo() {
  if (!hasResults()) return emptyResults('Video page', 'The player with the full Throughline beneath it, transcript, topic mix and entities.');
  const v = currentVideo();
  if (!v) return emptyResults('Video page', 'No video selected.');
  const single = S.results.videos.length === 1;
  const activeIdx = activeChunkIndex(v);

  return `<div class="video-grid">
    <div class="video-main">
      <div>
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap" class="kicker">
          <span>${esc(v.video_id)}</span><span style="color:var(--line2)">·</span><span>${esc(v.channel || 'unknown channel')}</span>
          ${single ? '<span style="letter-spacing:0;text-transform:none;border:1px solid var(--line);border-radius:2px;padding:3px 6px;color:var(--ink2)">single-video run</span>' : ''}
        </div>
        <h1 style="font-size:29px;line-height:1.15;margin:10px 0 0">${esc(v.title)}</h1>
      </div>

      <div class="card" style="overflow:hidden">
        <div class="player-wrap" id="player-wrap">
          <div class="player-fallback" id="player-fallback">
            <div style="letter-spacing:.14em">YOUTUBE PLAYER</div>
            <div>${esc(v.video_id)}</div>
            <div style="max-width:40ch;line-height:1.6">Loading the embed. If this machine is offline the throughline below still
              works — the playhead runs on an internal clock.</div>
          </div>
        </div>
        <div class="player-bar">
          <button type="button" class="btn-round" data-action="toggle-play" aria-label="${S.playing ? 'Pause' : 'Play'}">${S.playing ? '❚❚' : '▶'}</button>
          <span class="clock" id="clock">${fmtT(S.t)} / ${fmtT(v.duration_s)}</span>
          <div style="flex:1"></div>
          <span class="mono-note">hover for the excerpt · click anywhere to seek</span>
        </div>
        <div class="tl-full" id="tl-full">${throughline(v, 'full', { interactive: true })}</div>
        <div class="legend-btns">
          ${v.topic_mix.map((m) => `<button type="button" class="legend-btn" data-dim="${m.topic_id}" aria-pressed="${S.dim === m.topic_id}">
            ${swatch(m.topic_id, 12)}<span>${esc(topicInfo(m.topic_id).label)}</span>
            <span style="color:var(--ink3);font-family:var(--mono)">${pct(m.share, 0)}</span></button>`).join('')}
        </div>
      </div>

      <section class="card">
        <div style="display:flex;align-items:baseline;justify-content:space-between;padding:18px 22px 12px;border-bottom:1px solid var(--line)">
          <h2 style="font-size:18px">Transcript</h2>
          <span style="display:flex;align-items:center;gap:10px">
            <button type="button" class="follow-chip" id="follow-chip" data-action="resume-follow"
              ${S.followTranscript ? 'hidden' : ''}>↧ follow playhead</button>
            <span class="mono-note">${S.results.settings.chunk_max_seconds} s chunks · left rule = topic (dotted = weak assignment)</span>
          </span>
        </div>
        <div class="transcript" id="transcript">
          ${v.chunks.map((c) => `<button type="button"
            class="t-row ${c.index === activeIdx ? 'active' : ''} ${isLowConfidence(c) ? 'weak' : ''}"
            data-seek="${c.start}" data-video="${esc(v.video_id)}" data-chunk="${c.index}"
            title="${esc(topicInfo(c.topic_id).label)} · ${esc(confidenceLabel(c))}"
            style="border-left-color:${topicInfo(c.topic_id).color}">
            <span class="stamp">${fmtT(c.start)}</span>
            <span class="txt">${esc(c.text)}</span>
            <span class="sent">${valMini(c.valence)}<span class="mono-note">${sig(c.valence)}</span></span>
          </button>`).join('')}
        </div>
      </section>
    </div>

    <div class="side">
      <section>
        <div class="kicker-sm">METADATA</div>
        <div style="display:flex;flex-direction:column;gap:8px">
          <div class="summary-row"><span>url</span>
            <a href="${esc(v.url)}" target="_blank" rel="noopener"
               style="font-family:var(--mono);text-align:right;word-break:break-all">${esc(v.video_id)}</a></div>
          ${[['channel', v.channel || '—'], ['published', (v.published || '—').slice(0, 10)],
            ['runtime', fmtDur(v.duration_s)], [`chunks (${S.results.settings.chunk_max_seconds} s)`, String(v.n_chunks)],
            ['mean valence', sig(v.mean_valence)], ['topics present', `${v.topic_mix.length} of ${S.results.topics.length}`]]
            .map(([k, val]) => `<div class="summary-row"><span>${esc(k)}</span><span>${esc(val)}</span></div>`).join('')}
        </div>
      </section>

      <section>
        <div class="kicker-sm">TOPIC MIX · SHARE OF RUNTIME</div>
        ${stackBar(v.topic_mix, 18)}
        <div style="display:flex;flex-direction:column;gap:10px;margin-top:14px">
          ${v.topic_mix.map((m) => `<div class="mix-row">${swatch(m.topic_id, 13)}
            <button type="button" data-topic="${m.topic_id}">${esc(topicInfo(m.topic_id).label)}</button>
            <span style="text-align:right;font:400 12px var(--mono);color:var(--ink2)">${pct(m.share, 0)}</span></div>`).join('')}
        </div>
      </section>

      <section>
        <div class="kicker-sm" style="margin-bottom:6px">SENTIMENT BY TOPIC · THIS VIDEO</div>
        <div class="mono-note" style="margin-bottom:14px">mean ± SD, −1 … +1</div>
        <div style="display:flex;flex-direction:column;gap:12px">
          ${v.topic_mix.map((m) => videoSentBar(m)).join('')}
        </div>
      </section>

      <section>
        <div class="kicker-sm" style="margin-bottom:4px">PEOPLE &amp; ENTITIES MENTIONED</div>
        <div class="hint" style="margin-bottom:12px">Extracted from the transcript · count = mentions</div>
        <div style="display:flex;flex-wrap:wrap;gap:6px">
          ${v.entities.length ? v.entities.map((e) => `<span class="pill"><span style="font-family:var(--sans);font-size:12px">${esc(e.name)}</span><span>${e.count}</span></span>`).join('')
            : '<span class="hint">No person entities detected in this transcript.</span>'}
        </div>
      </section>
    </div>
  </div>`;
}

function videoSentBar(m) {
  const w = 170;
  const val = clamp(m.mean_valence, -1, 1);
  const lo = clamp(val - m.sd_valence, -1, 1), hi = clamp(val + m.sd_valence, -1, 1);
  return `<div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">
      ${swatch(m.topic_id, 11)}
      <span style="font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(topicInfo(m.topic_id).label)}</span>
      <span style="margin-left:auto;font:500 11.5px var(--mono);color:${sentHex(val)}">${sig(val)}</span>
    </div>
    <svg viewBox="0 0 ${w} 16" style="width:100%;height:16px;display:block" aria-hidden="true">
      <line x1="0" y1="8" x2="${w}" y2="8" stroke="var(--line)"/>
      <line x1="${w / 2}" y1="1" x2="${w / 2}" y2="15" stroke="var(--line2)"/>
      <line x1="${w / 2 + lo * (w / 2)}" y1="8" x2="${w / 2 + hi * (w / 2)}" y2="8" stroke="${sentHex(val)}" stroke-opacity=".35" stroke-width="6"/>
      <circle cx="${w / 2 + val * (w / 2)}" cy="8" r="4" fill="${sentHex(val)}"/>
    </svg></div>`;
}

function activeChunkIndex(v) {
  const c = v.chunks.find((x) => S.t >= x.start && S.t < x.end);
  return c ? c.index : (v.chunks[0] ? v.chunks[0].index : -1);
}

// ── screen: export ──────────────────────────────────────────────────
const SCHEMAS = {
  combined: [
    ['video_id', 'string', 'YouTube ID as submitted'],
    ['video_title', 'string', 'Title at fetch time'],
    ['chunk_index', 'int', '0-based, in playback order'],
    ['start_s', 'float', 'Chunk start in seconds — the seek target'],
    ['end_s', 'float', 'Chunk end in seconds'],
    ['text', 'string', 'Transcript text, quoted and escaped'],
    ['topic_id', 'int', '−1 = outlier bin'],
    ['topic_label', 'string', 'Generated from the topic’s top terms'],
    ['topic_prob', 'float', 'Assignment confidence, 0…1 — blank if reassigned'],
    ['topic_reassigned', 'int', '1 = moved out of the outlier bin, so no confidence exists'],
    ['valence', 'float', 'Sentiment, −1 … +1 (neutral is exactly 0)'],
    ['sentiment_label', 'string', 'Raw model label'],
    ['sentiment_score', 'float', 'Raw model confidence, 0…1'],
  ],
  video: [
    ['video_id', 'string', 'YouTube ID as submitted'],
    ['video_title', 'string', 'Title at fetch time'],
    ['channel', 'string', 'Uploader'],
    ['url', 'string', 'The URL you submitted'],
    ['duration_s', 'float', 'Runtime read'],
    ['n_chunks', 'int', 'Chunks after chunking'],
    ['mean_valence', 'float', 'Unweighted mean of chunk valence'],
    ['sd_valence', 'float', 'Controversy within this video'],
    ['topic_share_N', 'float', 'Share of this video’s chunks per topic'],
    ['outlier_share', 'float', 'Share in the outlier bin'],
  ],
  topic: [
    ['topic_id', 'int', '−1 = outlier bin'],
    ['topic_label', 'string', 'Generated from top terms'],
    ['keywords', 'string', 'Top terms, pipe-separated'],
    ['n_chunks', 'int', 'Chunks assigned across the batch'],
    ['share', 'float', 'Share of all chunks'],
    ['n_videos', 'int', 'Videos containing the topic'],
    ['mean_valence', 'float', 'Mean chunk valence'],
    ['sd_valence', 'float', 'Controversy measure'],
    ['controversy', 'string', 'low / moderate / high'],
  ],
};
const EXPORT_FILES = [
  ['combined', 'Every chunk of every video — the analysis unit'],
  ['video', 'One row per video: topic shares and valence'],
  ['topic', 'One row per topic: keywords, prevalence, valence, SD'],
];

function screenExport() {
  if (!hasResults()) return emptyResults('Export', 'Three CSVs — per chunk, per video, per topic — with the schema shown before you download.');
  const R = S.results;
  const kind = S.exportKind;
  const preview = previewRows(kind);

  return `<div class="stack">
    <div>
      <h1>Export</h1>
      <p class="lede">CSV, UTF-8, one header row. Every file carries the run settings in its filename so a figure
        can be traced back to the exact models it came from.</p>
    </div>
    <div class="export-grid">
      <section class="card card-pad">
        <div class="kicker" style="margin-bottom:12px">FILE</div>
        <div style="display:flex;flex-direction:column;gap:8px">
          ${EXPORT_FILES.map(([k, desc]) => `<button type="button" class="file-btn" data-export="${k}" aria-pressed="${kind === k}">
            <span class="f">${esc(exportFilename(k))}</span><span class="d">${esc(desc)}</span></button>`).join('')}
        </div>
        <div style="margin-top:20px;padding-top:16px;border-top:1px solid var(--line);display:flex;flex-direction:column;gap:8px">
          ${[['run', runClock()], ['chunk size', R.settings.chunk_max_seconds + ' s'],
            ['embeddings', shortModel(R.settings.embedding_model)], ['sentiment', shortModel(R.settings.sentiment_model)],
            ['encoding', 'UTF-8 · comma']].map(([k, v]) =>
            `<div class="summary-row"><span>${esc(k)}</span><span>${esc(v)}</span></div>`).join('')}
        </div>
        <a class="btn-primary btn-block" style="margin-top:18px;display:block;text-align:center;text-decoration:none"
          href="/api/runs/${esc(S.jobId)}/export/${esc(kind)}.csv" download>Download ${esc(kind)} CSV</a>
      </section>

      <div style="display:flex;flex-direction:column;gap:16px;min-width:0">
        <section class="card" style="overflow:hidden">
          <div style="display:flex;align-items:baseline;justify-content:space-between;padding:16px 20px;border-bottom:1px solid var(--line)">
            <h2 style="font-size:18px">Schema</h2>
            <span class="mono-note">${esc(schemaDims(kind))}</span>
          </div>
          <div class="schema-head"><div>COLUMN</div><div>TYPE</div><div>MEANING</div></div>
          ${SCHEMAS[kind].map(([c, t, d]) => `<div class="schema-row">
            <div class="c">${esc(c)}</div><div class="ty">${esc(t)}</div><div class="de">${esc(d)}</div></div>`).join('')}
        </section>
        <section class="card" style="overflow:hidden">
          <div style="display:flex;align-items:baseline;justify-content:space-between;padding:16px 20px;border-bottom:1px solid var(--line)">
            <h2 style="font-size:18px">First rows</h2>
            <span class="mono-note">exactly what lands in the file</span>
          </div>
          <div class="preview">
            <div class="prow head">${preview.cols.map((c) => `<div class="pcell">${esc(c)}</div>`).join('')}</div>
            ${preview.rows.map((r) => `<div class="prow">${r.map((c) => `<div class="pcell">${esc(c)}</div>`).join('')}</div>`).join('')}
          </div>
        </section>
      </div>
    </div>
  </div>`;
}

function exportFilename(kind) {
  const stamp = String(S.jobId || 'run').slice(0, 8);
  return `throughline_${kind}_${S.results.settings.chunk_max_seconds}s_${stamp}.csv`;
}
function schemaDims(kind) {
  const R = S.results;
  if (kind === 'combined') return `${R.totals.n_chunks.toLocaleString()} rows × ${SCHEMAS.combined.length} cols`;
  if (kind === 'video') return `${R.videos.length} rows × ${8 + R.topics.length * 2 + 1} cols`;
  return `${R.topics.length + 1} rows × ${SCHEMAS.topic.length} cols`;
}
function previewRows(kind) {
  const R = S.results;
  if (kind === 'combined') {
    const v = R.videos[0];
    return {
      cols: ['video_id', 'chunk_index', 'start_s', 'topic_id', 'topic_label', 'valence'],
      rows: v.chunks.slice(0, 7).map((c) => [v.video_id, c.index, c.start.toFixed(1), c.topic_id,
        topicInfo(c.topic_id).label, c.valence.toFixed(3)]),
    };
  }
  if (kind === 'video') {
    return {
      cols: ['video_id', 'duration_s', 'n_chunks', 'mean_valence', 'sd_valence', 'outlier_share'],
      rows: R.videos.slice(0, 7).map((v) => {
        const out = v.topic_mix.find((m) => m.topic_id === -1);
        return [v.video_id, v.duration_s.toFixed(1), v.n_chunks, v.mean_valence.toFixed(3),
          v.sd_valence.toFixed(3), (out ? out.share : 0).toFixed(3)];
      }),
    };
  }
  return {
    cols: ['topic_id', 'topic_label', 'n_chunks', 'share', 'mean_valence', 'sd_valence'],
    rows: R.topics.slice(0, 7).map((t) => [t.topic_id, t.label, t.n_chunks, t.share.toFixed(3),
      t.mean_valence.toFixed(3), t.sd_valence.toFixed(3)]),
  };
}

// ── screen: foundations ─────────────────────────────────────────────
function screenFoundations() {
  const R = S.results;
  const demo = R && R.videos.length ? R.videos[0] : null;
  const paletteTopics = R && R.topics.length
    ? R.topics.map((t, i) => [t.label, t.color, i])
    : ['Topic 0', 'Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5', 'Topic 6', 'Topic 7',
       'Topic 8', 'Topic 9', 'Topic 10', 'Topic 11'].map((l, i) => [l, PALETTE[i], i]);

  return `<div class="stack" style="gap:34px">
    <div>
      <h1>Pattern &amp; colour foundations</h1>
      <p class="lede">Two colour systems that never mix: topic identity is categorical, sentiment is diverging.
        Every topic colour carries a fill pattern, so hue is never the only encoding.</p>
    </div>

    <section>
      <h2 style="font-size:20px;margin-bottom:4px">The throughline, three sizes</h2>
      <p style="margin:0 0 16px;font-size:13.5px;color:var(--ink2);max-width:72ch">Topic lane on top (contiguous chunks,
        chapter-like), sentiment ribbon beneath (above centerline positive, below negative, height = |score|).</p>
      ${demo ? `<div style="display:flex;flex-direction:column;gap:16px">
        <div class="card card-pad">
          <div style="display:flex;justify-content:space-between;margin-bottom:12px" class="mono-note">
            <span>FULL · video page hero · 163px</span><span>hover → tooltip · click → seek · playhead mirrored live</span></div>
          ${throughline(demo, 'full', {})}
        </div>
        <div style="display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px">
          <div class="card card-pad"><div class="mono-note" style="margin-bottom:12px">MEDIUM · cards · 64px</div>${throughline(demo, 'med', {})}</div>
          <div class="card card-pad"><div class="mono-note" style="margin-bottom:12px">MICRO · every video list · 26px</div>
            <div style="display:flex;flex-direction:column;gap:10px">
              ${R.videos.slice(0, 4).map((v) => `<div style="display:grid;grid-template-columns:minmax(0,1fr) 150px;gap:12px;align-items:center">
                <span style="font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--ink2)">${esc(v.title)}</span>
                <span>${throughline(v, 'micro', {})}</span></div>`).join('')}
            </div></div>
        </div></div>`
        : '<div class="dashed-card"><div class="hint">Run an analysis to see the throughline drawn over real transcripts.</div></div>'}
    </section>

    <section style="display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);gap:20px;align-items:start">
      <div class="card card-pad">
        <h2 style="font-size:18px;margin-bottom:2px">Topic palette · categorical</h2>
        <div class="mono-note" style="margin-bottom:16px">12 hues, Tol-derived, deuteranopia-checked · fill pattern repeats every 4</div>
        <div class="palette-grid">
          ${paletteTopics.map(([label, hex, rank]) => `<div class="palette-item">
            ${paletteSwatch(hex, rank, 22)}
            <span style="min-width:0"><span class="pl">${esc(label)}</span><span class="ph">${esc(hex)}</span></span></div>`).join('')}
        </div>
      </div>
      <div class="card card-pad">
        <h2 style="font-size:18px;margin-bottom:2px">Sentiment scale · diverging</h2>
        <div class="mono-note" style="margin-bottom:16px">never used for topics · never red/green alone</div>
        <div class="scale-bar">${['#B4402F', '#CB7462', '#A29C91', '#4F9B90', '#14746F'].map((c) => `<span style="background:${c}"></span>`).join('')}</div>
        <div style="display:flex;justify-content:space-between;margin-top:6px" class="mono-note"><span>−1.0</span><span>0</span><span>+1.0</span></div>
        <div style="display:flex;flex-direction:column;gap:9px;margin-top:16px">
          ${[['#B4402F', '≤ −0.60', 'Strongly negative'], ['#CB7462', '−0.30', 'Negative'],
            ['#A29C91', '±0.08', 'Neutral / procedural'], ['#4F9B90', '+0.30', 'Positive'],
            ['#14746F', '≥ +0.60', 'Strongly positive']].map(([c, v, l]) => `<div style="display:flex;align-items:center;gap:10px">
            <span style="width:26px;height:14px;border-radius:2px;flex:none;background:${c}"></span>
            <span style="font:400 11.5px var(--mono);color:var(--ink2);width:52px">${esc(v)}</span>
            <span style="font-size:12.5px;color:var(--ink2)">${esc(l)}</span></div>`).join('')}
        </div>
        ${R ? neutralNote(R) : ''}
      </div>
    </section>

    <section>
      <h2 style="font-size:20px;margin-bottom:14px">Two core flows</h2>
      <div style="display:flex;flex-direction:column;gap:14px">
        ${FLOWS.map((f) => `<div class="card card-pad">
          <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:6px">
            <span class="kicker">${esc(f.kicker)}</span>
            <span style="font-family:var(--serif);font-weight:600;font-size:17px">${esc(f.title)}</span>
          </div>
          <div style="font-size:12.5px;color:var(--ink2);line-height:1.5;margin-bottom:16px;max-width:76ch">${esc(f.note)}</div>
          <div class="flow-steps">
            ${f.steps.map((s, i) => `<div style="display:flex;align-items:stretch">
              <button type="button" class="flow-step" data-go="${esc(s.go)}">
                <span style="font:500 10px var(--mono);color:var(--ink3)">${String(i + 1).padStart(2, '0')}</span>
                <span style="font-size:13px;font-weight:600;line-height:1.25">${esc(s.title)}</span>
                <span style="font-size:11.5px;line-height:1.45;color:var(--ink2)">${esc(s.note)}</span>
              </button>
              <span class="flow-arrow">${i < f.steps.length - 1 ? '→' : ''}</span></div>`).join('')}
          </div></div>`).join('')}
      </div>
    </section>
  </div>`;
}

const PALETTE = ['#332288', '#4477AA', '#88CCEE', '#117733', '#999933', '#DDCC77',
  '#CC6677', '#882255', '#AA4499', '#EE8866', '#8A5A2B', '#6E6E6E'];

function paletteSwatch(hex, rank, size) {
  const pre = 'pl' + (patternSeq++);
  return `<svg width="${size}" height="${size}" viewBox="0 0 6 6" style="display:block;border-radius:2px;flex:none">
    <defs>${patternMarkup(pre + '-p' + rank, hex, rank % 4)}</defs>
    <rect width="6" height="6" fill="${fillFor(pre, rank, hex)}"/></svg>`;
}

const FLOWS = [
  {
    kicker: 'FLOW A', title: 'The evidence loop — statistic to watchable moment',
    note: 'Every number in the app is one click from the seconds of video that produced it. This is the path researchers take when a finding needs checking before it goes in a paper.',
    steps: [
      { title: 'Paste URLs', note: 'Mixed URL forms welcome', go: 'run' },
      { title: 'Run progress', note: 'Stages + per-URL status', go: 'prog' },
      { title: 'Batch overview', note: 'Analyzed and skipped, named', go: 'batch' },
      { title: 'Topics', note: 'Excerpt quote cards per topic', go: 'topics' },
      { title: 'Video at t', note: 'Player cued; cursor on the throughline', go: 'video' },
    ],
  },
  {
    kicker: 'FLOW B', title: 'Single video — skip the batch ceremony',
    note: 'When n = 1 the batch views add nothing, so the run lands directly on the video page. Topics and Export stay reachable, scoped to that one transcript.',
    steps: [
      { title: 'One URL', note: 'Paste, hit Run', go: 'run' },
      { title: 'Progress', note: 'Same stages', go: 'prog' },
      { title: 'Video page', note: 'Lands here, no overview stop', go: 'video' },
      { title: 'Export', note: 'Chunk CSV for that video', go: 'export' },
    ],
  },
];

// ── render ──────────────────────────────────────────────────────────
const SCREENS = {
  run: screenRun, prog: screenProgress, batch: screenBatch, topics: screenTopics,
  tsent: screenTopicSentiment, video: screenVideo, export: screenExport,
  outliers: screenOutliers, found: screenFoundations,
};

function render() {
  document.documentElement.setAttribute('data-theme', S.theme);
  renderChrome();
  $('#screen').innerHTML = (SCREENS[S.screen] || screenRun)();
  if (S.screen === 'video') {
    mountPlayer();
    updatePlayhead();
    bindThroughline();
    bindTranscriptFollow();
  } else {
    hideTooltip();
  }
}

// ── throughline interaction ─────────────────────────────────────────
function bindThroughline() {
  const svg = document.querySelector('[data-tl-interactive]');
  if (!svg) return;
  const v = currentVideo();

  const timeAt = (event) => {
    const rect = svg.getBoundingClientRect();
    return clamp((event.clientX - rect.left) / rect.width, 0, 1) * v.duration_s;
  };

  svg.addEventListener('mousemove', (e) => {
    const t = timeAt(e);
    const chunk = v.chunks.find((c) => t >= c.start && t < c.end) || v.chunks[v.chunks.length - 1];
    if (!chunk) return;
    showTooltip(chunk, e);
    const hover = svg.querySelector('.tl-hover');
    if (hover) {
      const x = (t / v.duration_s) * 1000;
      hover.setAttribute('x1', x);
      hover.setAttribute('x2', x);
    }
  });
  svg.addEventListener('mouseleave', () => {
    hideTooltip();
    const hover = svg.querySelector('.tl-hover');
    if (hover) { hover.setAttribute('x1', -10); hover.setAttribute('x2', -10); }
  });
  svg.addEventListener('click', (e) => seek(timeAt(e)));
  svg.addEventListener('keydown', (e) => {
    const step = e.shiftKey ? 30 : 5;
    if (e.key === 'ArrowRight') { seek(S.t + step); e.preventDefault(); }
    if (e.key === 'ArrowLeft') { seek(S.t - step); e.preventDefault(); }
  });
}

function showTooltip(chunk, event) {
  const tip = $('#tooltip');
  const t = topicInfo(chunk.topic_id);
  tip.innerHTML = `<div class="head">${swatch(chunk.topic_id, 12)}
      <span class="lbl">${esc(t.label)}</span><span class="stamp">${fmtT(chunk.start)}</span></div>
    <div class="body">“${esc(chunk.text.slice(0, 240))}${chunk.text.length > 240 ? '…' : ''}”</div>
    <div class="foot">${valMini(chunk.valence)}
      <span style="font:500 11.5px var(--mono);color:${sentHex(chunk.valence)}">valence ${sig(chunk.valence)}</span>
      <span style="margin-left:auto" class="mono-note">click to seek</span></div>
    <div class="mono-note" style="margin-top:7px${isLowConfidence(chunk) ? ';color:var(--warn)' : ''}">${confidenceLabel(chunk)}</div>`;
  tip.hidden = false;
  const rect = tip.getBoundingClientRect();
  tip.style.left = clamp(event.clientX - rect.width / 2, 12, window.innerWidth - rect.width - 12) + 'px';
  tip.style.top = Math.max(12, event.clientY - rect.height - 16) + 'px';
}
function hideTooltip() { $('#tooltip').hidden = true; }

function updatePlayhead() {
  const v = currentVideo();
  if (!v) return;
  const head = document.querySelector('.tl-playhead');
  if (head) head.setAttribute('transform', `translate(${(clamp(S.t, 0, v.duration_s) / v.duration_s) * 1000},0)`);
  const clock = $('#clock');
  if (clock) clock.textContent = fmtT(S.t) + ' / ' + fmtT(v.duration_s);

  const active = activeChunkIndex(v);
  let activeRow = null;
  document.querySelectorAll('.t-row').forEach((row) => {
    const on = Number(row.dataset.chunk) === active;
    // Highlighting is unconditional: it continues whether or not the reader
    // has paused auto-follow.
    row.classList.toggle('active', on);
    if (on) activeRow = row;
  });
  if (activeRow && shouldFollow({ playing: S.playing, followEnabled: S.followTranscript })) {
    followRow(activeRow);
  }
}

/* Scrolls the transcript's own container and nothing else.

   The previous implementation called row.scrollIntoView(), which scrolls every
   scrollable ancestor up to the document — so playback dragged the page down,
   and did it again on the next tick whenever the reader scrolled back. */
function followRow(row) {
  const container = $('#transcript');
  if (!container) return;
  const target = followScrollTop({
    rowTop: row.offsetTop - container.offsetTop,
    rowHeight: row.offsetHeight,
    scrollTop: container.scrollTop,
    viewHeight: container.clientHeight,
    contentHeight: container.scrollHeight,
    margin: 12,
  });
  if (target === null) return;
  // Mark the scroll as ours so the container's scroll handler does not read
  // it back as the reader taking over.
  programmaticScroll = true;
  container.scrollTop = target;
  window.clearTimeout(programmaticScrollTimer);
  programmaticScrollTimer = window.setTimeout(() => { programmaticScroll = false; }, 120);
}

let programmaticScroll = false;
let programmaticScrollTimer = 0;

function setFollow(on) {
  if (S.followTranscript === on) return;
  S.followTranscript = on;
  const chip = $('#follow-chip');
  if (chip) chip.hidden = followChipHidden({ followEnabled: S.followTranscript });
}

/* A manual scroll inside the transcript hands control to the reader. */
function bindTranscriptFollow() {
  const container = $('#transcript');
  if (!container) return;
  container.addEventListener('scroll', () => {
    if (isReaderScroll({ programmatic: programmaticScroll })) setFollow(false);
  }, { passive: true });
}

function seek(t) {
  const v = currentVideo();
  if (!v) return;
  S.t = clamp(t, 0, v.duration_s);
  if (player && player.seekTo) {
    try { player.seekTo(S.t, true); } catch (_) { /* player not ready */ }
  }
  updatePlayhead();
}

// ── YouTube player (graceful when offline) ──────────────────────────
let player = null;
let playerVideoId = null;
let ytLoading = false;

/* getIframe() throws on a destroyed player, so probing it must not be the
   thing that breaks a re-render. */
function safeIframe(instance) {
  try {
    const el = instance.getIframe();
    return el && el.isConnected !== false ? el : null;
  } catch (_) {
    return null;
  }
}

function loadYT() {
  if (window.YT && window.YT.Player) return Promise.resolve(true);
  if (ytLoading) return ytLoading;
  ytLoading = new Promise((resolve) => {
    const script = document.createElement('script');
    script.src = 'https://www.youtube.com/iframe_api';
    script.onerror = () => resolve(false);
    window.onYouTubeIframeAPIReady = () => resolve(true);
    document.head.appendChild(script);
    setTimeout(() => resolve(!!(window.YT && window.YT.Player)), 6000);
  });
  return ytLoading;
}

function removeFallback() {
  const fb = $('#player-fallback');
  if (fb) fb.remove();
}

/* render() replaces #screen.innerHTML, which detaches the existing iframe. Any
   re-render of the video page — a theme toggle, a legend click, the excerpt
   toggle — therefore has to either re-adopt that iframe or destroy it. Leaving
   it behind orphans a player that keeps its own timers and network activity. */
async function mountPlayer() {
  const v = currentVideo();
  const wrap = $('#player-wrap');
  if (!v || !wrap) return;

  const iframe = player && player.getIframe ? safeIframe(player) : null;
  if (player && playerVideoId === v.video_id && iframe) {
    // Same video, still-valid player: re-adopt the iframe into the new DOM.
    wrap.appendChild(iframe);
    removeFallback();
    return;
  }
  if (player) {
    // Different video, or a player whose iframe is gone: tear it down before
    // building another, or its timers outlive it.
    try { player.destroy(); } catch (_) { /* already gone */ }
    player = null;
    playerVideoId = null;
  }

  const ok = await loadYT();
  if (!ok || !window.YT || !window.YT.Player) return; // fallback text stays
  const host = document.createElement('div');
  wrap.appendChild(host);
  player = new window.YT.Player(host, {
    videoId: v.video_id,
    playerVars: { rel: 0, modestbranding: 1, start: Math.floor(S.t) },
    events: {
      onReady: () => {
        playerVideoId = v.video_id;
        removeFallback();
        if (S.t) player.seekTo(S.t, true);
      },
      onStateChange: (e) => {
        S.playing = e.data === window.YT.PlayerState.PLAYING;
        const btn = document.querySelector('[data-action="toggle-play"]');
        if (btn) btn.textContent = S.playing ? '❚❚' : '▶';
      },
    },
  });
}

setInterval(() => {
  if (S.screen !== 'video') return;
  if (player && player.getCurrentTime && S.playing) {
    try { S.t = player.getCurrentTime(); } catch (_) { /* not ready */ }
    updatePlayhead();
  } else if (S.playing && !player) {
    // Offline: advance an internal clock so the throughline still demonstrates playback.
    const v = currentVideo();
    if (v) { S.t = S.t + 0.25 >= v.duration_s ? 0 : S.t + 0.25; updatePlayhead(); }
  }
}, 250);

// ── run lifecycle ───────────────────────────────────────────────────
async function startRun() {
  const urls = parseUrls(S.urlText);
  if (!urls.length) return;
  // Belt-and-braces against a double click: the server serializes pipeline
  // executions anyway, but a second job started here would queue behind the
  // first for minutes for no reason.
  if (S.starting) return;
  if (S.progress && S.progress.status === 'running') {
    S.runError = 'A run is already in progress. Wait for it to finish before starting another.';
    return render();
  }
  S.starting = true; S.runError = ''; render();
  try {
    const body = { urls, settings: S.settings };
    const res = await api('/api/runs', { method: 'POST', body: JSON.stringify(body) });
    S.jobId = res.job_id;
    S.results = null;
    S.progress = null;
    S.screen = 'prog';
    syncUrl();
    pollProgress();
  } catch (err) {
    S.runError = err.message;
  } finally {
    S.starting = false;
    render();
  }
}

let pollTimer = null;
async function pollProgress() {
  clearTimeout(pollTimer);
  if (!S.jobId) return;
  try {
    S.progress = await api('/api/runs/' + S.jobId);
  } catch (err) {
    S.runError = err.message;
    render();
    return;
  }
  if (S.progress.status === 'done' && !S.results) {
    try {
      S.results = await api('/api/runs/' + S.jobId + '/results');
      S.topicId = S.results.topics.length ? S.results.topics[0].topic_id : null;
      S.videoId = S.results.videos.length ? S.results.videos[0].video_id : null;
      S.t = 0;
      // Single-video runs skip the batch ceremony and land on the video page.
      S.screen = S.results.videos.length === 1 ? 'video' : 'batch';
    } catch (err) {
      S.runError = err.message;
    }
  }
  render();
  if (S.progress && S.progress.status === 'running') {
    pollTimer = setTimeout(pollProgress, 1000);
  }
}

// ── events ──────────────────────────────────────────────────────────
document.addEventListener('click', (event) => {
  const el = event.target.closest('[data-go],[data-action],[data-topic],[data-video],[data-field],[data-sort],[data-export],[data-dim],[data-excerpt-mode],[data-outlier-video]');
  if (!el) return;

  if (el.dataset.go) {
    const entry = NAV.find(([k]) => k === el.dataset.go);
    if (entry && !entry[2]()) return;
    S.screen = el.dataset.go;
    return render();
  }
  if (el.dataset.video) {
    // Any click that navigates to a moment — a transcript row, an excerpt's
    // Play button, a row in the batch list — is an explicit "take me there",
    // so it hands scrolling back to the playhead.
    setFollow(true);
    S.videoId = el.dataset.video;
    S.t = Number(el.dataset.seek || 0);
    S.dim = null;
    if (S.screen === 'video') { seek(S.t); return; }
    S.screen = 'video';
    return render();
  }
  if (el.dataset.topic) {
    S.topicId = Number(el.dataset.topic);
    S.screen = 'topics';
    return render();
  }
  if (el.dataset.dim) {
    const id = Number(el.dataset.dim);
    S.dim = S.dim === id ? null : id;
    return render();
  }
  if (el.dataset.sort) { S.sort = el.dataset.sort; return render(); }
  if (el.dataset.export) { S.exportKind = el.dataset.export; return render(); }
  if (el.dataset.excerptMode) { S.excerptMode = el.dataset.excerptMode; return render(); }
  if (el.dataset.outlierVideo) { S.outlierVideo = el.dataset.outlierVideo; return render(); }
  if (el.dataset.field) {
    const value = el.dataset.field === 'chunk_max_seconds' ? Number(el.dataset.value) : el.dataset.value;
    S.settings[el.dataset.field] = value;
    return render();
  }

  switch (el.dataset.action) {
    case 'toggle-adv': S.adv = !S.adv; return render();
    case 'toggle-whisper': S.settings.use_whisper_fallback = !S.settings.use_whisper_fallback; return render();
    case 'toggle-people': S.settings.detect_people = !S.settings.detect_people; return render();
    case 'start-run': return startRun();
    case 'toggle-play': return togglePlay();
    case 'resume-follow': {
      setFollow(true);
      const v = currentVideo();
      const row = v && document.querySelector(`.t-row[data-chunk="${activeChunkIndex(v)}"]`);
      if (row) followRow(row);
      return undefined;
    }
    default: return undefined;
  }
});

function togglePlay() {
  if (player && player.playVideo) {
    if (S.playing) player.pauseVideo(); else player.playVideo();
    return;
  }
  S.playing = !S.playing;
  const btn = document.querySelector('[data-action="toggle-play"]');
  if (btn) btn.textContent = S.playing ? '❚❚' : '▶';
}

document.addEventListener('input', (event) => {
  if (event.target.id === 'url-input') {
    S.urlText = event.target.value;
    // Update only the counters; re-rendering would steal focus from the textarea.
    const head = document.querySelector('.url-box-head .mono-note');
    const urls = parseUrls(S.urlText);
    if (head) head.textContent = `${urls.length} URL${urls.length === 1 ? '' : 's'} · ${urls.length} unique`;
    const btn = document.querySelector('[data-action="start-run"]');
    if (btn) btn.disabled = !urls.length;
  }
});

document.addEventListener('change', async (event) => {
  if (event.target.id !== 'csv-input') return;
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  const text = await file.text();
  const merged = mergeCsvUrls(S.urlText, text);
  if (merged === null) {
    S.runError = "CSV has no 'url', 'link', 'id' or 'video_id' column — nothing merged.";
  } else {
    S.urlText = merged;
    S.runError = '';
  }
  render();
});

/* Append a CSV's URL column to the pasted list, de-duplicated. Returns null
   when the file has no recognizable column. */
function mergeCsvUrls(existing, csvText) {
  const lines = csvText.split(/\r?\n/).filter((l) => l.trim());
  if (!lines.length) return null;
  const header = lines[0].split(',').map((h) => h.trim().toLowerCase().replace(/^"|"$/g, ''));
  const col = header.findIndex((h) => ['url', 'link', 'id', 'video_id'].includes(h));
  if (col < 0) return null;
  const found = lines.slice(1)
    .map((line) => (line.split(',')[col] || '').trim().replace(/^"|"$/g, ''))
    .filter(Boolean);
  return parseUrls([existing, found.join('\n')].filter(Boolean).join('\n')).join('\n');
}

$('#theme-toggle').addEventListener('click', () => {
  S.theme = S.theme === 'light' ? 'dark' : 'light';
  localStorage.setItem('tl-theme', S.theme);
  render();
});

/* Keep the run id in the address bar. Results are session-scoped, but the job
   lives in the server process — so a refresh mid-run reattaches instead of
   stranding a run the user can no longer see. */
function syncUrl() {
  if (!S.jobId) return;
  const params = new URLSearchParams({ job: S.jobId, screen: S.screen });
  history.replaceState(null, '', '?' + params.toString());
}

// ── boot ────────────────────────────────────────────────────────────
(async function boot() {
  document.documentElement.setAttribute('data-theme', S.theme);
  const params = new URLSearchParams(location.search);
  if (params.get('theme')) S.theme = params.get('theme');
  try {
    const defaults = await api('/api/defaults');
    // Adopt only the keys this UI controls — /api/defaults also reports
    // chunk_max_words, and copying it in would put it back on the wire.
    Object.keys(S.settings).forEach((key) => {
      if (defaults[key] !== undefined) S.settings[key] = defaults[key];
    });
  } catch (_) { /* keep built-in defaults */ }

  const job = params.get('job');
  if (job) {
    S.jobId = job;
    const wanted = params.get('screen');
    render();
    await pollProgress();
    if (wanted && SCREENS[wanted]) {
      // Screens reachable from within a page (the outlier inspector) are not
      // in NAV; only screens that ARE in NAV have to satisfy a nav guard.
      const entry = NAV.find(([k]) => k === wanted);
      if (!entry || entry[2]()) { S.screen = wanted; render(); }
    }
    return;
  }
  render();
})();
