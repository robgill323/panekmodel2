/* Channel picker: the decisions, isolated and pure.

   The New Run screen can list a channel's newest uploads and let the
   researcher tick the ones to analyze. Everything the picker DECIDES lives
   here as functions of plain values — the count it asks for, the selection,
   the URLs that land in the textarea, which panel shows — so it can be tested
   in node without a browser. app.js owns the DOM around these. */
'use strict';

const PICKER_DEFAULT_COUNT = 25;
const PICKER_MAX_COUNT = 100; // mirrors MAX_COUNT in channel_resolver.py

/** The count to request: a whole number in 1..100, 25 when unset or junk. */
function clampCount(raw) {
  if (raw === '' || raw == null) return PICKER_DEFAULT_COUNT;
  const n = Math.round(Number(raw));
  if (!Number.isFinite(n)) return PICKER_DEFAULT_COUNT;
  return Math.min(PICKER_MAX_COUNT, Math.max(1, n));
}

/** A new selection with `id` checked or unchecked. Never mutates `selected`. */
function setSelected(selected, id, on) {
  const rest = (selected || []).filter((s) => s !== id);
  return on ? rest.concat([id]) : rest;
}

function selectAll(videos) {
  return (videos || []).map((v) => v.id);
}

function selectNone() {
  return [];
}

/**
 * Watch URLs for the selected videos, in the channel's (newest-first) order.
 * Only IDs present in `videos` count, so a selection left over from a
 * previous channel can never add anything.
 */
function selectedUrls(videos, selected) {
  const wanted = new Set(selected || []);
  return (videos || []).filter((v) => wanted.has(v.id)).map((v) => v.url);
}

/**
 * Append `urls` to the textarea's text without rewriting what is there.
 *
 * `existingUrls` is the textarea already run through app.js's parseUrls, so
 * "already there" means exactly what the run's own de-duplication means.
 * Returns the new text plus how many were added and how many were skipped.
 */
function appendToCorpus(text, existingUrls, urls) {
  const seen = new Set(existingUrls || []);
  const fresh = [];
  let already = 0;
  (urls || []).forEach((u) => {
    if (seen.has(u)) { already += 1; return; }
    seen.add(u);
    fresh.push(u);
  });
  const base = String(text || '');
  if (!fresh.length) return { text: base, added: 0, already };
  const sep = base === '' || base.endsWith('\n') ? '' : '\n';
  return { text: base + sep + fresh.join('\n'), added: fresh.length, already };
}

/**
 * Which picker panel to show: idle | loading | list | empty | error.
 * "Nothing to list" is its own designed state, not an error banner.
 */
function pickerPanelState(ch) {
  if (!ch) return 'idle';
  if (ch.status === 'loading') return 'loading';
  if (ch.status === 'ready') return ch.videos && ch.videos.length ? 'list' : 'empty';
  if (ch.status === 'error') return ch.kind === 'private_or_empty' ? 'empty' : 'error';
  return 'idle';
}

/**
 * Should a thumbnail be swapped for the fallback tile?
 *
 * i.ytimg.com answers a missing mqdefault.jpg with HTTP 404 *and* a valid
 * 120×90 grey placeholder JPEG, so a browser may render it and never fire
 * `error`. A real mqdefault is 320×180; anything 120 wide or narrower that
 * loaded is the placeholder.
 */
function thumbIsBroken(event) {
  if (!event) return false;
  if (event.type === 'error') return true;
  const w = Number(event.naturalWidth) || 0;
  return event.type === 'load' && w > 0 && w <= 120;
}

function addLabel(n) {
  return n ? `Add ${n} selected to run` : 'Select videos to add';
}

function addedNote(added, already) {
  if (!added) return `All ${already} were already in the list — nothing added.`;
  const head = `Added ${added} video${added === 1 ? '' : 's'} to the list above`;
  return already ? `${head} · ${already} ${already === 1 ? 'was' : 'were'} already there.` : `${head}.`;
}

/* Exported for the node-based unit tests; the browser picks up the globals. */
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    clampCount, setSelected, selectAll, selectNone, selectedUrls,
    appendToCorpus, pickerPanelState, thumbIsBroken, addLabel, addedNote,
  };
}
