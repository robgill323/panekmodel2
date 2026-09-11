/* Transcript auto-follow: the scroll arithmetic, isolated and pure.

   Playback used to call row.scrollIntoView(), which scrolls EVERY scrollable
   ancestor including the document — so following the playhead dragged the
   whole page down, and re-fired on the next tick the moment the reader
   scrolled back up. An un-winnable fight for the viewport.

   Keeping the maths here, as a function of numbers only, means it can be
   tested without a browser: the rest of the follow behaviour is DOM wiring
   around this one decision. */
'use strict';

/**
 * Where the transcript container should be scrolled to keep a row visible.
 *
 * @param {object} view
 * @param {number} view.rowTop        row offset from the top of the container's content
 * @param {number} view.rowHeight     height of the row
 * @param {number} view.scrollTop     container's current scrollTop
 * @param {number} view.viewHeight    container's visible height
 * @param {number} [view.margin]      breathing room to keep above/below the row
 * @param {number} [view.contentHeight] total scrollable content height, for clamping
 * @returns {number|null} target scrollTop, or null when no scrolling is needed
 */
function followScrollTop(view) {
  const rowTop = Number(view.rowTop) || 0;
  const rowHeight = Number(view.rowHeight) || 0;
  const scrollTop = Number(view.scrollTop) || 0;
  const viewHeight = Number(view.viewHeight) || 0;
  const margin = Number(view.margin) || 0;

  // A container that cannot scroll has nothing to decide.
  if (viewHeight <= 0) return null;

  const rowBottom = rowTop + rowHeight;
  const viewTop = scrollTop;
  const viewBottom = scrollTop + viewHeight;

  let target = null;
  if (rowTop - margin < viewTop) {
    // Row is above the visible band — bring it to the top.
    target = rowTop - margin;
  } else if (rowBottom + margin > viewBottom) {
    // Row is below it — bring its bottom to the bottom. For a row taller than
    // the container, prefer showing its start over its end.
    target = rowHeight + margin * 2 >= viewHeight
      ? rowTop - margin
      : rowBottom + margin - viewHeight;
  } else {
    return null; // already comfortably in view: do not move the reader
  }

  const maxScroll = Number.isFinite(view.contentHeight)
    ? Math.max(0, view.contentHeight - viewHeight)
    : Infinity;
  target = Math.max(0, Math.min(target, maxScroll));

  // Never report a "move" that is not one — it would only reset the
  // programmatic-scroll flag for nothing.
  return Math.round(target) === Math.round(scrollTop) ? null : target;
}


/* ── The decisions the DOM layer makes, extracted so they can be tested ──
   followScrollTop above answers "where to scroll". These answer "whether to",
   which is where the regressions live: each is one boolean that a plausible
   one-line change could invert, and none of them needs a browser to check. */

/**
 * Should this playback tick scroll the transcript?
 * Scrolling is the only thing pausing suspends — highlighting is unconditional.
 */
function shouldFollow(state) {
  return Boolean(state && state.playing && state.followEnabled);
}

/**
 * Does this scroll event mean the reader has taken over?
 *
 * Auto-follow scrolls the container itself, so the container's scroll handler
 * sees our own scrolls too. Without the programmatic flag it reads the first
 * one back as the reader taking over, disables following, and never recovers.
 */
function isReaderScroll(event) {
  return !(event && event.programmatic);
}

/**
 * Is the "follow playhead" chip hidden?
 * It exists to offer the way back, so it is visible exactly when follow is off.
 */
function followChipHidden(state) {
  return Boolean(state && state.followEnabled);
}

/* Exported for the node-based unit tests; the browser picks up the globals. */
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { followScrollTop, shouldFollow, isReaderScroll, followChipHidden };
}
