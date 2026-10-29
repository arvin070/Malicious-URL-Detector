// AI Phishing URL Guard -- background service worker.
//
// Strategy: Chrome's Manifest V3 doesn't allow truly pausing a navigation
// mid-flight, so instead we listen for onBeforeNavigate (fires as early as
// possible, before the page has rendered), immediately ask our local AI
// model API whether the URL looks malicious, and if so, redirect the tab
// to a local warning page before the phishing page has a chance to load.

const API_BASE = "http://127.0.0.1:5000";
const CHECK_TIMEOUT_MS = 2500;

// URLs the user explicitly chose to "proceed anyway" on -- skip re-checking
// them once, so they don't get redirected back to the warning page.
const bypassedUrls = new Set();

// Simple in-memory cache so revisiting the same URL in one session doesn't
// re-call the API every time.
const resultCache = new Map();
const CACHE_MAX = 500;

function shouldSkip(url) {
  if (!url) return true;
  if (!url.startsWith("http://") && !url.startsWith("https://")) return true;
  if (url.startsWith(API_BASE)) return true;
  return false;
}

async function checkUrl(url) {
  if (resultCache.has(url)) return resultCache.get(url);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), CHECK_TIMEOUT_MS);

  try {
    const res = await fetch(`${API_BASE}/api/check?url=${encodeURIComponent(url)}`, {
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (!res.ok) return null;
    const data = await res.json();
    if (resultCache.size >= CACHE_MAX) {
      resultCache.delete(resultCache.keys().next().value);
    }
    resultCache.set(url, data);
    return data;
  } catch (err) {
    clearTimeout(timeout);
    // API unreachable (server not running) or timed out -- fail OPEN, not
    // closed, so a down local server never blocks normal browsing.
    console.warn("AI Phishing Guard: could not reach local API", err);
    return null;
  }
}

chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
  // Only act on the main frame (the actual page navigation), not iframes,
  // ads, or sub-resources -- checking every sub-request would be noisy
  // and slow for no benefit.
  if (details.frameId !== 0) return;

  const url = details.url;
  if (shouldSkip(url)) return;

  if (bypassedUrls.has(url)) {
    bypassedUrls.delete(url);
    return;
  }

  const result = await checkUrl(url);
  if (!result || result.error) return; // fail open
  if (!result.malicious) return;

  const warningUrl =
    chrome.runtime.getURL("warning.html") +
    `?blocked=${encodeURIComponent(url)}` +
    `&score=${encodeURIComponent(result.score)}` +
    `&votes=${encodeURIComponent(result.votes_malicious)}` +
    `&total=${encodeURIComponent(result.total_models)}`;

  chrome.tabs.update(details.tabId, { url: warningUrl });
});

// Messages from warning.js: user clicked "Proceed anyway"
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "PROCEED_ANYWAY" && message.url) {
    bypassedUrls.add(message.url);
    sendResponse({ ok: true });
  }
  return true;
});
