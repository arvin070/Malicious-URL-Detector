// Background worker: the only thing it does is forward a URL to the native
// half and return the verdict.
//
// Scoring is NOT reimplemented in JavaScript on purpose. A second
// implementation of the 124 features would be a third place for them to drift
// away from src/features.py, and this project has already shipped three such
// parity bugs between Python and Kotlin. The native handler runs the same
// ONNX model and the same URLGuardCore extractor the app uses.
//
// No page content is read and no request is made to the site.

const cache = new Map();          // url -> verdict, per browser session
const MAX_CACHE = 200;

async function assess(url) {
  if (cache.has(url)) return cache.get(url);
  let verdict;
  try {
    verdict = await browser.runtime.sendNativeMessage("application.id", { url });
  } catch (e) {
    // Fail open. A broken extension must never stop someone browsing.
    verdict = { available: false, reason: "the native component did not respond" };
  }
  if (cache.size >= MAX_CACHE) cache.delete(cache.keys().next().value);
  cache.set(url, verdict);
  return verdict;
}

browser.runtime.onMessage.addListener((msg, sender, respond) => {
  if (msg && msg.type === "assess" && typeof msg.url === "string") {
    assess(msg.url).then(respond);
    return true;                   // keep the channel open for the async reply
  }
});
