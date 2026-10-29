// Shows a dismissible banner when the page being loaded scores as high risk.
//
// WHAT THIS DELIBERATELY DOES NOT DO
// ----------------------------------
// It does not block navigation. Safari web extensions can, but a false
// positive that strands someone on a page they wanted is how a security tool
// gets switched off -- and the hard-benign benchmark for this model measured
// a real false-positive rate on well-known sites. So the user is told and the
// user decides.
//
// It also reads nothing from the page. Only location.href is sent, and only to
// the extension's own native component on this device.

(async () => {
  const url = location.href;
  if (!/^https?:/.test(url)) return;

  let v;
  try {
    v = await browser.runtime.sendMessage({ type: "assess", url });
  } catch (e) {
    return;                        // fail open
  }
  if (!v || !v.available) return;
  if (v.band !== "high") return;   // only the top band interrupts

  const show = () => {
    if (document.getElementById("urlguard-banner")) return;
    const bar = document.createElement("div");
    bar.id = "urlguard-banner";
    bar.setAttribute("role", "alert");
    bar.style.cssText = [
      "position:fixed", "top:0", "left:0", "right:0", "z-index:2147483647",
      "background:#7f1d1d", "color:#fff", "padding:12px 16px",
      "font:14px/1.45 -apple-system,system-ui,sans-serif",
      "box-shadow:0 2px 12px rgba(0,0,0,.35)",
    ].join(";");

    const title = document.createElement("strong");
    title.textContent = `URLGuard: ${v.label} (${v.score}/100)`;
    const body = document.createElement("div");
    body.style.cssText = "margin-top:4px;opacity:.92";
    body.textContent = v.detail;

    const signals = document.createElement("div");
    signals.style.cssText = "margin-top:6px;font-size:12.5px;opacity:.85";
    signals.textContent = (v.signals || []).slice(0, 3).join("  ·  ");

    const note = document.createElement("div");
    note.style.cssText = "margin-top:6px;font-size:11.5px;opacity:.7";
    note.textContent = v.disclaimer;

    const close = document.createElement("button");
    close.textContent = "Dismiss";
    close.style.cssText = [
      "position:absolute", "top:10px", "right:12px", "background:#fff3",
      "border:1px solid #fff6", "color:#fff", "border-radius:6px",
      "padding:4px 10px", "font-size:12px", "cursor:pointer",
    ].join(";");
    close.addEventListener("click", () => bar.remove());

    bar.append(title, body, signals, note, close);
    (document.body || document.documentElement).prepend(bar);
  };

  if (document.body) show();
  else document.addEventListener("DOMContentLoaded", show, { once: true });
})();
