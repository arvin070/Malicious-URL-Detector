const params = new URLSearchParams(window.location.search);
const blockedUrl = params.get("blocked") || "";
const score = params.get("score") || "0";
const votes = params.get("votes") || "0";
const total = params.get("total") || "0";

document.getElementById("blocked-url").textContent = blockedUrl;
document.getElementById("score-text").textContent = `${Math.round(parseFloat(score) * 100)}% malicious probability`;
document.getElementById("votes-text").textContent = `${votes} / ${total} models flagged this as malicious`;

document.getElementById("go-back-btn").addEventListener("click", () => {
  if (window.history.length > 1) {
    window.history.back();
  } else {
    window.location.href = "https://www.google.com";
  }
});

document.getElementById("proceed-btn").addEventListener("click", () => {
  chrome.runtime.sendMessage({ type: "PROCEED_ANYWAY", url: blockedUrl }, () => {
    window.location.href = blockedUrl;
  });
});
