fetch("http://127.0.0.1:5000/api/check?url=https://example.com")
  .then((res) => {
    if (res.ok) {
      document.getElementById("status-dot").className = "dot dot-ok";
      document.getElementById("status-text").textContent = "Connected — protection active";
    } else {
      throw new Error("bad response");
    }
  })
  .catch(() => {
    document.getElementById("status-dot").className = "dot dot-bad";
    document.getElementById("status-text").textContent = "Backend not running (python webapp/app.py)";
  });
