const API_URL = "http://127.0.0.1:8000/predict";

function escapeHtml(str) {
  return str.replaceAll("&","&amp;")
            .replaceAll("<","&lt;")
            .replaceAll(">","&gt;");
}

function highlightText(original, flaggedWords) {
  if (!flaggedWords || flaggedWords.length === 0) return escapeHtml(original);

  const escaped = flaggedWords.map(w => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp("\\b(" + escaped.join("|") + ")\\b", "gi");
  return escapeHtml(original).replace(re, (m) => `<mark>${m}</mark>`);
}

window.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("analyzeBtn");
  const textInput = document.getElementById("textInput");
  const removeToggle = document.getElementById("removeToggle");

  if (!btn || !textInput || !removeToggle) {
    alert("HTML IDs mismatch. Check analyzeBtn, textInput, removeToggle.");
    console.error("Missing elements:", { btn, textInput, removeToggle });
    return;
  }

  btn.addEventListener("click", async () => {
    try {
      console.log("Analyze clicked ✅");

      const text = textInput.value || "";
      const remove = removeToggle.checked;

      const resBox = document.getElementById("result");
      resBox.classList.add("hidden");

      const resp = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, remove })
      });

      const data = await resp.json().catch(() => ({}));

      if (!resp.ok) {
        alert(data.error || `Request failed: ${resp.status}`);
        console.error("Backend error:", data);
        return;
      }

      const label = data.label;
      const conf = Math.round((data.confidence || 0) * 100);
      const flagged = data.flagged_words || [];

      const badge = document.getElementById("badge");
      badge.textContent = label === 1 ? "HATE DETECTED" : "NON-HATE";
      badge.className = "badge " + (label === 1 ? "hate" : "clean");

      document.getElementById("meta").textContent =
        `Confidence: ${conf}%  |  Flagged words: ${flagged.length ? flagged.join(", ") : "None"}`;

      document.getElementById("highlighted").innerHTML = highlightText(text, flagged);
      document.getElementById("cleaned").textContent = data.cleaned_text || text;

      resBox.classList.remove("hidden");
    } catch (err) {
      alert("Frontend error (check Console).");
      console.error(err);
    }
  });
});

