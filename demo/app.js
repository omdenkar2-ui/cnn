const $ = (id) => document.getElementById(id);
let selected = null;
let previewUrl = null;
let busy = false;

function clearResult() {
  $("label").textContent = "Ready when you are.";
  $("status").textContent = "Upload a cropped face to see the model’s two class scores.";
  for (const label of ["happy", "sad"]) {
    $(`${label}-value`).textContent = "—";
    $(`${label}-bar`).value = 0;
  }
  $("error").hidden = true;
}

function choose(file) {
  if (busy || !file) return;
  clearResult();
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type) || file.size > 10 * 1024 * 1024) {
    $("error").textContent = "Choose a JPG, PNG or WebP image smaller than 10 MB.";
    $("error").hidden = false;
    return;
  }
  selected = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  $("preview").src = previewUrl;
  $("preview").hidden = false;
  $("empty").hidden = true;
  $("reset").hidden = false;
  $("analyze").disabled = false;
  $("status").textContent = "Image ready. Classify it to see the two class scores.";
}

$("file").addEventListener("change", (event) => choose(event.target.files[0]));
for (const event of ["dragenter", "dragover"]) {
  $("dropzone").addEventListener(event, (e) => {
    e.preventDefault();
    $("dropzone").classList.add("dragging");
  });
}
for (const event of ["dragleave", "drop"]) {
  $("dropzone").addEventListener(event, (e) => {
    e.preventDefault();
    $("dropzone").classList.remove("dragging");
    if (event === "drop") choose(e.dataTransfer.files[0]);
  });
}
$("reset").addEventListener("click", () => {
  if (busy) return;
  selected = null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  $("preview").removeAttribute("src");
  $("preview").hidden = true;
  $("empty").hidden = false;
  $("file").value = "";
  $("reset").hidden = true;
  $("analyze").disabled = true;
  clearResult();
});
$("analyze").addEventListener("click", async () => {
  if (!selected || busy) return;
  busy = true;
  $("analyze").disabled = true;
  $("file").disabled = true;
  $("reset").disabled = true;
  $("error").hidden = true;
  $("label").textContent = "Taking a closer look…";
  $("status").textContent = "Running the model locally on your computer.";
  try {
    const response = await fetch("/api/predict", {
      method: "POST", headers: {"Content-Type": "application/octet-stream"}, body: selected
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Prediction failed.");
    $("label").textContent = result.label === "happy" ? "Happy expression" : "Sad expression";
    $("status").textContent = result.uncertain
      ? "Low confidence. The two class scores are close; treat this prediction cautiously."
      : "Class scores use validation-set temperature scaling. Confidence is not a guarantee of correctness.";
    for (const label of ["happy", "sad"]) {
      $(`${label}-value`).textContent = `${(result.scores[label] * 100).toFixed(1)}%`;
      $(`${label}-bar`).value = result.scores[label] * 100;
    }
  } catch (error) {
    clearResult();
    $("label").textContent = "Let’s try another image.";
    $("error").textContent = error.message;
    $("error").hidden = false;
  } finally {
    busy = false;
    $("analyze").disabled = false;
    $("file").disabled = false;
    $("reset").disabled = false;
  }
});
