const state = {
  status: null,
};

const $ = (id) => document.getElementById(id);

function setMessage(text, isError = false) {
  const message = $("message");
  message.textContent = text;
  message.classList.toggle("error", isError);
}

function detailToText(detail) {
  if (!detail) return "Unknown error / 未知错误";
  if (typeof detail === "string") return detail;
  if (detail.zh || detail.en) return `${detail.zh || ""} / ${detail.en || ""}`;
  return JSON.stringify(detail);
}

async function refreshStatus() {
  $("ollamaState").textContent = "Checking / 检查中";
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    state.status = data;
    $("ollamaState").textContent = data.ollama_available ? "Connected / 已连接" : "Unavailable / 不可用";
    $("imageModelState").textContent = data.default_image_model;
    $("outputState").textContent = "Ready / 就绪";
    $("imageModel").value = data.default_image_model;

    const select = $("ollamaModel");
    select.innerHTML = "";
    const models = data.ollama_models.length ? data.ollama_models : [data.default_ollama_model];
    for (const model of models) {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      if (model === data.default_ollama_model) option.selected = true;
      select.append(option);
    }
  } catch (error) {
    $("ollamaState").textContent = "Error / 错误";
    setMessage(`Status failed / 状态读取失败: ${error.message}`, true);
  }
}

async function generateImage(event) {
  event.preventDefault();
  const button = $("generateButton");
  button.disabled = true;
  button.textContent = "Generating / 生成中";
  setMessage("Running Ollama prompt expansion and local diffusion render / 正在运行 Ollama 提示词优化和本地扩散生图");

  const seedValue = $("seed").value.trim();
  const payload = {
    prompt: $("prompt").value.trim(),
    ollama_model: $("ollamaModel").value,
    image_model: $("imageModel").value.trim(),
    use_ollama: $("useOllama").checked,
    style: $("style").value,
    steps: Number($("steps").value || 6),
    seed: seedValue ? Number(seedValue) : null,
  };

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(detailToText(data.detail));

    const frame = $("imageFrame");
    frame.innerHTML = "";
    const img = document.createElement("img");
    img.src = `${data.image_url}?t=${Date.now()}`;
    img.alt = "Generated image / 生成图片";
    frame.append(img);

    setMessage(`${data.zh_summary} / ${data.en_summary} (${data.elapsed_seconds}s)`);
    $("promptUsed").textContent = data.prompt_used;
    $("savedFile").textContent = data.image_path;
  } catch (error) {
    setMessage(`Generation failed / 生成失败: ${error.message}`, true);
  } finally {
    button.disabled = false;
    button.textContent = "Generate Image / 生成图片";
  }
}

$("refreshStatus").addEventListener("click", refreshStatus);
$("generateForm").addEventListener("submit", generateImage);
refreshStatus();
