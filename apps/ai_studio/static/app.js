const $ = (id) => document.getElementById(id);
const appUrl = (path) => new URL(path.replace(/^\/+/, ""), new URL("./", document.baseURI)).toString();

const panelNames = {support: "문서 도우미", media: "이미지 스튜디오", video: "동영상 생성", labs: "모델 실험실"};
document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab, .panel").forEach((node) => node.classList.remove("active"));
    button.classList.add("active");
    $(button.dataset.panel).classList.add("active");
    $("breadcrumb-current").textContent = panelNames[button.dataset.panel] || "AI Studio";
  });
});

$("operation").addEventListener("change", () => {
  $("upload-fields").classList.toggle("hidden", $("operation").value !== "inpaint");
});

async function checkHealth() {
  try {
    const response = await fetch(appUrl("api/health"));
    if (!response.ok) throw new Error("health check failed");
    const data = await response.json();
    const online = data.status === "development";
    $("health").textContent = online ? "서비스 연결됨" : "상태 확인 필요";
    $("health").parentElement.classList.toggle("is-online", online);
  } catch {
    $("health").textContent = "서버 연결 실패";
    $("health").parentElement.classList.remove("is-online");
  }
}

$("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const status = $("chat-status");
  const result = $("chat-result");
  status.textContent = "검색 및 답변 생성 중…";
  result.classList.add("hidden");
  try {
    const response = await fetch(appUrl("api/support/chat"), {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question: $("question").value, top_k: 3})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "요청 실패");
    result.textContent = `${data.abstained ? "답변 보류\n\n" : ""}${data.answer}\n\n근거:\n${(data.citations || []).map((c) => `${c.source} · 청크 ${c.chunk} · 점수 ${c.score}`).join("\n") || "없음"}`;
    result.classList.remove("hidden");
    status.textContent = "완료";
  } catch (error) { status.textContent = `오류: ${error.message}`; }
});

async function pollJob(jobId, statusId = "media-status", resultId = "media-result") {
  const status = $(statusId);
  const result = $(resultId);
  const poll = async () => {
    const response = await fetch(appUrl(`api/jobs/${jobId}`));
    const job = await response.json();
    const labels = {queued: "대기 중", running: "실행 중", succeeded: "완료", failed: "실패"};
    status.dataset.state = job.status;
    status.textContent = `작업 ${job.id}: ${labels[job.status] || job.status}${job.error ? ` · ${job.error}` : ""}`;
    if (job.status === "succeeded") {
      result.replaceChildren();
      if (["result.png", "result.mp4"].includes(job.result_name)) {
        const preview = job.result_name.endsWith(".mp4") ? document.createElement("video") : document.createElement("img");
        preview.src = `${appUrl(job.result)}?t=${Date.now()}`;
        if (preview.tagName === "VIDEO") { preview.controls = true; preview.playsInline = true; }
        else preview.alt = "생성 결과";
        result.append(preview);
      } else if (["classification.json", "infer_results.json"].includes(job.result_name)) {
        const response = await fetch(appUrl(job.result));
        const preview = document.createElement("pre");
        preview.textContent = await response.text();
        result.append(preview);
      }
      const link = document.createElement("a");
      link.href = appUrl(job.result);
      link.download = job.result_name || "ai-studio-result";
      link.textContent = `결과 다운로드 (${job.result_name || "파일"})`;
      result.append(link);
      result.classList.remove("hidden");
      return;
    }
    if (job.status === "failed") return;
    window.setTimeout(poll, 2000);
  };
  await poll();
}

$("media-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const status = $("media-status");
  const result = $("media-result");
  const form = new FormData();
  form.append("operation", $("operation").value);
  form.append("prompt", $("prompt").value);
  form.append("seed", $("seed").value || "42");
  if ($("source").files[0]) form.append("image", $("source").files[0]);
  if ($("mask").files[0]) form.append("mask", $("mask").files[0]);
  status.textContent = "GPU 작업 대기 중…";
  result.classList.add("hidden");
  try {
    const response = await fetch(appUrl("api/media/image"), {method: "POST", body: form});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "작업 제출 실패");
    await pollJob(data.id);
  } catch (error) { status.textContent = `오류: ${error.message}`; }
});

$("video-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData();
  form.append("prompt", $("video-prompt").value);
  form.append("frame_num", $("video-frames").value);
  form.append("seed", $("video-seed").value || "42");
  if ($("video-image").files[0]) form.append("image", $("video-image").files[0]);
  $("video-status").textContent = "동영상 GPU 작업 대기 중…";
  $("video-result").classList.add("hidden");
  try {
    const response = await fetch(appUrl("api/media/video"), {method: "POST", body: form});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "작업 제출 실패");
    await pollJob(data.id, "video-status", "video-result");
  } catch (error) { $("video-status").textContent = `오류: ${error.message}`; }
});

$("lab-action").addEventListener("change", () => {
  $("lab-image-wrap").classList.toggle("hidden", $("lab-action").value !== "vl-receipt");
  $("classify-input-wrap").classList.toggle("hidden", $("lab-action").value !== "classify");
});

$("lab-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData();
  form.append("action", $("lab-action").value);
  form.append("request_text", $("classify-request").value);
  const image = $("lab-image").files[0];
  if (image) form.append("image", image);
  $("lab-status").textContent = "실습 작업 대기 중…";
  $("lab-result").classList.add("hidden");
  try {
    const response = await fetch(appUrl("api/labs/run"), {method: "POST", body: form});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "작업 제출 실패");
    await pollJob(data.id, "lab-status", "lab-result");
  } catch (error) { $("lab-status").textContent = `오류: ${error.message}`; }
});

async function loadFeatures() {
  try {
    const response = await fetch(appUrl("api/features"));
    const data = await response.json();
    const video = data.models.find((model) => model.key === "wan22-ti2v");
    $("video-submit").disabled = !video?.enabled;
    $("video-readiness").textContent = video?.enabled
      ? "Wan2.2 저장소와 체크포인트가 설정되었습니다. 모델 처리에는 시간이 걸릴 수 있습니다."
      : "Wan2.2 공식 저장소와 체크포인트 경로를 설정해야 사용할 수 있습니다.";
  } catch { $("video-readiness").textContent = "서버 기능 상태를 확인할 수 없습니다."; }
}

checkHealth();
loadFeatures();
