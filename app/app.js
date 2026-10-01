const $ = (id) => document.getElementById(id);

const ui = {
  video: $("video"), stage: $("videoStage"), empty: $("cameraEmpty"), canvas: $("analysisCanvas"), liveCanvas: $("liveCanvas"),
  camera: $("cameraButton"), record: $("recordButton"), file: $("fileInput"), badge: $("recordingBadge"),
  timer: $("timer"), state: $("modelState"), modeLabel: $("resultModeLabel"), emptyResult: $("resultEmpty"),
  output: $("resultOutput"), required: $("modelRequired"), translation: $("translationText"),
  confidence: $("confidenceText"), confidenceBar: $("confidenceBar"), latency: $("latencyText"),
  model: $("modelText"), copy: $("copyButton"), toast: $("toast"), qualityState: $("qualityState"),
  livePanel: $("livePanel"), historyPanel: $("historyPanel"), liveTab: $("liveTab"), historyTab: $("historyTab"),
  historyEmpty: $("historyEmpty"), historyOutput: $("historyOutput"), historyText: $("historyText"),
};

let stream = null;
let recorder = null;
let chunks = [];
let currentBlob = null;
let objectUrl = null;
let startedAt = 0;
let timerId = null;
let analysisId = null;
let previousLuma = null;
let samples = [];
let busy = false;
let mediaBusy = false;
let modelReady = false;
let statusTimer;
let toastTimer;
let mediaTimer;
let liveId = null;
let liveSession = 0;
let liveBusy = false;
let liveHistory = [];
let liveSeenAt = 0;
let liveErrorShown = false;
let historyWords = [];
const historyByMode = {gestures: [], asl: [], asl_letters: []};
let recognitionMode = "gestures";
let lastCommittedGesture = null;
let lastObservedAt = 0;

const gestureGuide = [
  ["👍", "Yes", "Thumb up"], ["👎", "No", "Thumb down"],
  ["✋", "Stop", "Open palm"], ["✌️", "Peace", "Index + middle"],
  ["🤟", "I love you", "Thumb + index + pinky"], ["☝️", "One", "Index finger"],
  ["✊", "Ready", "Closed fist"], ["👌", "Okay", "Thumb + index touch; other fingers up"],
  ["🤙", "Call", "Thumb + pinky"], ["🤘", "Rock", "Index + pinky"],
  ["🖐️", "Four", "Four fingers; thumb tucked"], ["🤟", "Three", "Thumb + index + middle"],
  ["🫰", "Pinch", "Thumb + index touch; other fingers folded"], ["☝️", "L", "Index + thumb at a right angle"],
  ["🤏", "Little", "Pinky only"], ["🖐️", "Triple", "Index + middle + ring"],
  ["", "Thumb + middle", "Two fingers raised"],
  ["", "Thumb + ring", "Two fingers raised"],
  ["", "Thumb + middle + ring", "Three fingers raised"],
  ["", "Thumb + middle + pinky", "Three fingers raised"],
  ["", "Thumb + ring + pinky", "Three fingers raised"],
  ["", "Index + ring", "Two fingers raised"],
  ["", "Index + middle + pinky", "Three fingers raised"],
  ["", "Index + ring + pinky", "Three fingers raised"],
  ["", "Middle + ring + pinky", "Three fingers raised"],
  ["", "Middle + pinky", "Two fingers raised"],
];
const extraFingerPatterns = ["10100", "10010", "10110", "10101", "10011",
                             "01010", "01101", "01011", "00111", "00101"];
const fingerNames = ["Thumb", "Index", "Middle", "Ring", "Pinky"];
const aslGuide = [
  ["☝️", "One", "Index only"], ["✌️", "Two", "Index + middle"],
  ["🤟", "Three", "Thumb + index + middle"], ["🖐️", "Four", "Four fingers; thumb folded"],
  ["✋", "Five", "All five fingers"], ["⑥", "Six", "Thumb touches pinky"],
  ["⑦", "Seven", "Thumb touches ring finger"], ["⑧", "Eight", "Thumb touches middle finger"],
  ["⑨", "Nine", "Thumb touches index finger"],
];
const aslLetterGuide = [
  ["A", "A", "Fist; thumb beside the fingers"],
  ["B", "B", "Four fingers together; thumb across palm"],
  ["C", "C", "Curve the hand into a C"],
  ["D", "D", "Index up; other fingertips meet thumb"],
  ["E", "E", "Bent fingertips touch the thumb"],
  ["F", "F", "Thumb + index touch; three fingers up"],
  ["G", "G", "Index + thumb point sideways"],
  ["H", "H", "Index + middle together, sideways"],
  ["I", "I", "Pinky straight up"],
  ["J", "J", "Draw a J in the air with the pinky"],
  ["K", "K", "Index + middle up; thumb between"],
  ["L", "L", "Thumb + index make an L"],
  ["M", "M", "Thumb under three folded fingers"],
  ["N", "N", "Thumb under two folded fingers"],
  ["O", "O", "Fingertips curve to meet thumb"],
  ["P", "P", "K handshape pointing down"],
  ["Q", "Q", "G handshape pointing down"],
  ["R", "R", "Index + middle crossed"],
  ["S", "S", "Fist with thumb across fingers"],
  ["T", "T", "Thumb between index + middle"],
  ["U", "U", "Index + middle together, upright"],
  ["V", "V", "Index + middle spread apart"],
  ["W", "W", "Index + middle + ring up"],
  ["X", "X", "Index finger hooked"],
  ["Y", "Y", "Thumb + pinky out"],
  ["Z", "Z", "Draw a Z in the air with the index"],
];

function aslHandIcon(number) {
  // Small schematic: raised fingers are blue; the thumb/contact for 6–9 is the blue curve.
  const raised = {
    1:[0], 2:[0,1], 3:[0,1], 4:[0,1,2,3], 5:[0,1,2,3],
    6:[0,1,2], 7:[0,1,3], 8:[0,2,3], 9:[1,2,3],
  }[number];
  const fingerX = [36,47,58,69];
  const fingerTop = [20,12,17,27];
  const fingers = fingerX.map((x,i) => `<rect x="${x}" y="${raised.includes(i) ? fingerTop[i] : 51}" width="8" height="${raised.includes(i) ? 64-fingerTop[i] : 13}" rx="4" fill="${raised.includes(i) ? '#164bff' : '#b8cafa'}"/>`).join("");
  const target = {6:3,7:2,8:1,9:0}[number];
  const thumb = target === undefined
    ? (number === 3 || number === 5 ? '<path d="M33 70 Q12 62 18 48" fill="none" stroke="#164bff" stroke-width="9" stroke-linecap="round"/>' : '<path d="M33 72 Q24 72 22 64" fill="none" stroke="#b8cafa" stroke-width="9" stroke-linecap="round"/>')
    : `<path d="M32 75 Q15 62 ${fingerX[target]+4} 56" fill="none" stroke="#164bff" stroke-width="8" stroke-linecap="round"/><circle cx="${fingerX[target]+4}" cy="56" r="5" fill="#164bff"/>`;
  return `<svg viewBox="0 0 104 108" aria-hidden="true" focusable="false"><rect x="30" y="61" width="53" height="39" rx="13" fill="#dfe8ff"/>${fingers}${thumb}</svg>`;
}

function fingerPatternIcon(pattern) {
  const raised = [...pattern.slice(1)].map(value => value === "1");
  const xs = [36, 47, 58, 69];
  const tops = [20, 12, 17, 27];
  const fingers = xs.map((x, finger) => `<rect x="${x}" y="${raised[finger] ? tops[finger] : 51}" width="8" height="${raised[finger] ? 64 - tops[finger] : 13}" rx="4" fill="${raised[finger] ? '#164bff' : '#b8cafa'}"/>`).join("");
  const thumb = pattern[0] === "1" ? '<path d="M33 70 Q12 62 18 48" fill="none" stroke="#164bff" stroke-width="9" stroke-linecap="round"/>' : '<path d="M33 72 Q24 72 22 64" fill="none" stroke="#b8cafa" stroke-width="9" stroke-linecap="round"/>';
  return `<svg viewBox="0 0 104 108" aria-hidden="true" focusable="false"><rect x="30" y="61" width="53" height="39" rx="13" fill="#dfe8ff"/>${fingers}${thumb}</svg>`;
}

function fingerPatternKey(pattern) {
  const key = document.createElement("span");
  key.className = "finger-key";
  key.setAttribute("aria-hidden", "true");
  for (let i = 0; i < fingerNames.length; i++) {
    const finger = document.createElement("span");
    finger.className = pattern[i] === "1" ? "raised" : "folded";
    finger.textContent = fingerNames[i][0];
    finger.title = fingerNames[i];
    key.append(finger);
  }
  return key;
}

function renderGuide() {
  const asl = recognitionMode !== "gestures";
  const letters = recognitionMode === "asl_letters";
  const guide = letters ? aslLetterGuide : asl ? aslGuide : gestureGuide;
  const container = $("gestureGuide");
  container.replaceChildren();
  guide.forEach(([symbol, word, shape], index) => {
    const card = document.createElement("div");
    const icon = document.createElement("span");
    icon.className = "guide-icon";
    if (asl && !letters) icon.innerHTML = aslHandIcon(index + 1);
    else if (!asl && index >= 16) icon.innerHTML = fingerPatternIcon(extraFingerPatterns[index - 16]);
    else icon.textContent = symbol;
    icon.setAttribute("aria-hidden", "true");
    const label = document.createElement("b"); label.textContent = word;
    const detail = document.createElement("small"); detail.textContent = shape;
    card.append(icon);
    if (!asl && index >= 16) {
      card.className = "finger-combination";
      card.append(fingerPatternKey(extraFingerPatterns[index - 16]));
    }
    card.append(label, detail);
    if (asl) {
      const source = document.createElement("a");
      source.href = letters ? "https://www.lifeprint.com/asl101/fingerspelling/abc.htm" : "https://www.lifeprint.com/asl101/pages-signs/n/numbers1-10.htm";
      source.target = "_blank";
      source.rel = "noopener noreferrer";
      source.textContent = "See sign ↗";
      card.append(source);
    }
    container.append(card);
  });
  $("guideEyebrow").textContent = letters ? "ASL LETTER GUIDE" : asl ? "ASL NUMBER GUIDE" : "GESTURE GUIDE";
  $("guideTitle").textContent = letters ? "ASL alphabet · A–Z" : asl ? "ASL numbers 1–9" : "26 gestures to try";
  $("workspaceEyebrow").textContent = letters ? "ASL LETTERS → TEXT" : asl ? "ASL NUMBERS → WORDS" : "GESTURES → WORDS";
  $("modeHint").textContent = letters ? "Spell one letter at a time. Move your hand for J and Z." : asl ? "Show a number handshape from the guide and hold still." : "Show one or two hands and hold still.";
  $("resultHint").textContent = letters ? "Show an ASL letter from the guide." : asl ? "Show an ASL number from the guide." : "Show a gesture from the guide.";
  ui.modeLabel.textContent = asl ? "ASL → ENGLISH" : "ENGLISH";
  $("aslSubmode").hidden = !asl;
  $("aslLettersMode").textContent = "Letters · A–Z";
  $("wordBuilderActions").hidden = !letters;
  $("historyEmptyHint").textContent = letters ? "Spell letters to build a word here." : "Recognized gestures will appear here in order.";
  $("aslNumbersMode").setAttribute("aria-pressed", String(recognitionMode === "asl"));
  $("aslLettersMode").setAttribute("aria-pressed", String(letters));
}

function setRecognitionMode(mode) {
  if (mode === recognitionMode || busy || recorder?.state === "recording") return;
  historyByMode[recognitionMode] = historyWords;
  recognitionMode = mode;
  historyWords = historyByMode[mode];
  $("gesturesMode").setAttribute("aria-pressed", String(mode === "gestures"));
  $("aslMode").setAttribute("aria-pressed", String(mode !== "gestures"));
  renderGuide();
  renderHistory();
  liveSession++;
  liveHistory = [];
  lastCommittedGesture = null;
  clearResult();
  if (stream) startLiveRecognition();
  refreshModelStatus();
}

renderGuide();

function setResultMode(mode) {
  const history = mode === "history";
  ui.livePanel.hidden = history;
  ui.historyPanel.hidden = !history;
  ui.liveTab.setAttribute("aria-selected", String(!history));
  ui.historyTab.setAttribute("aria-selected", String(history));
}

function renderHistory() {
  const localize = window.sign2textI18n?.translate || (word => word);
  ui.historyText.textContent = recognitionMode === "asl_letters"
    ? historyWords.join("")
    : historyWords.map(localize).join(" ");
  ui.historyEmpty.hidden = historyWords.length > 0;
  ui.historyOutput.hidden = historyWords.length === 0;
}
window.sign2textRefreshLocale = renderHistory;

function addToHistory(gesture, word) {
  if (!gesture || !word || gesture === lastCommittedGesture) return;
  if (recognitionMode === "asl_letters" && !/^[A-Z]$/.test(word)) return;
  historyWords.push(word);
  lastCommittedGesture = gesture;
  renderHistory();
}

function updateControls() {
  const recording = recorder?.state === "recording";
  ui.camera.disabled = busy || mediaBusy || recording;
  ui.camera.textContent = stream ? "Turn camera off" : "Enable camera";
  ui.record.disabled = busy || mediaBusy || !stream;
  $("uploadButton").disabled = busy || mediaBusy || recording;
  $("translateButton").disabled = busy || mediaBusy || recording || !!stream || !currentBlob || !modelReady;
  $("saveButton").disabled = !currentBlob || recording;
  document.querySelectorAll("[data-demo]").forEach(button => button.disabled = busy || recording);
}

function clearResult() {
  ui.emptyResult.hidden = false;
  ui.output.hidden = true;
  ui.required.hidden = true;
  ui.translation.textContent = "";
}

function stopCamera() {
  window.clearInterval(analysisId);
  window.clearInterval(liveId);
  liveSession++;
  liveBusy = false;
  liveHistory = [];
  lastCommittedGesture = null;
  lastObservedAt = 0;
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  ui.video.srcObject = null;
  ui.stage.classList.remove("live");
}

function reviewClip(blob) {
  stopCamera();
  releaseObjectUrl();
  currentBlob = blob;
  objectUrl = URL.createObjectURL(blob);
  ui.video.src = objectUrl;
  ui.video.controls = true;
  ui.video.muted = true;
  ui.stage.classList.add("playback");
  ui.empty.hidden = true;
  ui.video.load();
  clearResult();
  updateControls();
}

const modelAdapter = window.sign2textModel;
async function refreshModelStatus() {
  window.clearTimeout(statusTimer);
  try {
    const status = await modelAdapter.status();
    modelReady = status.installed && status.runtimeReady && (recognitionMode !== "asl_letters" || (status.aslLettersInstalled && status.aslLettersRuntimeReady));
    ui.state.classList.toggle("loaded", modelReady);
    ui.state.lastChild.textContent = !status.runtimeReady ? " Setup required" : recognitionMode === "asl_letters" && (!status.aslLettersInstalled || !status.aslLettersRuntimeReady) ? " ASL letters setup needed" : status.installed ? " Ready" : ` Setting up${status.download ? ` · ${status.download.percent}%` : ""}`;
  } catch {
    modelReady = false;
    ui.state.classList.remove("loaded");
    ui.state.lastChild.textContent = " Disconnected";
  } finally {
    updateControls();
    statusTimer = window.setTimeout(refreshModelStatus, 15000);
  }
}
refreshModelStatus();

function toast(message) {
  window.clearTimeout(toastTimer);
  ui.toast.textContent = message;
  ui.toast.classList.add("show");
  toastTimer = window.setTimeout(() => ui.toast.classList.remove("show"), 4000);
}

function formatTime(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function selectedMode() {
  return recognitionMode;
}

async function enableCamera() {
  if (busy || mediaBusy) return;
  if (stream) {
    stopCamera();
    clearResult();
    ui.empty.hidden = !!currentBlob;
    if (currentBlob) reviewClip(currentBlob);
    updateControls();
    return;
  }
  mediaBusy = true;
  updateControls();
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera access requires a supported browser on localhost.");
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false,
    });
    ui.video.removeAttribute("src");
    ui.video.srcObject = stream;
    ui.video.controls = false;
    ui.video.muted = true;
    await ui.video.play();
    ui.empty.hidden = true;
    ui.stage.classList.add("live");
    ui.stage.classList.remove("playback");
    startAnalysis();
    startLiveRecognition();
    $("clipStatus").textContent = "Live recognition is on.";
  } catch (error) {
    stopCamera();
    if (currentBlob) reviewClip(currentBlob);
    else ui.empty.hidden = false;
    toast(error.name === "NotAllowedError" ? "Camera permission was not granted" : "Camera is unavailable");
  } finally {
    mediaBusy = false;
    updateControls();
  }
}

function startLiveRecognition() {
  window.clearInterval(liveId);
  const session = ++liveSession;
  liveHistory = [];
  liveSeenAt = 0;
  liveErrorShown = false;
  clearResult();
  const tick = async () => {
    if (!stream || !modelReady || ui.video.readyState < 2 || liveBusy) return;
    liveBusy = true;
    try {
      const width = ui.video.videoWidth || 640;
      const height = ui.video.videoHeight || 360;
      ui.liveCanvas.width = 960;
      ui.liveCanvas.height = Math.max(1, Math.round(960 * height / width));
      ui.liveCanvas.getContext("2d").drawImage(ui.video, 0, 0, ui.liveCanvas.width, ui.liveCanvas.height);
      const frame = await new Promise(resolve => ui.liveCanvas.toBlob(resolve, "image/jpeg", .82));
      if (!frame || session !== liveSession) return;
      const result = await modelAdapter.live(frame, recognitionMode);
      if (session !== liveSession) return;
      liveHistory.push(result.gesture && result.text ? result : null);
      if (liveHistory.length > 5) liveHistory.shift();
      const matches = result.gesture ? liveHistory.filter(item => item?.gesture === result.gesture) : [];
      if (result.handPresent !== false) lastObservedAt = Date.now();
      else if (Date.now() - lastObservedAt > 900) lastCommittedGesture = null;
      if (matches.length >= 3) {
        liveSeenAt = Date.now();
        const scores = matches.map(item => item.confidence).filter(score => typeof score === "number");
        const confidence = scores.length ? scores.reduce((sum, score) => sum + score, 0) / scores.length : null;
        showResult(result.text, confidence, null, modelAdapter.id);
        addToHistory(result.gesture, result.text);
      } else if (Date.now() - liveSeenAt > 1200 && !ui.output.hidden) {
        clearResult();
      }
    } catch (error) {
      if (session === liveSession && !liveErrorShown) {
        liveErrorShown = true;
        toast("Live recognition is unavailable. Check the local server.");
        refreshModelStatus();
      }
    } finally {
      if (session === liveSession) liveBusy = false;
    }
  };
  liveId = window.setInterval(tick, 300);
  tick();
}

function startAnalysis() {
  window.clearInterval(analysisId);
  samples = [];
  previousLuma = null;
  const ctx = ui.canvas.getContext("2d", { willReadFrequently: true });
  const tick = () => {
    if (!stream || ui.video.readyState < 2) return;
    ctx.drawImage(ui.video, 0, 0, ui.canvas.width, ui.canvas.height);
    const pixels = ctx.getImageData(0, 0, ui.canvas.width, ui.canvas.height).data;
    let total = 0;
    let squared = 0;
    let motion = 0;
    const luma = new Uint8Array(pixels.length / 16);
    let n = 0;
    for (let i = 0; i < pixels.length; i += 16) {
      const value = Math.round(.2126 * pixels[i] + .7152 * pixels[i + 1] + .0722 * pixels[i + 2]);
      luma[n] = value;
      total += value;
      squared += value * value;
      if (previousLuma) motion += Math.abs(value - previousLuma[n]);
      n += 1;
    }
    const mean = total / n;
    const contrast = Math.sqrt(Math.max(0, squared / n - mean * mean));
    samples.push({ light: mean / 255, contrast: Math.min(1, contrast / 70), motion: previousLuma ? Math.min(1, motion / n / 22) : 0 });
    if (samples.length > 80) samples.shift();
    previousLuma = luma;
  };
  analysisId = window.setInterval(tick, 250);
}

function aggregateQuality(duration) {
  const average = (key) => samples.length ? samples.reduce((sum, x) => sum + x[key], 0) / samples.length : 0;
  const lightRaw = average("light");
  return {
    light: Math.max(0, 1 - Math.abs(lightRaw - .52) / .52),
    contrast: average("contrast"),
    motion: Math.min(1, average("motion") * 2.2),
    duration: Math.min(1, duration / 3, 8 / Math.max(duration, .1)),
    seconds: duration,
  };
}

function renderQuality(quality) {
  const metrics = [
    ["light", "lightMetric", "lightBar"], ["contrast", "contrastMetric", "contrastBar"],
    ["motion", "motionMetric", "motionBar"], ["duration", "durationMetric", "durationBar"],
  ];
  metrics.forEach(([key, textId, barId]) => {
    if (quality[key] == null) {
      $(textId).textContent = "—";
      $(barId).style.width = "0";
      return;
    }
    const value = Math.round(quality[key] * 100);
    $(textId).textContent = key === "duration" ? `${quality.seconds.toFixed(1)}s` : `${value}%`;
    $(barId).style.width = `${value}%`;
    $(barId).style.background = value < 35 ? "var(--danger)" : value < 60 ? "var(--warn)" : "var(--accent)";
  });
  const weakest = Math.min(quality.light, quality.contrast, quality.motion, quality.duration);
  ui.qualityState.textContent = quality.light == null ? "Not measured" : weakest < .35 ? "Try again" : weakest < .6 ? "Usable" : "Good";
}

function startRecording() {
  if (!stream || busy) return;
  if (!window.MediaRecorder) { toast("Recording is unavailable in this browser. Choose a video instead."); return; }
  $("translateButton").disabled = true;
  $("uploadButton").disabled = true;
  ui.timer.textContent = "0:00";
  try {
  const mimeType = ["video/webm;codecs=vp8", "video/webm", "video/mp4"].find(type => MediaRecorder.isTypeSupported(type));
  recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  chunks = [];
  samples = [];
  previousLuma = null;
  startedAt = performance.now();
  recorder.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
  recorder.onstop = finishRecording;
  recorder.start(250);
  currentBlob = null;
  releaseObjectUrl();
  clearResult();
  updateControls();
  $("clipStatus").textContent = "Recording… stops automatically at 10 seconds.";
  ui.record.querySelector("b").textContent = "Stop";
  ui.badge.classList.add("show");
  timerId = window.setInterval(() => {
    const elapsed = (performance.now() - startedAt) / 1000;
    ui.timer.textContent = formatTime(elapsed);
    if (elapsed >= 10) stopRecording();
  }, 200);
  recorder.onerror = () => {
    toast("Recording failed. Try again or choose a video.");
    stopRecording();
  };
  } catch (error) {
    recorder = null;
    toast("This browser could not start recording. Choose a video instead.");
    updateControls();
  }
}

function stopRecording() {
  if (recorder?.state === "recording") recorder.stop();
}

async function finishRecording() {
  window.clearInterval(timerId);
  const duration = (performance.now() - startedAt) / 1000;
  currentBlob = new Blob(chunks, { type: recorder.mimeType || "video/webm" });
  chunks = [];
  ui.record.querySelector("b").textContent = "Record again";
  ui.badge.classList.remove("show");
  renderQuality(aggregateQuality(duration));
  if (!currentBlob.size || duration < 1 || currentBlob.size > 32*1024*1024) {
    currentBlob = null;
    $("clipStatus").textContent = "Record a clip of 1–10 seconds, smaller than 32 MB.";
    updateControls();
    return;
  }
  reviewClip(currentBlob);
  $("clipStatus").textContent = "Recording ready to review or save.";
}

async function runTranslation() {
  if (busy || mediaBusy || stream || !currentBlob || !modelReady || recorder?.state === "recording") return;
  busy = true;
  $("translateButton").disabled = true;
  $("uploadButton").disabled = true;
  ui.record.disabled = true;
  ui.camera.disabled = true;
  document.querySelectorAll("[data-demo]").forEach(button => button.disabled = true);
  ui.emptyResult.hidden = true;
  ui.output.hidden = true;
  ui.required.hidden = true;
  const mode = selectedMode();
  ui.required.hidden = false;
  ui.required.querySelector("strong").textContent = "Reading your signs…";
  ui.required.querySelector("p").textContent = "This may take a moment.";
  try {
    const start = performance.now();
    const result = await modelAdapter.translate({ blob: currentBlob, mode, onProgress: message => {
      ui.required.querySelector("p").textContent = message?.includes("Reconnecting") ? "Reconnecting…" : "This may take a moment.";
    } });
    showResult(result.text, result.confidence, result.latencyMs ?? performance.now() - start, result.model || modelAdapter.id || "local", false);
    addToHistory(result.gesture, result.text);
    $("modelWarning").hidden = true;
    $("clipStatus").textContent = "Done.";
  } catch (error) {
    $("clipStatus").textContent = "Translation failed. Your clip is still available to save or retry.";
    ui.required.hidden = false;
    ui.required.querySelector("strong").textContent = "Couldn’t read this recording.";
    ui.required.querySelector("p").textContent = error.message || "Check the model adapter and try again.";
  } finally {
    busy = false;
    updateControls();
    refreshModelStatus();
  }
}

function showResult(text, confidence = null, latency = null, model = "fixture", isDemo = false) {
  $("exampleLabel").hidden = !isDemo;
  $("modelWarning").hidden = true;
  ui.emptyResult.hidden = true;
  ui.required.hidden = true;
  ui.output.hidden = false;
  ui.translation.textContent = text;
  ui.confidence.textContent = confidence == null ? "Not scored" : `${Math.round(confidence * 100)}%`;
  ui.confidenceBar.style.width = confidence == null ? "0" : `${confidence * 100}%`;
  ui.latency.textContent = latency == null ? "N/A" : `${(latency / 1000).toFixed(1)} s`;
  ui.model.textContent = isDemo ? "Labelled fixture" : model;
}

function releaseObjectUrl() {
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null;
}

ui.camera.addEventListener("click", enableCamera);
$("uploadButton").addEventListener("click", () => ui.file.click());
$("translateButton").addEventListener("click", runTranslation);
ui.record.addEventListener("click", () => recorder?.state === "recording" ? stopRecording() : startRecording());
ui.file.addEventListener("change", async () => {
  const file = ui.file.files[0];
  ui.file.value = "";
  if (!file) return;
  if (busy || recorder?.state === "recording") return;
  if (file.size > 32 * 1024 * 1024) { toast("Choose a clip smaller than 32 MB"); return; }
  if (!file.size || (file.type && !file.type.startsWith("video/"))) { toast("Choose a video file"); return; }
  renderQuality({light:null, contrast:null, motion:null, duration:null});
  mediaBusy = true;
  reviewClip(file);
  $("clipStatus").textContent = "Checking video…";
  window.clearTimeout(mediaTimer);
  mediaTimer = window.setTimeout(() => {
    if (!mediaBusy || ui.video.srcObject) return;
    mediaBusy = false;
    currentBlob = null;
    $("clipStatus").textContent = "Video could not be opened. Try an MP4 or WebM clip.";
    updateControls();
  }, 15000);
});
ui.video.addEventListener("loadedmetadata", () => {
  if (!mediaBusy || ui.video.srcObject) return;
  window.clearTimeout(mediaTimer);
  mediaBusy = false;
  const seconds = ui.video.duration;
  if (Number.isFinite(seconds) && (seconds < 1 || seconds > 10.5)) {
    currentBlob = null;
    $("clipStatus").textContent = "Choose a video of 1–10 seconds.";
  } else {
    renderQuality({light:null, contrast:null, motion:null, duration:Number.isFinite(seconds) ? Math.min(1, seconds/3,8/seconds) : null, seconds});
    $("clipStatus").textContent = "Video ready to review or save.";
  }
  updateControls();
});
ui.video.addEventListener("error", () => {
  if (!currentBlob || stream) return;
  window.clearTimeout(mediaTimer);
  mediaBusy = false;
  currentBlob = null;
  $("clipStatus").textContent = "This video cannot be played. Choose an MP4 or WebM clip.";
  updateControls();
});
$("saveButton").addEventListener("click", () => {
  if (!currentBlob || !objectUrl) return;
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = currentBlob.name || `sign2text-recording.${currentBlob.type.includes("mp4") ? "mp4" : "webm"}`;
  link.click();
});
ui.copy.addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(ui.translation.textContent); toast("Text copied"); }
  catch { toast("Copy unavailable. Select the result text to copy it."); }
});
ui.liveTab.addEventListener("click", () => setResultMode("live"));
ui.historyTab.addEventListener("click", () => setResultMode("history"));
$("gesturesMode").addEventListener("click", () => setRecognitionMode("gestures"));
$("aslMode").addEventListener("click", () => setRecognitionMode("asl"));
$("aslNumbersMode").addEventListener("click", () => setRecognitionMode("asl"));
$("aslLettersMode").addEventListener("click", () => setRecognitionMode("asl_letters"));
$("copyHistoryButton").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(ui.historyText.textContent); toast("All words copied"); }
  catch { toast("Copy unavailable. Select the words to copy them."); }
});
$("clearHistoryButton").addEventListener("click", () => {
  historyWords = [];
  lastCommittedGesture = null;
  renderHistory();
});
$("wordSpaceButton").addEventListener("click", () => {
  if (recognitionMode !== "asl_letters" || !historyWords.length || historyWords.at(-1) === " ") return;
  historyWords.push(" ");
  renderHistory();
});
$("wordRepeatButton").addEventListener("click", () => {
  if (recognitionMode !== "asl_letters" || !/^[A-Z]$/.test(historyWords.at(-1) || "")) return;
  historyWords.push(historyWords.at(-1));
  renderHistory();
});
$("wordBackspaceButton").addEventListener("click", () => {
  if (recognitionMode !== "asl_letters") return;
  historyWords.pop();
  lastCommittedGesture = null;
  renderHistory();
});
document.querySelectorAll("[data-demo]").forEach((button) => {
  button.addEventListener("click", () => showResult(button.dataset.demo, null, null, "fixture", true));
});
window.addEventListener("beforeunload", () => {
  if (stream) stream.getTracks().forEach((track) => track.stop());
  releaseObjectUrl();
});
