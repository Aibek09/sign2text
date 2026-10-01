async function localJSON(url, options = {}, timeoutMs = 30000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {...options, signal: controller.signal});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Local server returned ${response.status}`);
    return data;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("The local server did not respond. Your recording is still available to save.");
    throw error;
  } finally { clearTimeout(timer); }
}

window.sign2textModel = {
  id: "simple-gestures",
  modes: ["video"],
  async status() {
    return localJSON("/api/model");
  },
  async live(frame, mode="gestures") {
    return localJSON("/api/live", {method:"POST", headers:{"Content-Type":"image/jpeg","X-Sign-Mode":mode}, body:frame}, 5000);
  },
  async translate({ blob, mode="gestures", onProgress }) {
    const data = await localJSON("/api/translate", {method:"POST", headers:{"Content-Type":"application/octet-stream","X-Sign-Mode":mode}, body:blob});
    const deadline = Date.now() + 10 * 60 * 1000;
    let connectionFailures = 0;
    while (Date.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, 1200));
      let job;
      try { job = await localJSON(`/api/jobs/${data.jobId}`); connectionFailures = 0; }
      catch (error) {
        if (++connectionFailures >= 3) throw error;
        onProgress?.("Reconnecting to the local server…");
        continue;
      }
      if (job.status === "complete") return job.result;
      if (job.status === "error") throw new Error(job.message);
      onProgress?.(job.message);
    }
    throw new Error("Translation exceeded 10 minutes. The server may still be working; save your video before restarting it.");
  },
};
