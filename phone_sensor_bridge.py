import json
import os
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HOST = "0.0.0.0"
PORT = int(os.environ.get("PHONE_SENSOR_PORT", "8765"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LATEST_PATH = os.path.join(BASE_DIR, "phone_motion_latest.json")


COLLECTOR_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>HAR Phone Sensor</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #080d19;
      --panel: #111827;
      --line: #263654;
      --text: #eef2ff;
      --muted: #8090bf;
      --blue: #60a5fa;
      --green: #34d399;
      --amber: #fbbf24;
      --red: #fb7185;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: #080d19;
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      padding: 18px;
    }

    .wrap {
      max-width: 620px;
      margin: 0 auto;
    }

    h1 {
      margin: 8px 0 4px;
      font-size: 30px;
      line-height: 1.05;
      letter-spacing: 0;
    }

    .sub {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
      margin-bottom: 18px;
    }

    .card {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: linear-gradient(135deg, rgba(17, 24, 39, 0.98), rgba(18, 25, 45, 0.98));
      padding: 16px;
      margin-bottom: 14px;
    }

    .status {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.35;
      margin-bottom: 14px;
    }

    .dot {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: var(--amber);
      box-shadow: 0 0 18px rgba(251, 191, 36, 0.35);
      flex: 0 0 auto;
    }

    .dot.live {
      background: var(--green);
      box-shadow: 0 0 18px rgba(52, 211, 153, 0.42);
    }

    .dot.err {
      background: var(--red);
      box-shadow: 0 0 18px rgba(251, 113, 133, 0.48);
    }

    .actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    button {
      appearance: none;
      border: 1px solid #33466f;
      background: #172036;
      color: var(--text);
      border-radius: 8px;
      min-height: 46px;
      padding: 11px 12px;
      font-weight: 800;
      font-size: 15px;
    }

    button.primary {
      background: linear-gradient(135deg, #2563eb, #7c3aed);
      border-color: #6388f8;
    }

    button:disabled {
      opacity: 0.52;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .metric {
      border: 1px solid #21304f;
      border-radius: 10px;
      background: rgba(10, 14, 26, 0.62);
      padding: 12px;
    }

    .value {
      color: var(--blue);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 20px;
      font-weight: 900;
      white-space: nowrap;
    }

    .label {
      color: var(--muted);
      margin-top: 5px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 10px;
      letter-spacing: 1.4px;
      text-transform: uppercase;
    }

    .note {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }

    code {
      background: rgba(96, 165, 250, 0.12);
      border: 1px solid rgba(96, 165, 250, 0.22);
      border-radius: 6px;
      color: #bfdbfe;
      padding: 1px 5px;
      word-break: break-word;
    }
  </style>
</head>
<body>
  <main class="wrap">
    <h1>HAR Phone Sensor</h1>
    <div class="sub">Streams this phone's accelerometer and gyroscope data to the Streamlit model running on your Mac.</div>

    <section class="card">
      <div class="status">
        <span class="dot" id="dot"></span>
        <span id="statusText">Ready. Keep this page open and tap Start.</span>
      </div>
      <div class="actions">
        <button class="primary" id="startBtn">Start Streaming</button>
        <button id="stopBtn" disabled>Stop</button>
      </div>
    </section>

    <section class="card">
      <div class="grid">
        <div class="metric"><div class="value" id="samples">0</div><div class="label">Samples</div></div>
        <div class="metric"><div class="value" id="rate">0 Hz</div><div class="label">Sample Rate</div></div>
        <div class="metric"><div class="value" id="bodyPeak">0.00g</div><div class="label">Body Peak</div></div>
        <div class="metric"><div class="value" id="totalPeak">0.00g</div><div class="label">Impact Peak</div></div>
        <div class="metric"><div class="value" id="jerkPeak">0.0</div><div class="label">Jerk Peak</div></div>
        <div class="metric"><div class="value" id="stillness">0%</div><div class="label">Stillness</div></div>
      </div>
    </section>

    <section class="card note">
      If Chrome says motion sensors are unavailable on this local HTTP page, open
      <code>chrome://flags/#unsafely-treat-insecure-origin-as-secure</code>,
      add this page's origin, enable it, and relaunch Chrome. This is only for local project testing.
    </section>
  </main>

  <script>
    const G = 9.80665;
    const BUFFER_MS = 3200;
    const SEND_MS = 250;
    const MAX_BUFFER = 260;

    const els = {
      dot: document.getElementById("dot"),
      statusText: document.getElementById("statusText"),
      startBtn: document.getElementById("startBtn"),
      stopBtn: document.getElementById("stopBtn"),
      samples: document.getElementById("samples"),
      rate: document.getElementById("rate"),
      bodyPeak: document.getElementById("bodyPeak"),
      totalPeak: document.getElementById("totalPeak"),
      jerkPeak: document.getElementById("jerkPeak"),
      stillness: document.getElementById("stillness"),
    };

    let isRunning = false;
    let buffer = [];
    let lastSent = 0;
    let lastSample = null;
    let gravity = { x: 0, y: 0, z: 0 };
    let peakSinceReset = { body: 0, total: 0, jerk: 0, at: null };

    function setStatus(kind, text) {
      els.dot.className = `dot ${kind === "live" ? "live" : kind === "error" ? "err" : ""}`;
      els.statusText.textContent = text;
    }

    function clamp(value, min, max) {
      if (!Number.isFinite(value)) return 0;
      return Math.max(min, Math.min(max, value));
    }

    function mean(values, key) {
      if (!values.length) return 0;
      return values.reduce((sum, item) => sum + item[key], 0) / values.length;
    }

    function max(values, key) {
      if (!values.length) return 0;
      return values.reduce((current, item) => Math.max(current, item[key]), 0);
    }

    function render(payload) {
      els.samples.textContent = String(payload.samples);
      els.rate.textContent = `${payload.sample_rate_hz.toFixed(0)} Hz`;
      els.bodyPeak.textContent = `${payload.peak_body_g.toFixed(2)}g`;
      els.totalPeak.textContent = `${payload.peak_total_g.toFixed(2)}g`;
      els.jerkPeak.textContent = payload.peak_jerk_gs.toFixed(1);
      els.stillness.textContent = `${Math.round(payload.stillness_score * 100)}%`;
    }

    async function publish(payload) {
      try {
        const response = await fetch("/motion", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        setStatus("live", "Streaming to Mac. Watch the Streamlit prediction update.");
      } catch (error) {
        setStatus("error", `Cannot reach receiver: ${error.message}`);
      }
    }

    function aggregate(now) {
      const cutoff = now - BUFFER_MS;
      buffer = buffer.filter((sample) => sample.t >= cutoff).slice(-MAX_BUFFER);
      const elapsed = buffer.length >= 2 ? (buffer[buffer.length - 1].t - buffer[0].t) / 1000 : 0;
      const rate = elapsed > 0 ? (buffer.length - 1) / elapsed : 0;
      const recent = buffer.filter((sample) => sample.t >= now - 1200);
      const stillSamples = recent.filter((sample) => sample.bodyMag < 0.14 && sample.gyroMag < 0.18).length;
      const stillness = recent.length ? stillSamples / recent.length : 0;
      const windowBodyPeak = max(buffer, "bodyMag");
      const windowTotalPeak = max(buffer, "totalMag");
      const windowJerkPeak = max(buffer, "jerk");
      const impactSamples = buffer.filter((sample) =>
        sample.bodyMag >= 1.45 || sample.totalMag >= 2.45 || sample.jerk >= 10
      );
      const latestImpact = impactSamples.length ? impactSamples[impactSamples.length - 1] : null;

      const payload = {
        source: "phone_bridge",
        running: isRunning,
        timestamp: Date.now(),
        samples: buffer.length,
        sample_rate_hz: rate,
        acc_x: clamp(mean(buffer, "bodyX"), -1, 1),
        acc_y: clamp(mean(buffer, "bodyY"), -1, 1),
        acc_z: clamp(mean(buffer, "bodyZ"), -1, 1),
        gyro_x: clamp(mean(buffer, "gyroX"), -2, 2),
        gyro_y: clamp(mean(buffer, "gyroY"), -2, 2),
        gyro_z: clamp(mean(buffer, "gyroZ"), -2, 2),
        acc_rms_g: Math.sqrt(mean(buffer, "bodyMagSq")),
        gyro_rms: Math.sqrt(mean(buffer, "gyroMagSq")),
        peak_body_g: windowBodyPeak,
        peak_total_g: windowTotalPeak,
        peak_jerk_gs: windowJerkPeak,
        session_peak_body_g: peakSinceReset.body,
        session_peak_total_g: peakSinceReset.total,
        session_peak_jerk_gs: peakSinceReset.jerk,
        stillness_score: stillness,
        seconds_since_impact: latestImpact ? (now - latestImpact.t) / 1000 : null,
      };

      render(payload);
      publish(payload);
    }

    function onMotion(event) {
      if (!isRunning) return;

      const now = performance.now();
      const acc = event.acceleration || {};
      const incl = event.accelerationIncludingGravity || {};

      let bodyX;
      let bodyY;
      let bodyZ;

      if ([acc.x, acc.y, acc.z].some((v) => Number.isFinite(v))) {
        bodyX = (acc.x || 0) / G;
        bodyY = (acc.y || 0) / G;
        bodyZ = (acc.z || 0) / G;
      } else {
        const rawX = incl.x || 0;
        const rawY = incl.y || 0;
        const rawZ = incl.z || 0;
        gravity.x = 0.86 * gravity.x + 0.14 * rawX;
        gravity.y = 0.86 * gravity.y + 0.14 * rawY;
        gravity.z = 0.86 * gravity.z + 0.14 * rawZ;
        bodyX = (rawX - gravity.x) / G;
        bodyY = (rawY - gravity.y) / G;
        bodyZ = (rawZ - gravity.z) / G;
      }

      const rot = event.rotationRate || {};
      const gyroX = ((rot.beta || 0) * Math.PI) / 180;
      const gyroY = ((rot.gamma || 0) * Math.PI) / 180;
      const gyroZ = ((rot.alpha || 0) * Math.PI) / 180;

      const totalX = (incl.x || 0) / G;
      const totalY = (incl.y || 0) / G;
      const totalZ = (incl.z || 0) / G;
      const bodyMag = Math.sqrt(bodyX * bodyX + bodyY * bodyY + bodyZ * bodyZ);
      const totalMag = Math.sqrt(totalX * totalX + totalY * totalY + totalZ * totalZ);
      const gyroMag = Math.sqrt(gyroX * gyroX + gyroY * gyroY + gyroZ * gyroZ);

      let jerk = 0;
      if (lastSample) {
        const dt = Math.max((now - lastSample.t) / 1000, 0.001);
        jerk = Math.abs(bodyMag - lastSample.bodyMag) / dt;
      }

      const sample = {
        t: now,
        bodyX,
        bodyY,
        bodyZ,
        gyroX,
        gyroY,
        gyroZ,
        bodyMag,
        totalMag,
        gyroMag,
        bodyMagSq: bodyMag * bodyMag,
        gyroMagSq: gyroMag * gyroMag,
        jerk,
      };

      buffer.push(sample);
      lastSample = sample;

      if (bodyMag > peakSinceReset.body || totalMag > peakSinceReset.total || jerk > peakSinceReset.jerk) {
        peakSinceReset = {
          body: Math.max(peakSinceReset.body, bodyMag),
          total: Math.max(peakSinceReset.total, totalMag),
          jerk: Math.max(peakSinceReset.jerk, jerk),
          at: Date.now(),
        };
      }

      if (now - lastSent >= SEND_MS) {
        lastSent = now;
        aggregate(now);
      }
    }

    async function start() {
      if (!("DeviceMotionEvent" in window)) {
        setStatus("error", "Device motion is not available in this browser.");
        return;
      }

      try {
        if (typeof DeviceMotionEvent.requestPermission === "function") {
          const permission = await DeviceMotionEvent.requestPermission();
          if (permission !== "granted") {
            setStatus("error", "Motion permission was denied.");
            return;
          }
        }

        isRunning = true;
        buffer = [];
        lastSent = 0;
        lastSample = null;
        gravity = { x: 0, y: 0, z: 0 };
        peakSinceReset = { body: 0, total: 0, jerk: 0, at: null };
        els.startBtn.disabled = true;
        els.stopBtn.disabled = false;
        setStatus("live", "Sensor started. Move with the phone on your body.");
        window.addEventListener("devicemotion", onMotion);
      } catch (error) {
        setStatus("error", error.message || "Could not start motion sensors.");
      }
    }

    function stop() {
      isRunning = false;
      els.startBtn.disabled = false;
      els.stopBtn.disabled = true;
      window.removeEventListener("devicemotion", onMotion);
      aggregate(performance.now());
      setStatus("idle", "Stopped.");
    }

    els.startBtn.addEventListener("click", start);
    els.stopBtn.addEventListener("click", stop);
  </script>
</body>
</html>
"""


class SensorBridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))

    def _send_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def do_OPTIONS(self):
        self._send_headers(204)

    def do_GET(self):
        if self.path.startswith("/latest"):
            payload = read_latest()
            self._send_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return

        self._send_headers(content_type="text/html; charset=utf-8")
        self.wfile.write(COLLECTOR_HTML.encode("utf-8"))

    def do_POST(self):
        if not self.path.startswith("/motion"):
            self._send_headers(404)
            self.wfile.write(b'{"error":"not found"}')
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            payload["received_at"] = time.time()
            atomic_write_json(LATEST_PATH, payload)
            self._send_headers()
            self.wfile.write(b'{"ok":true}')
        except Exception as exc:
            self._send_headers(400)
            self.wfile.write(json.dumps({"ok": False, "error": str(exc)}).encode("utf-8"))


def read_latest():
    if not os.path.exists(LATEST_PATH):
        return {"running": False, "samples": 0, "received_at": None}
    with open(LATEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path, payload):
    directory = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(prefix=".phone_motion_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    server = ThreadingHTTPServer((HOST, PORT), SensorBridgeHandler)
    print(f"Phone sensor bridge running on http://{HOST}:{PORT}")
    print(f"Latest motion file: {LATEST_PATH}")
    server.serve_forever()


if __name__ == "__main__":
    main()
