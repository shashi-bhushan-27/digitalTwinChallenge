"""Capture dashboard screenshots for the README and slides (headless Chrome + DevTools protocol).

    uv run streamlit run dashboard/app.py --server.port 8503      # in another terminal
    uv run python scripts/screenshot_dashboard.py --url http://localhost:8503

Writes docs/img/dashboard_*.png. Clicks are done like a user would (tabs and buttons found by
their visible text), so the screenshots show exactly what the app renders.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
MAIN = "document.querySelector('[data-testid=\"stMainBlockContainer\"]')"


class Page:
    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def call(self, method: str, **params):
        self.n += 1
        mid = self.n
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})

    async def js(self, expr: str):
        r = await self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("value")

    async def wait_for(self, expr: str, timeout: float = 60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if await self.js(expr):
                return True
            await asyncio.sleep(0.5)
        raise TimeoutError(expr)

    async def settle(self, seconds: float = 2.0):
        # wait until Streamlit reports it is no longer running the script
        await self.wait_for("!document.querySelector('[data-testid=\"stStatusWidget\"]')", 120)
        await asyncio.sleep(seconds)

    async def shot(self, name: str, height: int):
        await self.call("Emulation.setDeviceMetricsOverride", width=1440, height=height, deviceScaleFactor=1.25, mobile=False)
        await asyncio.sleep(2.5)
        r = await self.call("Page.captureScreenshot", format="png", captureBeyondViewport=False)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{name}.png").write_bytes(base64.b64decode(r["data"]))
        print(f"wrote docs/img/{name}.png")

    async def shot_element(self, name: str, expr: str, pad: int = 10, trim: bool = False):
        """Screenshot of one element (expr is a JS expression returning it); `trim` cuts blank margins."""
        box = await self.js(f"(() => {{ const el = {expr}; if (!el) return null; const b = el.getBoundingClientRect(); "
                            "return [b.x, b.y, b.width, b.height]; })()")
        if not box:
            raise RuntimeError(f"element for {name} not found")
        x, y, w, h = box
        clip = {"x": max(0, x - pad), "y": max(0, y - pad), "width": w + 2 * pad, "height": h + 2 * pad, "scale": 1}
        r = await self.call("Page.captureScreenshot", format="png", clip=clip, captureBeyondViewport=True)
        raw = base64.b64decode(r["data"])
        if trim:
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            bbox = ImageChops.difference(im, Image.new("RGB", im.size, im.getpixel((im.width - 1, im.height - 1)))).getbbox()
            if bbox:
                m = 16
                im = im.crop((max(0, bbox[0] - m), max(0, bbox[1] - m), min(im.width, bbox[2] + m), min(im.height, bbox[3] + m)))
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            raw = buf.getvalue()
        (OUT / f"{name}.png").write_bytes(raw)
        print(f"wrote docs/img/{name}.png")

    async def mouse_click_text(self, selector: str, text: str):
        """Real mouse click at the element's centre (some widgets ignore element.click())."""
        box = await self.js(f"""(() => {{ let el = [...document.querySelectorAll({json.dumps(selector)})]
            .find(b => b.innerText.trim().includes({json.dumps(text)})); if (!el) return null;
            el = el.querySelector('[role="combobox"]')?.parentElement || el;
            el.scrollIntoView({{block: "center"}}); const r = el.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; }})()""")
        if not box:
            raise RuntimeError(f"no {selector} with text {text!r}")
        for kind in ("mousePressed", "mouseReleased"):
            await self.call("Input.dispatchMouseEvent", type=kind, x=box[0], y=box[1], button="left", clickCount=1)

    async def click_text(self, selector: str, text: str):
        ok = await self.js(f"""(() => {{ const el = [...document.querySelectorAll({json.dumps(selector)})]
            .find(b => b.innerText.trim().includes({json.dumps(text)})); if (el) {{ el.click(); return true; }} return false; }})()""")
        if not ok:
            raise RuntimeError(f"no {selector} with text {text!r}")


async def run(url: str) -> None:
    port = 9227
    profile = tempfile.mkdtemp(prefix="chrome-shot-")
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
                             "--window-size=1440,1000", "--hide-scrollbars", "about:blank"])
    try:
        for _ in range(40):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json").read())
                break
            except OSError:
                await asyncio.sleep(0.5)
        ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
        async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
            p = Page(ws)
            await p.call("Page.enable")
            await p.call("Emulation.setDeviceMetricsOverride", width=1440, height=1000, deviceScaleFactor=1.25, mobile=False)
            await p.call("Page.navigate", url=url)
            await p.wait_for("document.body && document.body.innerText.includes('Why this prediction')", 180)
            await p.settle(3)
            await p.shot("dashboard_patient", 1000)
            await p.shot("dashboard_patient_full", 1500)
            await p.shot_element("panel_why", "[...document.querySelectorAll('.card')].find(e => e.innerText.includes('Why this prediction'))")
            await p.shot_element("panel_hero", "[...document.querySelectorAll('[data-testid=\"stPlotlyChart\"]')].filter(e => e.offsetParent)"
                                               ".sort((a, b) => b.offsetWidth * b.offsetHeight - a.offsetWidth * a.offsetHeight)[0]")
            await p.call("Emulation.setDeviceMetricsOverride", width=1440, height=1000, deviceScaleFactor=1.25, mobile=False)
            await p.click_text('[role="tab"]', "Triage")
            await p.settle(3)
            await p.shot("dashboard_triage", 1000)
            await p.shot_element("panel_triage", MAIN, pad=0, trim=True)
            await p.click_text('[role="tab"]', "About this twin")
            await p.settle(2)
            await p.shot("dashboard_about", 1100)
            # what-if on the stable patient before dinner, where the default change is a dish swap
            await p.call("Emulation.setDeviceMetricsOverride", width=1440, height=1000, deviceScaleFactor=1.25, mobile=False)
            await p.mouse_click_text('[data-testid="stSelectbox"]', "Demo scene")
            await p.wait_for("document.querySelectorAll('[role=\"option\"]').length > 0", 20)
            await p.mouse_click_text('[role="option"]', "2. Stable")
            await p.settle(3)
            await p.click_text('[role="tab"]', "What-if")
            await p.settle(1)
            await p.click_text("button", "Simulate on the twin")
            await p.wait_for("document.body.innerText.includes('not from the prediction model')", 120)
            await p.settle(2)
            await p.shot("dashboard_whatif", 1250)
            await p.shot_element("panel_whatif", MAIN, pad=0, trim=True)
    finally:
        proc.terminate()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8503")
    asyncio.run(run(ap.parse_args().url))


if __name__ == "__main__":
    main()
