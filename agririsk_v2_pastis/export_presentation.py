"""
Pré-visualização da apresentação: um PNG por slide, para conferir o layout. As imagens vão
para uma pasta temporária fora do repositório (ou AGRIRISK_PREVIEW_DIR); não fazem parte da entrega.
Requer: pip install playwright && playwright install chromium

    python export_presentation.py
"""
import os
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

import config as C

SRC = C.ROOT / "docs" / "AgriRisk_MA_v2_apresentacao.html"
OUT = Path(os.environ.get("AGRIRISK_PREVIEW_DIR", Path(tempfile.gettempdir()) / "agririsk_slides"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("slide_*.png"):
        f.unlink()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1366, "height": 900}, device_scale_factor=1.5)
        page.goto(SRC.as_uri())
        page.wait_for_load_state("networkidle")
        slides = page.query_selector_all("section.slide")
        for i, s in enumerate(slides):
            s.screenshot(path=str(OUT / f"slide_{i:02d}.png"))
        browser.close()
    print(f"ok → {len(slides)} PNGs em {OUT}")


if __name__ == "__main__":
    main()
