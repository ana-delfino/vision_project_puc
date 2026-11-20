"""
Exporta a apresentação para avaliação offline: um PNG por slide (docs/slides/) e um
PDF com um slide por página, montado a partir desses PNGs.  Requer: pip install playwright && playwright install chromium

    python export_presentation.py
"""
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

import config as C

SRC = C.ROOT / "docs" / "AgriRisk_MA_v2_apresentacao.html"
OUT = C.ROOT / "docs" / "slides"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1366, "height": 900}, device_scale_factor=1.5)
        page.goto(SRC.as_uri())
        page.wait_for_load_state("networkidle")
        slides = page.query_selector_all("section.slide")
        for i, s in enumerate(slides):
            s.screenshot(path=str(OUT / f"slide_{i:02d}.png"))
        browser.close()
    # PDF montado a partir dos PNGs: exatamente um slide por página, igual à tela
    pngs = sorted(OUT.glob("slide_*.png"))
    pages = [Image.open(f).convert("RGB") for f in pngs]
    pdf = C.ROOT / "docs" / "AgriRisk_MA_v2_apresentacao.pdf"
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=144)
    print(f"ok → {len(pngs)} PNGs em {OUT} · PDF {pdf} ({pdf.stat().st_size / 1e6:.1f} MB, {len(pages)} páginas)")


if __name__ == "__main__":
    main()
