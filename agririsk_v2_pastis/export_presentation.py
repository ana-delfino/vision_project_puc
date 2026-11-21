"""
Exporta a apresentação HTML para PDF (o GitHub não renderiza HTML): uma página por slide, com o
tamanho exato de cada slide, texto selecionável e gráficos vetoriais (SVG).
Requer: pip install playwright pypdf && playwright install chromium

    python export_presentation.py            # → docs/AgriRisk_MA_v2_apresentacao.pdf
    python export_presentation.py --png      # também PNGs de pré-visualização (pasta temporária)
"""
import io
import os
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright
from pypdf import PdfReader, PdfWriter

import config as C

SRC = C.ROOT / "docs" / "AgriRisk_MA_v2_apresentacao.html"
PDF = C.ROOT / "docs" / "AgriRisk_MA_v2_apresentacao.pdf"
PREVIEW = Path(os.environ.get("AGRIRISK_PREVIEW_DIR", Path(tempfile.gettempdir()) / "agririsk_slides"))
W = 1366


def main(png=False):
    writer = PdfWriter()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": W, "height": 900}, device_scale_factor=1.5)
        page.goto(SRC.as_uri())
        page.wait_for_load_state("networkidle")
        page.emulate_media(media="screen")          # mantém o tema escuro e os fundos da tela
        n = page.locator("section.slide").count()
        if png:
            PREVIEW.mkdir(parents=True, exist_ok=True)
            for f in PREVIEW.glob("slide_*.png"):
                f.unlink()
        for i in range(n):
            if png:
                page.locator("section.slide").nth(i).screenshot(path=str(PREVIEW / f"slide_{i:02d}.png"))
            # isola o slide i (os outros somem), mede a altura e imprime numa página desse tamanho
            h = page.evaluate("""i => {
                const s = [...document.querySelectorAll('section.slide')];
                s.forEach((el, k) => el.style.display = k === i ? '' : 'none');
                window.scrollTo(0, 0);
                return Math.ceil(document.documentElement.scrollHeight);
            }""", i)
            pdf = page.pdf(width=f"{W}px", height=f"{h + 2}px", print_background=True,
                           margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
            writer.add_page(PdfReader(io.BytesIO(pdf)).pages[0])
        browser.close()
    writer.add_metadata({"/Title": "AgriRisk-MA · apresentação técnica"})
    with open(PDF, "wb") as fh:
        writer.write(fh)
    print(f"ok → {PDF} ({PDF.stat().st_size / 1e6:.1f} MB, {n} páginas)" + (f" · PNGs em {PREVIEW}" if png else ""))


if __name__ == "__main__":
    main(png="--png" in sys.argv)
