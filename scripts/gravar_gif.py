"""Grava o GIF de demonstração do painel (figuras/demo.gif).

    streamlit run app/painel.py --server.headless true --server.port 8599   # num terminal
    python scripts/gravar_gif.py                                          # em outro

Precisa de Playwright (pip install playwright; usa o Chrome instalado) e do
ffmpeg no PATH. Percorre as telas como uma pessoa faria e converte o vídeo em
GIF de cerca de 25 segundos.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
URL = "http://localhost:8599"
LARGURA, ALTURA = 1280, 800


def esperar(pagina) -> None:
    pagina.wait_for_selector("h1", timeout=60000)
    pagina.wait_for_timeout(400)
    for _ in range(40):
        if pagina.locator("[data-testid='stStatusWidget']").count() == 0:
            break
        pagina.wait_for_timeout(250)
    pagina.wait_for_timeout(900)


def gravar(pasta: Path) -> Path:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        try:
            navegador = p.chromium.launch(channel="chrome")
        except Exception:
            navegador = p.chromium.launch()
        contexto = navegador.new_context(
            viewport={"width": LARGURA, "height": ALTURA},
            record_video_dir=str(pasta),
            record_video_size={"width": LARGURA, "height": ALTURA},
        )
        pagina = contexto.new_page()
        pagina.goto(URL)
        esperar(pagina)
        pagina.wait_for_timeout(1500)

        for tecnica in ("Nenhuma correção", "SMOTE", "Subamostragem aleatória", "Ajuste de limiar"):
            pagina.get_by_text(tecnica, exact=True).first.click()
            esperar(pagina)
            pagina.wait_for_timeout(1800)

        pagina.get_by_role("link", name="Comparativo").first.click()
        esperar(pagina)
        pagina.wait_for_timeout(3000)

        pagina.mouse.wheel(0, 350)
        pagina.wait_for_timeout(2500)
        pagina.mouse.wheel(0, -350)
        pagina.wait_for_timeout(800)

        pagina.get_by_role("combobox").first.click()
        pagina.wait_for_timeout(600)
        pagina.get_by_role("option", name="Insuficiência cardíaca").click()
        esperar(pagina)
        pagina.wait_for_timeout(3000)

        video = Path(pagina.video.path())
        contexto.close()
        navegador.close()
    return video


def main() -> int:
    if shutil.which("ffmpeg") is None:
        print("Precisa do ffmpeg no PATH.", file=sys.stderr)
        return 2
    destino = RAIZ / "figuras" / "demo.gif"
    with tempfile.TemporaryDirectory() as tmp:
        video = gravar(Path(tmp))
        paleta = Path(tmp) / "paleta.png"
        filtro = "fps=8,scale=960:-1:flags=lanczos"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", "1.5", "-i", str(video),
             "-vf", f"{filtro},palettegen=stats_mode=diff", str(paleta)],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", "1.5", "-i", str(video),
             "-i", str(paleta), "-lavfi",
             f"{filtro}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
             str(destino)],
            check=True,
        )
    print(f"GIF gravado em {destino} ({destino.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
