"""Grava o GIF de demonstração do painel (figuras/demo.gif).

    streamlit run app/painel.py --server.headless true --server.port 8599   # num terminal
    python scripts/gravar_gif.py                                          # em outro

Precisa de Playwright (pip install playwright; usa o Chrome instalado) e do
ffmpeg no PATH. Percorre as telas como uma pessoa faria, tirando capturas de
tela em sequência, e monta um GIF de cerca de 25 segundos.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
URL = "http://localhost:8599"
LARGURA, ALTURA = 1280, 800
QUADROS_POR_SEGUNDO = 5


class Gravador:
    """Tira uma captura a cada 1/QUADROS_POR_SEGUNDO segundo enquanto a tela "descansa"."""

    def __init__(self, pagina, pasta: Path) -> None:
        self.pagina, self.pasta, self.n = pagina, pasta, 0

    def quadro(self) -> None:
        self.pagina.screenshot(path=str(self.pasta / f"q{self.n:04d}.png"))
        self.n += 1

    def pausar(self, segundos: float) -> None:
        intervalo = 1 / QUADROS_POR_SEGUNDO
        for _ in range(max(1, round(segundos * QUADROS_POR_SEGUNDO))):
            inicio = time.perf_counter()
            self.quadro()
            time.sleep(max(0.0, intervalo - (time.perf_counter() - inicio)))

    def esperar_tela(self) -> None:
        """Grava enquanto o Streamlit recalcula, até a página ficar parada."""
        self.pagina.wait_for_selector("#page-title", timeout=60000)
        for _ in range(40):
            self.pausar(0.2)
            if self.pagina.locator("[data-testid='stStatusWidget']").count() == 0:
                break
        self.pausar(0.6)


def gravar(pasta: Path) -> int:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        try:
            navegador = p.chromium.launch(channel="chrome")
        except Exception:
            navegador = p.chromium.launch()
        pagina = navegador.new_page(viewport={"width": LARGURA, "height": ALTURA})
        pagina.goto(URL)
        pagina.wait_for_selector("text=Laboratório de amostragem", timeout=90000)
        g = Gravador(pagina, pasta)
        g.esperar_tela()
        g.pausar(2.0)

        # Laboratório: cada técnica, com a animação antes → depois
        for tecnica in ("nenhuma", "smote", "subamostragem", "limiar"):
            pagina.locator(f'[data-tech="{tecnica}"]').click()
            g.pausar(0.6)
            if tecnica in ("smote", "subamostragem"):
                pagina.locator("#replay").click()
                g.pausar(3.0)
            else:
                g.pausar(1.6)

        # Comparativo: ranking por recall, depois por Brier, e a calibração
        pagina.locator('nav a[data-page="comparativo"]').click()
        g.pausar(3.0)
        pagina.select_option("#sort", "brier")
        g.pausar(3.0)
        pagina.locator(".calibration-panel").scroll_into_view_if_needed()
        g.pausar(3.0)

        # Outra base, já no comparativo
        pagina.select_option("#base", "insuficiencia")
        g.esperar_tela()
        g.pausar(3.0)
        navegador.close()
    return g.n


def main() -> int:
    if shutil.which("ffmpeg") is None:
        print("Precisa do ffmpeg no PATH.", file=sys.stderr)
        return 2
    destino = RAIZ / "figuras" / "demo.gif"
    with tempfile.TemporaryDirectory() as tmp:
        pasta = Path(tmp)
        quadros = gravar(pasta)
        entrada = ["-framerate", str(QUADROS_POR_SEGUNDO), "-i", str(pasta / "q%04d.png")]
        escala = "scale=960:-1:flags=lanczos"
        paleta = pasta / "paleta.png"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", *entrada,
             "-vf", f"{escala},palettegen=stats_mode=diff", str(paleta)],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", *entrada, "-i", str(paleta),
             "-lavfi", f"{escala}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5"
             ":diff_mode=rectangle", str(destino)],
            check=True,
        )
    segundos = quadros / QUADROS_POR_SEGUNDO
    tamanho = destino.stat().st_size / 1e6
    print(f"GIF gravado em {destino}: {quadros} quadros, {segundos:.0f} s, {tamanho:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
