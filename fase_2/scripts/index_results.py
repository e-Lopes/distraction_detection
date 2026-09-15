"""Gera uma galeria local dos relatórios e figuras, sem executar experimentos."""
from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
from urllib.parse import quote


def generate_index(results: Path) -> Path:
    results = results.resolve()
    results.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[Path]] = {}
    for path in sorted(results.rglob('*')):
        if path.is_file() and path.suffix.lower() in {'.png', '.svg', '.jpg', '.jpeg', '.md', '.txt', '.pdf'}:
            if path.name == 'README.md':
                continue
            relative = path.relative_to(results)
            group = '/'.join(relative.parts[:2]) if len(relative.parts) > 2 else relative.parts[0]
            groups.setdefault(group, []).append(path)

    def url(path):
        return quote(path.relative_to(results).as_posix(), safe='/')

    sections = []
    for group, paths in groups.items():
        reports = [p for p in paths if p.suffix.lower() in {'.md', '.txt', '.pdf'}]
        images = [p for p in paths if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.svg'}]
        # PNG e SVG são versões da mesma figura; mostrar uma miniatura e ambos os downloads.
        image_set = set(images)
        images = [p for p in images if p.suffix.lower() != '.svg' or p.with_suffix('.png') not in image_set]
        cards = []
        for path in images:
            svg = path.with_suffix('.svg')
            extra = f' · <a href="{url(svg)}">SVG</a>' if svg in image_set and svg != path else ''
            cards.append(f'<figure data-search="{escape((group + " " + path.stem).lower(), quote=True)}">'
                f'<a href="{url(path)}"><img loading="lazy" src="{url(path)}" alt="{escape(path.stem)}"></a>'
                f'<figcaption>{escape(path.stem)}<br><a href="{url(path)}">Abrir {path.suffix[1:].upper()}</a>{extra}</figcaption></figure>')
        links = ''.join(f'<li><a href="{url(p)}">{escape(p.name)}</a></li>' for p in reports)
        experiment = group.split('/')[-1]
        metrics = results.parent / 'outputs' / ('metrics' if group.startswith('historical/') else '') / experiment
        metrics_link = ''
        if metrics.is_dir():
            import os
            relative_metrics = quote(Path(os.path.relpath(metrics, results)).as_posix(), safe='/')
            metrics_link = f'<p><a href="{relative_metrics}/">Arquivos de execução e métricas</a></p>'
        sections.append(f'<section data-group="{escape(group)}"><h2>{escape(group)}</h2>'
            f'<p>{len(images)} figuras · {len(reports)} relatórios</p>{metrics_link}<ul>{links}</ul>'
            f'<div class="gallery">{"".join(cards)}</div></section>')
    options = ''.join(f'<option value="{escape(g)}">{escape(g)}</option>' for g in groups)
    document = '''<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Resultados · Fase 2</title>
<style>body{font:16px system-ui;background:#f3f5f8;color:#182536;margin:0;padding:28px;max-width:1500px;margin:auto}
h1{margin-bottom:8px}a{color:#135ca1}header{background:white;padding:24px;border-radius:12px}
input,select{font:inherit;padding:10px;margin:8px 12px 0 0;max-width:100%}.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}
figure{margin:0;padding:12px;background:white;border:1px solid #dbe2ea;border-radius:10px}img{width:100%;height:220px;object-fit:contain}
figcaption{overflow-wrap:anywhere;font-size:14px;padding-top:10px}section{margin:32px 0}[hidden]{display:none!important}</style>
<header><h1>Resultados da fase 2</h1><p>Relatórios e gráficos por experimento. A presença de um arquivo não certifica a conclusão ou validade do experimento.</p>
<p>current: experimentos atuais · historical: histórico · audits: auditorias · summaries: resumos.</p>
<label>Experimento <select id="group"><option value="">Todos</option>OPTIONS</select></label>
<label>Buscar figura <input id="search" type="search" placeholder="F1, matriz, vídeo…"></label>
<p>Atualize este índice: <code>python -m fase_2 results</code></p></header>SECTIONS
<script>function filter(){const g=document.getElementById('group').value,q=document.getElementById('search').value.toLowerCase();
document.querySelectorAll('section').forEach(s=>{s.hidden=Boolean(g&&s.dataset.group!==g);
s.querySelectorAll('figure').forEach(f=>f.hidden=!f.dataset.search.includes(q));});}
document.getElementById('group').addEventListener('change',filter);document.getElementById('search').addEventListener('input',filter);</script></html>'''
    target = results / 'index.html'
    target.write_text(document.replace('OPTIONS', options).replace('SECTIONS', ''.join(sections)), encoding='utf-8')
    return target


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] / 'results')
    args = parser.parse_args(arguments)
    print(generate_index(args.root))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
