from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

from fase_2.scripts.index_results import generate_index


def test_gallery_links_escape_names_and_group_png_svg(tmp_path):
    folder = tmp_path / 'historical' / 'G4' / 'figures'
    folder.mkdir(parents=True)
    for name in ('F1 & recall.png', 'F1 & recall.svg'):
        (folder / name).write_bytes(b'figure')
    (folder.parent / 'report.md').write_text('result')
    target = generate_index(tmp_path)
    content = target.read_text(encoding='utf-8')
    assert content.count('<figure ') == 1
    assert 'F1 &amp; recall' in content
    assert 'historical/G4' in content

    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in {'href', 'src'}:
                    assert (target.parent / unquote(value)).exists()
    Links().feed(content)


def test_empty_gallery_is_valid_and_repeatable(tmp_path):
    target = generate_index(tmp_path / 'results')
    first = target.read_bytes()
    assert generate_index(target.parent).read_bytes() == first
    assert b'<figure ' not in first
