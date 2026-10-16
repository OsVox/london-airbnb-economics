"""Export PPTX with a documented scatter-rendering workaround for LibreOffice.

The source PPTX remains unchanged and fully editable. Page 34 uses the matching
Artifact Tool render because LibreOffice drops scatter series on this deck.
Requires LibreOffice, pypdf and reportlab; see README for the tested runtime.
"""
from pathlib import Path
import argparse
import io
import subprocess
import tempfile
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--soffice', default='soffice')
args = parser.parse_args()
source = ROOT / 'london_airbnb_group_project.pptx'
render = ROOT / 'slides/slide-34-render.png'
assert render.exists(), 'Regenerate the slide renders with the presentation builder.'
with tempfile.TemporaryDirectory() as directory:
    subprocess.run([args.soffice, '--headless', '--convert-to', 'pdf', '--outdir',
                    directory, str(source)], check=True)
    pdf = PdfReader(Path(directory) / source.with_suffix('.pdf').name)
    assert len(pdf.pages) == 54
    page = pdf.pages[33]
    size = (float(page.mediabox.width), float(page.mediabox.height))
    buffer = io.BytesIO()
    canvas = Canvas(buffer, pagesize=size)
    canvas.drawImage(str(render), 0, 0, width=size[0], height=size[1])
    canvas.showPage()
    canvas.save()
    buffer.seek(0)
    replacement = PdfReader(buffer)
    writer = PdfWriter()
    for index, page in enumerate(pdf.pages):
        writer.add_page(replacement.pages[0] if index == 33 else page)
    writer.add_metadata({'/Title': 'London Airbnb — economics group project',
                         '/Subject': '18 main slides and 36 appendix slides; asking prices and review credibility'})
    with source.with_suffix('.pdf').open('wb') as target:
        writer.write(target)
print(source.with_suffix('.pdf'))
