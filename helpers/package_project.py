"""Create a portable project ZIP, omitting temporary build files and caches."""
from pathlib import Path
import zipfile
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
archive = ROOT / 'london_airbnb_group_project.zip'
build_files = {'build_deck.mjs', 'main_notes.json', 'slide_data.json',
               'final-validation-v3.json', 'deck-summary.json', 'slide_notes.json'}
files = []
for file in sorted(ROOT.rglob('*')):
    if not file.is_file() or file.is_symlink():
        continue
    relative = file.relative_to(ROOT)
    if any(part in {'__pycache__', 'node_modules', '.ipynb_checkpoints', '.venv'} for part in relative.parts):
        continue
    if file == archive or file.name in {'.DS_Store', 'package_manifest.json'}:
        continue
    if relative.parts[0] == '.build' and file.name not in build_files:
        continue
    if relative.parts[0] == 'slides' and file.name != 'slide-34-render.png':
        continue
    files.append(file)
manifest = {str(file.relative_to(ROOT)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in files}
manifest_file = ROOT / 'package_manifest.json'
manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as writer:
    for file in files + [manifest_file]:
        writer.write(file, Path('final-group-project') / file.relative_to(ROOT))
with zipfile.ZipFile(archive) as reader:
    assert reader.testzip() is None
    for file, expected in manifest.items():
        assert hashlib.sha256(reader.read('final-group-project/' + file)).hexdigest() == expected
print(json.dumps({'archive': str(archive), 'files': len(files) + 1,
                  'size_mb': round(archive.stat().st_size / 1024**2, 1),
                  'all_archived_hashes_verified': True}, indent=2))
