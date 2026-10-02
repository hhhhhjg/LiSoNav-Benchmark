"""Build a release with explicit inclusion; never dereference external assets."""
from pathlib import Path
import hashlib
import tarfile

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ['README.md','LICENSE','THIRD_PARTY_NOTICES.md',
           'pyproject.toml','.gitignore','lisonav','agents','docs','tools',
           'LiSoNav-Eval/LICENSE','LiSoNav-Eval/assets/README.md',
           'LiSoNav-Eval/assets/ASSET_ATTRIBUTION.md',
           'LiSoNav-Eval/lifelong_navigation_sequences']

def main():
    output = ROOT/'dist'
    output.mkdir(exist_ok=True)
    archive = output/'LiSoNav-Benchmark.tar.gz'
    files = []
    for name in INCLUDE:
        path = ROOT/name
        if not path.exists():
            raise FileNotFoundError(path)
        candidates = sorted(path.rglob('*')) if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.is_symlink():
                raise ValueError(f'Unexpected link in release source: {candidate}')
            if candidate.is_file() and '__pycache__' not in candidate.parts and candidate.suffix not in ['.pyc','.pyo']:
                files.append(candidate)
    with tarfile.open(archive,'w:gz',dereference=False) as tar:
        for path in files:
            tar.add(path,arcname=str(Path('LiSoNav-Benchmark')/path.relative_to(ROOT)),recursive=False)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            name = member.name
            assert not member.issym() and not member.islnk()
            assert '/results/' not in name
            assert '/assets/data/' not in name
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.gz.sha256').write_text(f'{digest}  {archive.name}\n')
    print(f'{archive}\n{len(files)} files, {archive.stat().st_size} bytes\nsha256 {digest}')

if __name__ == '__main__':
    main()
