"""분할 파일 세 개를 합치고 원본 best.pt의 SHA-256을 검증한다."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
expected = json.loads((root / 'results/environment.json').read_text())['best_pt_sha256']
parts = [root / 'weights' / f'best.pt.part{i:02d}' for i in range(1, 4)]
for part in parts:
    if not part.is_file():
        raise FileNotFoundError(f'가중치 분할 파일이 없습니다: {part}')

temporary = root / 'weights/best.pt.tmp'
digest = hashlib.sha256()
with temporary.open('wb') as output:
    for part in parts:
        with part.open('rb') as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
if digest.hexdigest() != expected:
    temporary.unlink()
    raise ValueError('SHA-256이 원본 가중치와 다릅니다. 분할 파일을 다시 확인하십시오.')
target = root / 'weights/best.pt'
temporary.replace(target)
print(f'가중치 복원 완료: {target}')
print(f'SHA-256: {digest.hexdigest()}')
