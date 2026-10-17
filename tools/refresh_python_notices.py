"""Inventory installed redistributable runtimes without importing their packages."""
import argparse
import csv
import hashlib
import json
import re
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STREAM = ROOT
RUNTIMES = {
    'Runtime/Python_CUDA': ROOT / 'Runtime/Python_CUDA',
    'Runtime/Python_DirectML': ROOT / 'Runtime/Python_DirectML',
}
LEGAL = re.compile(r'^(licen[cs]e|notice|copying|copyright|authors|third.?party)', re.I)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    blocks, rows = {}, []

    def collect(path, runtime, owner):
        body = path.read_text(encoding='utf-8', errors='replace').strip()
        digest = hashlib.sha256(body.encode()).hexdigest()
        entry = blocks.setdefault(digest, {'text': body, 'sources': set(), 'owners': set()})
        entry['sources'].add(runtime + '/' + path.relative_to(RUNTIMES[runtime]).as_posix())
        entry['owners'].add(owner)

    for runtime, root in RUNTIMES.items():
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in root.glob('*'):
            if path.is_file() and path.suffix != '.meta' and LEGAL.match(path.name):
                collect(path, runtime, 'CPython/runtime libraries')
        site = root / 'Lib/site-packages'
        for info in sorted(site.glob('*.dist-info')):
            meta = Parser().parsestr((info / 'METADATA').read_text(encoding='utf-8'))
            name, version = meta['Name'], meta['Version']
            candidates = set(p for p in info.rglob('*') if p.is_file() and p.suffix != '.meta' and LEGAL.match(p.name))
            record = info / 'RECORD'
            if record.exists():
                for row in csv.reader(record.read_text(encoding='utf-8').splitlines()):
                    if Path(row[0]).suffix == '.meta' or not LEGAL.match(Path(row[0]).name):
                        continue
                    path = (site / row[0]).resolve()
                    if path.is_relative_to(root.resolve()) and path.is_file() and LEGAL.match(path.name):
                        candidates.add(path)
            for path in sorted(candidates):
                collect(path, runtime, f'{name} {version}')
            license_value = meta.get('License-Expression') or meta.get('License') or '; '.join(
                v for v in meta.get_all('Classifier', []) if v.startswith('License ::')) or 'Not declared'
            rows.append({'runtime': runtime, 'name': name, 'version': version,
                         'license_metadata': license_value,
                         'metadata': runtime + '/Lib/site-packages/' + info.name + '/METADATA',
                         'license_files': [runtime + '/' + p.relative_to(root).as_posix() for p in sorted(candidates)]})

    supplemental = (ROOT / 'Licenses/python/supplemental-license-texts.txt').read_text(encoding='utf-8')
    supplemental += '\n\n' + (ROOT / 'Licenses/python/python-fire-license-supplement.txt').read_text(encoding='utf-8')
    notice = ['Python Third Party Notices / Python 第三方许可清单', '',
              'Generated from installed runtime metadata; license declarations are not independent legal verification.',
              '根据实际安装环境生成；元数据声明不等于独立授权认定。完整原文见 PythonThirdPartyLicenseTexts.txt。',
              'Original package licenses remain in their runtime directories. Supplemental texts retain their source URLs.',
              'Non-Python code, model weights, dictionaries and character rights also have separate notices.', '',
              'Supplemental texts cover flatbuffers, jieba, jieba_fast, loguru, sentencepiece, tokenizers and wordsegment.',
              'python-fire 0.1.0: distributor supplement preserves the MIT declaration from the wheel/source release, author attribution, and standard SPDX MIT text.',
              'This supplement is identified as distributor-supplied, not an original upstream LICENSE. No copyright year is inferred.',
              'python-fire is a different distribution from Google Fire; their license texts are not interchangeable.', '',
              f'Installed distribution entries: {len(rows)}', f'Unique license text blocks: {len(blocks)}', '']
    for runtime in RUNTIMES:
        subset = [r for r in rows if r['runtime'] == runtime]
        notice += [runtime, f'Distributions: {len(subset)}', '']
        for row in subset:
            value = row['license_metadata']
            if len(value) > 240 or '\n' in value:
                value = 'Full declaration in METADATA (see path below)'
            notice += [f"{row['name']} {row['version']} — {value}", '  Metadata: ' + row['metadata']]
            notice += ['  License: ' + p for p in row['license_files']]
            if not row['license_files']:
                notice += ['  No license text found in installed distribution; consult supplemental texts and audit report.']
        notice.append('')
    texts = ['Python Third Party License Texts', '',
             'Current installed-runtime license texts. Identical text is deduplicated; original files are retained.',
             'Supplemental upstream/data notices follow the installed-runtime texts.', '']
    for i, entry in enumerate(blocks.values(), 1):
        texts += [f'[{i}]', 'Package(s): ' + ', '.join(sorted(entry['owners'])),
                  'Source file(s):', *sorted(entry['sources']), '', entry['text'], '']
    texts += ['Supplemental upstream and data notices', supplemental]
    outputs = {
        STREAM / 'PythonThirdPartyNotices.txt': '\n'.join(notice) + '\n',
        STREAM / 'PythonThirdPartyLicenseTexts.txt': '\n'.join(texts) + '\n',
        ROOT / 'Licenses/python/python-license-inventory.json': json.dumps(rows, ensure_ascii=False, indent=2) + '\n',
    }
    for path, content in outputs.items():
        if args.check:
            if path.read_text(encoding='utf-8') != content:
                raise RuntimeError(f'Outdated: {path}')
        else:
            path.write_text(content, encoding='utf-8')
    print(json.dumps({'entries': len(rows), 'license_blocks': len(blocks),
                      'without_installed_license': sorted({r['name'] for r in rows if not r['license_files']})}))


if __name__ == '__main__':
    main()
