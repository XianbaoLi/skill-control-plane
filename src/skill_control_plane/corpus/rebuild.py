"""Rebuild manifest-identified SKILL.md files without mutating any source tree."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from skill_control_plane.corpus.snapshot import build_corpus_manifest, _snapshot_id
from skill_control_plane.registry import load_skill_tree
from skill_control_plane.registry.loader import decode_skill_bytes, skill_content_hash, _split_frontmatter


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def identity(data: bytes, path: str) -> tuple[str, str]:
    frontmatter, _ = _split_frontmatter(decode_skill_bytes(data))
    return str(frontmatter.get('name') or Path(path).parent.name).strip(), skill_content_hash(data)


def validate_manifest(manifest: dict[str, Any], expected_count: int) -> dict[str, dict]:
    rows = manifest['skills']
    ids = [r['skill_id'] for r in rows]
    paths = [r['relative_path'] for r in rows]
    if len(rows) != expected_count or manifest['skill_count'] != expected_count:
        raise ValueError('Manifest skill count does not match expected count')
    if len(set(ids)) != len(ids) or len(set(paths)) != len(paths):
        raise ValueError('Duplicate manifest skill_id or path')
    for row in rows:
        path = Path(row['relative_path'])
        if path.is_absolute() or '..' in path.parts or '\\' in str(path) or path.name != 'SKILL.md':
            raise ValueError(f'Unsafe manifest path: {path}')
        digest = row['content_hash']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid manifest content hash')
    if _snapshot_id(rows) != manifest['snapshot_id']:
        raise ValueError('Manifest snapshot_id does not match its entries')
    return {r['skill_id']: r for r in rows}


def tree_inventory(root: Path) -> list[dict]:
    rows = []
    for s in load_skill_tree(root):
        data = Path(s.source_path).read_bytes()
        if identity(data, s.source_path) != (s.skill_id, s.content_hash):
            raise ValueError(f'Source changed while reading: {s.source_path}')
        rows.append(dict(skill_id=s.skill_id, content_hash=s.content_hash,
                         path=s.source_path, raw_sha256=sha(data)))
    return rows


def audit_tree(root: Path, manifest: dict) -> dict:
    actual = build_corpus_manifest(root, source='historical-rebuild-audit')
    expected = {s['skill_id']: s for s in manifest['skills']}
    observed = {s['skill_id']: s for s in actual['skills']}
    missing = sorted(expected.keys() - observed.keys())
    extra = sorted(observed.keys() - expected.keys())
    mismatch = sorted(k for k in expected.keys() & observed.keys() if expected[k]['content_hash'] != observed[k]['content_hash'])
    paths = sorted(k for k in expected.keys() & observed.keys() if expected[k]['relative_path'] != observed[k]['relative_path'])
    return dict(skill_count=actual['skill_count'], missing_skill_ids=missing, extra_skill_ids=extra,
                content_hash_mismatch=mismatch, duplicate_skill_ids=actual['duplicate_skill_ids'],
                relative_path_mismatch=paths, snapshot_id=actual['snapshot_id'],
                passed=actual['skill_count']==manifest['skill_count'] and not (missing or extra or mismatch or paths or actual['duplicate_skill_ids']) and actual['snapshot_id']==manifest['snapshot_id'])


def git_skill_blobs(repo: Path) -> dict[str, list[str]]:
    """Enumerate both sides of every reachable SKILL.md change, including merges.

    NUL records handle spaces/tabs in paths; --no-renames ensures one path per
    record. Blob IDs are immutable provenance; no checkout or Git writes occur.
    """
    raw = subprocess.check_output(['git','-C',str(repo),'log','--all','--full-history',
        '-m','--root','--format=','--raw','-z','--no-abbrev','--no-renames','--', 'SKILL.md','**/SKILL.md'])
    tokens = iter(raw.split(b'\0'))
    blobs: dict[str, set[str]] = {}
    for token in tokens:
        header = token.lstrip(b'\n')
        if not header:
            continue
        if not header.startswith(b':'):
            raise ValueError('Unexpected git raw log record')
        path = next(tokens).decode('utf-8', errors='surrogateescape')
        if Path(path).name != 'SKILL.md':
            continue
        fields = header.split()
        for mode, oid in ((fields[0][1:],fields[2]),(fields[1],fields[3])):
            if mode not in (b'100644',b'100755') or set(oid)=={ord('0')}:
                continue
            blobs.setdefault(oid.decode('ascii'),set()).add(path)
    return {oid:sorted(paths) for oid,paths in sorted(blobs.items())}


def matching_git_history(repo: Path, expected: dict[str, dict]) -> tuple[dict, int]:
    blobs = git_skill_blobs(repo)
    wanted_hashes = {r['content_hash'] for r in expected.values()}
    matches: dict[str, list[dict]] = {}
    proc = subprocess.Popen(['git','-C',str(repo),'cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    try:
        for oid, paths in blobs.items():
            proc.stdin.write((oid+'\n').encode('ascii')); proc.stdin.flush()
            header = proc.stdout.readline().split()
            if len(header)!=3 or header[1]!=b'blob':
                raise ValueError(f'Expected Git blob: {oid}')
            size = int(header[2]); data = proc.stdout.read(size)
            if len(data)!=size or proc.stdout.read(1)!=b'\n':
                raise ValueError('Truncated git cat-file response')
            if skill_content_hash(data) not in wanted_hashes:
                continue
            for path in paths:
                sid, digest = identity(data,path)
                if sid in expected and digest==expected[sid]['content_hash']:
                    matches.setdefault(sid,[]).append(dict(blob=oid,path=path,raw_sha256=sha(data),content_hash=digest))
    finally:
        proc.stdin.close()
        proc.stdout.close()
        code = proc.wait()
    if code:
        raise RuntimeError(f'git cat-file failed: {code}')
    return matches, len(blobs)


def byte_audit(data: bytes, expected_hash: str) -> dict:
    normalized = decode_skill_bytes(data).encode('utf-8')
    try:
        data.decode('utf-8-sig',errors='strict'); valid_utf8=True
    except UnicodeDecodeError:
        valid_utf8=False
    return dict(raw_sha256=sha(data), registry_sha256=sha(normalized),
        raw_matches=sha(data)==expected_hash, registry_matches=sha(normalized)==expected_hash,
        raw_bytes=len(data), normalized_bytes=len(normalized), utf8_bom=data.startswith(b'\xef\xbb\xbf'),
        valid_utf8=valid_utf8, crlf_count=data.count(b'\r\n'), bare_cr_count=data.count(b'\r')-data.count(b'\r\n'),
        crlf_only_replacement_matches=sha(data.replace(b'\r\n',b'\n'))==expected_hash)


def audit_cards(path: Path, expected: dict[str, dict]) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [r.get('skill_id') for r in rows]
    present = {sid for sid in ids if isinstance(sid,str)}
    stale = sorted({r['skill_id'] for r in rows if r.get('skill_id') in expected and r.get('source_content_hash')!=expected[r['skill_id']]['content_hash']})
    missing = sorted(expected.keys()-present); extra = sorted(present-expected.keys())
    duplicates = [sid for sid,n in Counter(ids).items() if n>1]
    no_hash = [r.get('skill_id') for r in rows if not r.get('source_content_hash')]
    return dict(path=str(path),raw_sha256=sha(path.read_bytes()),row_count=len(rows),
        skill_id_recorded=all(isinstance(sid,str) and sid for sid in ids),source_hash_recorded=not no_hash,
        missing_skill_ids=missing,extra_skill_ids=extra,duplicate_skill_ids=duplicates,
        source_hash_mismatch=stale,matching_historical_rows=sum(r.get('skill_id') in expected and r.get('source_content_hash')==expected[r['skill_id']]['content_hash'] for r in rows),
        one_to_one_historical=len(rows)==len(expected) and not (missing or extra or duplicates or stale or no_hash))


def rebuild(manifest_path: Path, live: Path, repo: Path, old_frozen: Path,
            output_parent: Path, *, expected_count: int = 87, card_paths: tuple[Path,...] = ()) -> dict:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    expected = validate_manifest(manifest,expected_count)
    parent = output_parent.resolve()
    for source in (live,repo,old_frozen):
        if parent.is_relative_to(source.resolve()):
            raise ValueError('Output parent must be outside all protected source trees')
    live_before = tree_inventory(live); old_before = tree_inventory(old_frozen)
    git_head = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()
    refs_before = subprocess.check_output(['git','-C',str(repo),'show-ref'])
    print('Scanning reachable Git SKILL.md blobs (read-only)',flush=True)
    history,blob_count = matching_git_history(repo,expected)
    selected = {}; details = []
    for sid,row in expected.items():
        same = [s for s in live_before if (s['skill_id'],s['content_hash'])==(sid,row['content_hash'])]
        if same:
            source = same[0]; data=Path(source['path']).read_bytes()
            provenance=dict(source='live',path=source['path'])
        elif history.get(sid):
            source=history[sid][0]
            data=subprocess.check_output(['git','-C',str(repo),'cat-file','blob',source['blob']])
            provenance=dict(source='git',**source)
        else:
            raise ValueError(f'Historical content not found for {sid}; no snapshot written')
        if identity(data,row['relative_path'])!=(sid,row['content_hash']):
            raise ValueError(f'Identity mismatch while restoring {sid}')
        selected[sid]=(data,provenance)
        current=[s for s in live_before if s['skill_id']==sid]
        details.append(dict(skill_id=sid,manifest_hash=row['content_hash'],relative_path=row['relative_path'],
            live=[{**s,**byte_audit(Path(s['path']).read_bytes(),row['content_hash'])} for s in current],
            old_frozen=[s for s in old_before if s['skill_id']==sid],git_matches=history.get(sid,[]),
            chosen=provenance,output_raw_sha256=sha(data)))
    parent.mkdir(parents=True,exist_ok=True)
    output=Path(tempfile.mkdtemp(prefix='hermes-frozen-87-rebuild-',dir=parent))
    for sid,(data,_) in selected.items():
        dest=output/expected[sid]['relative_path']; dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(data)
    audit=audit_tree(output,manifest)
    if not audit['passed']:
        raise ValueError(f'Rebuilt tree failed validation: {audit}; retained at {output}')
    unchanged=dict(live=tree_inventory(live)==live_before, old_frozen=tree_inventory(old_frozen)==old_before,
        git_refs=subprocess.check_output(['git','-C',str(repo),'show-ref'])==refs_before,
        git_head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==git_head,
        manifest=manifest_path.read_bytes()==manifest_bytes)
    if not all(unchanged.values()):
        raise ValueError(f'Source drift during rebuild: {unchanged}')
    cards=[audit_cards(p,expected) for p in card_paths]
    result=dict(manifest=str(manifest_path),manifest_raw_sha256=sha(manifest_bytes),output=str(output),
        hash_semantics='SHA256(UTF-8 encoding of utf-8-sig decoded text, errors=replace, universal newlines); no whitespace stripping',
        live_skill_count=len(live_before),live_normalized_matches=sum(any(s['skill_id']==sid and s['content_hash']==r['content_hash'] for s in live_before) for sid,r in expected.items()),
        live_raw_matches=sum(any(s['skill_id']==sid and s['raw_sha256']==r['content_hash'] for s in live_before) for sid,r in expected.items()),
        live_extra_skill_ids=sorted({s['skill_id'] for s in live_before}-expected.keys()),
        old_frozen_audit=audit_tree(old_frozen,manifest),git_head=git_head,git_unique_skill_blobs_scanned=blob_count,
        git_scope='All local refs, full history including merge diffs; no remote fetch or dangling-object search',
        git_shallow=subprocess.check_output(['git','-C',str(repo),'rev-parse','--is-shallow-repository']).decode().strip(),
        restored_by_source=dict(Counter(p['source'] for _,p in selected.values())),audit=audit,
        sources_unchanged=unchanged,skills=details,representations=cards,
        scope='Manifest certifies SKILL.md text only. Supporting scripts/assets are neither hashed by this manifest nor reconstructed; no claim of an executable full historical skill package.',
        e2e_gate='Historical SKILL.md corpus audit passes; enhanced-representation E2E remains blocked unless a hash-matching representation set is supplied/explicitly prepared. Runtime dependencies require separate validation.')
    (output/'rebuild-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result
