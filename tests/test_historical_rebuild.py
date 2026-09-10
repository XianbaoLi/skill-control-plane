from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from skill_control_plane.corpus import build_corpus_manifest
from skill_control_plane.corpus.rebuild import audit_cards, audit_tree, rebuild, validate_manifest
from skill_control_plane.registry import load_skill_tree
from skill_control_plane.registry.loader import decode_skill_bytes, skill_content_hash


@pytest.mark.parametrize('data', [b'line\nnext\n', b'line\r\nnext\r\n', b'line\rnext\r',
    b'\xef\xbb\xbfline\r\n', b'bad \xff text\r\n', b' trailing \n\n'])
def test_hash_matches_original_registry_read_text(tmp_path, data):
    path=tmp_path/'sample'/'SKILL.md'; path.parent.mkdir(); path.write_bytes(data)
    original=path.read_text(encoding='utf-8-sig',errors='replace')
    assert decode_skill_bytes(data)==original
    assert skill_content_hash(data)==hashlib.sha256(original.encode('utf-8')).hexdigest()
    assert load_skill_tree(tmp_path)[0].content_hash==skill_content_hash(data)


def put(root, rel, name, text):
    path=root/rel; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(f'---\nname: {name}\n---\n{text}\n'.encode())
    return path


def git(repo, *args):
    return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.STDOUT)


def fixture_sources(tmp_path):
    historical=tmp_path/'historical'
    a=put(historical,'old category/a/SKILL.md','a','one')
    put(historical,'original/b/SKILL.md','b','historical b')
    manifest=build_corpus_manifest(historical,source='test')
    mp=tmp_path/'manifest.json';mp.write_text(json.dumps(manifest))
    live=tmp_path/'live';live.mkdir()
    put(live,'moved/a/SKILL.md','a','one').write_bytes(a.read_bytes().replace(b'\n',b'\r\n'))
    put(live,'new/b/SKILL.md','b','changed')
    put(live,'new/github/SKILL.md','github','extra')
    repo=tmp_path/'repo';repo.mkdir();git(repo,'init','-q')
    git(repo,'config','user.name','Test');git(repo,'config','user.email','test@example.invalid')
    put(repo,'skills/old path/b/SKILL.md','b','historical b')
    git(repo,'add','.');git(repo,'commit','-qm','historical')
    put(repo,'skills/old path/b/SKILL.md','b','changed')
    git(repo,'add','.');git(repo,'commit','-qm','changed')
    old=tmp_path/'old';old.mkdir()
    return mp,live,repo,old,tmp_path/'outputs',manifest


def test_rebuild_live_crlf_git_history_moved_paths_extra_and_no_overwrite(tmp_path):
    mp,live,repo,old,parent,manifest=fixture_sources(tmp_path)
    before=(git(repo,'status','--porcelain'),git(repo,'rev-parse','HEAD'))
    result=rebuild(mp,live,repo,old,parent,expected_count=2)
    output=Path(result['output'])
    assert result['audit']['passed']
    assert result['live_normalized_matches']==1 and result['live_raw_matches']==0
    assert result['restored_by_source']=={'live':1,'git':1}
    assert result['audit']['snapshot_id']==manifest['snapshot_id']
    assert (output/'old category/a/SKILL.md').read_bytes()==(live/'moved/a/SKILL.md').read_bytes()
    assert result['live_extra_skill_ids']==['github']
    assert all(result['sources_unchanged'].values())
    assert before==(git(repo,'status','--porcelain'),git(repo,'rev-parse','HEAD'))
    assert list(old.iterdir())==[]
    second=rebuild(mp,live,repo,old,parent,expected_count=2)
    assert second['output']!=result['output']
    assert audit_tree(output,manifest)['passed']


def test_missing_history_fails_before_output_creation(tmp_path):
    mp,live,repo,old,parent,manifest=fixture_sources(tmp_path)
    # An additional unreachable content version with the same ID must not be substituted.
    put(tmp_path/'historical','original/b/SKILL.md','b','not in history')
    mp.write_text(json.dumps(build_corpus_manifest(tmp_path/'historical',source='test')))
    with pytest.raises(ValueError,match='Historical content not found'):
        rebuild(mp,live,repo,old,parent,expected_count=2)
    assert not parent.exists()


def test_manifest_path_traversal_duplicate_id_and_count_rejected(tmp_path):
    _,_,_,_,_,manifest=fixture_sources(tmp_path)
    with pytest.raises(ValueError,match='count'):validate_manifest(manifest,87)
    bad=json.loads(json.dumps(manifest));bad['skills'][1]['skill_id']='a'
    with pytest.raises(ValueError,match='Duplicate'):validate_manifest(bad,2)
    bad=json.loads(json.dumps(manifest));bad['skills'][0]['relative_path']='../escape/SKILL.md'
    with pytest.raises(ValueError,match='Unsafe'):validate_manifest(bad,2)


def test_refuse_output_under_source_and_detect_stale_representation(tmp_path):
    mp,live,repo,old,_,manifest=fixture_sources(tmp_path)
    with pytest.raises(ValueError,match='protected'):
        rebuild(mp,live,repo,old,live/'output',expected_count=2)
    expected={r['skill_id']:r for r in manifest['skills']}
    cards=tmp_path/'cards.jsonl'
    cards.write_text('\n'.join(json.dumps(dict(skill_id=sid,source_content_hash=r['content_hash'])) for sid,r in expected.items()))
    assert audit_cards(cards,expected)['one_to_one_historical']
    rows=[json.loads(l) for l in cards.read_text().splitlines()];rows[0]['source_content_hash']='wrong'
    cards.write_text('\n'.join(map(json.dumps,rows)))
    audit=audit_cards(cards,expected)
    assert not audit['one_to_one_historical'] and len(audit['source_hash_mismatch'])==1
