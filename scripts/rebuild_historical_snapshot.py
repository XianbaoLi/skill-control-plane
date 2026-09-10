"""Read-only source recovery into a fresh temporary historical snapshot."""
import argparse
from pathlib import Path
from skill_control_plane.corpus.rebuild import rebuild


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('local_artifacts/corpora/hermes-local-v0.1/manifest.json'))
    parser.add_argument('--live',type=Path,default=Path('/mnt/d/Hermes/skills'))
    parser.add_argument('--git-repo',type=Path,default=Path('/mnt/d/Hermes/hermes-agent'))
    parser.add_argument('--old-frozen',type=Path,default=Path('local_artifacts/v0.5/hermes-frozen-87'))
    parser.add_argument('--output-parent',type=Path,default=Path('local_artifacts/v0.5'))
    args=parser.parse_args()
    result=rebuild(args.manifest,args.live,args.git_repo,args.old_frozen,args.output_parent,
                   card_paths=tuple(sorted(Path('local_artifacts/v0.5').glob('retrieval-cards*.jsonl'))))
    print('Snapshot:',result['output'])
    print('Audit:',result['audit'])
    print('Restored:',result['restored_by_source'])
    print('Full audit:',Path(result['output'])/'rebuild-audit.json')


if __name__=='__main__':
    main()
