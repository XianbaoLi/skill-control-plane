#!/usr/bin/env python3
"""Optional local completion adapter. No repository/Gold context is passed.

Uses existing Codex login, disables tools/context discovery, runs in an empty
working directory, and rejects any tool-use event. Stdout is the final response.
Audit prompts, provider events and model banner are saved only when --audit-dir
is supplied. Authentication is managed by Codex, never serialized here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-dir', type=Path)
    parser.add_argument('--model', default='gpt-6-astra')
    parser.add_argument('--replay-dir', type=Path, help='Replay exact audited completions without a model call')
    args = parser.parse_args()
    prompt = sys.stdin.read()
    if args.replay_dir:
        key = hashlib.sha256(prompt.encode()).hexdigest()
        data = json.loads((args.replay_dir / f'{key}.json').read_text())
        if data['prompt'] != prompt or data['returncode'] != 0:
            raise ValueError('Replay must match the exact successful prompt')
        if any(e.get('item', {}).get('type') not in {None, 'agent_message', 'reasoning'}
               for e in data['events']):
            raise ValueError('Replay contains non-text events')
        sys.stdout.write(data['completion'])
        return
    with tempfile.TemporaryDirectory(prefix='capability-need-') as directory:
        output = Path(directory) / 'completion.json'
        command = [
            'codex', 'exec', '--model', args.model, '--ignore-user-config', '--ephemeral',
            '--skip-git-repo-check', '-C', directory, '-s', 'read-only',
            '-c', 'suppress_unstable_features_warning=true',
            '-c', 'project_doc_max_bytes=0', '-c', 'web_search="disabled"',
            '-c', 'mcp_servers={}', '-c', 'features.skip_host_skill_discovery=true',
            '-c', 'features.shell_tool=false', '-c', 'features.apps=false',
            '-c', 'features.plugins=false', '-c', 'features.multi_agent=false',
            '-c', 'features.browser_use=false', '-c', 'features.computer_use=false',
            '-c', 'features.image_generation=false', '-c', 'features.view_image=false',
            '-c', 'features.memories=false', '-c', 'features.hooks=false',
            '--json', '-o', str(output), '-',
        ]
        def audit(data):
            if args.audit_dir:
                args.audit_dir.mkdir(parents=True, exist_ok=True)
                key = hashlib.sha256(prompt.encode()).hexdigest()
                (args.audit_dir / f'{key}.json').write_text(json.dumps({
                    'prompt': prompt, 'command': command, **data,
                }, ensure_ascii=False, indent=2) + '\n')
        try:
            process = subprocess.run(command, input=prompt, text=True, capture_output=True,
                                     timeout=110)
        except subprocess.TimeoutExpired as exc:
            def text(value):
                return value.decode(errors='replace') if isinstance(value, bytes) else value
            audit({'error': 'timeout', 'stdout': text(exc.stdout),
                   'provider_log': text(exc.stderr)})
            raise
        audit({'returncode': process.returncode, 'stdout': process.stdout,
               'provider_log': process.stderr})
        events = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
        # This completion experiment accepts only message/reasoning events.
        for event in events:
            item = event.get('item', {})
            if item and item.get('type') not in {'agent_message', 'reasoning'}:
                raise RuntimeError('Completion used a tool; reject potentially contaminated extraction')
        if args.audit_dir:
            args.audit_dir.mkdir(parents=True, exist_ok=True)
            key = hashlib.sha256(prompt.encode()).hexdigest()
            (args.audit_dir / f'{key}.json').write_text(json.dumps({
                'prompt': prompt, 'command': command, 'events': events,
                'provider_log': process.stderr, 'returncode': process.returncode,
                'completion': output.read_text() if output.exists() else None,
            }, ensure_ascii=False, indent=2) + '\n')
        if process.returncode:
            raise RuntimeError('Codex completion failed; see local provider audit')
        sys.stdout.write(output.read_text())


if __name__ == '__main__':
    main()
