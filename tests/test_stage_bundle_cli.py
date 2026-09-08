from skill_control_plane.cli import _build_parser, _print_stage_bundle_report


def test_stage_bundle_cli_parses_expected_defaults() -> None:
    args = _build_parser().parse_args(
        [
            "eval",
            "stage-bundle",
            "/skills",
            "--gold",
            "gold.jsonl",
            "--manifest",
            "manifest.json",
        ]
    )

    assert args.command == "eval"
    assert args.eval_command == "stage-bundle"
    assert args.root == "/skills"
    assert args.k == 5
    assert args.max_bundles == 4
    assert args.max_skills_per_bundle == 4
    assert args.as_json is False


def test_stage_bundle_report_surfaces_shelf_metrics(capsys) -> None:
    _print_stage_bundle_report(
        {
            "case_count": 1,
            "stage_count": 2,
            "k_per_retriever": 5,
            "max_bundles": 4,
            "max_skills_per_bundle": 4,
            "initial_shelf_future_skill_recall": 0.5,
            "initial_shelf_future_skill_hits": 1,
            "future_transition_skill_count": 2,
            "initial_shelf_future_bundle_recall": 1.0,
            "initial_shelf_future_bundle_hits": 1,
            "future_transition_bundle_count": 1,
            "shelf_reuse_rate": 0.5,
            "new_bundle_rate": 0.5,
            "mean_active_required_recall": 0.75,
            "mean_shelf_required_recall": 1.0,
            "cases": [
                {
                    "case_id": "ST-X",
                    "initial_shelf_future_skill_recall": 0.5,
                    "initial_shelf_future_skill_hits": ["future-skill"],
                    "initial_shelf_future_bundle_recall": 1.0,
                    "initial_shelf_future_bundle_hits": ["runtime"],
                    "shelf_reuse_rate": 0.5,
                    "new_bundle_rate": 0.5,
                    "stages": [
                        {
                            "stage_id": "S1",
                            "active_bundle_ids": ["testing"],
                            "bundle_ids": ["testing", "runtime"],
                            "registered_skill_count": 2,
                            "shelf_required_recall": 1.0,
                        }
                    ],
                }
            ],
        }
    )

    output = capsys.readouterr().out
    assert "initial_shelf_future_bundle_recall: 1.0000 (1/1)" in output
    assert "shelf_reuse_rate: 0.5000" in output
    assert "active=['testing']" in output
