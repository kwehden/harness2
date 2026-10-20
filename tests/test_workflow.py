# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Focused regressions for the paper's execution, evidence, and adoption contracts."""

import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from harness2 import config, preflight
from harness2.core import context, judge, recursion, schedule, tree
from harness2.core.library import Library
from harness2.core.proposer import Proposer, ProposerError, Reply
from harness2.core.spec import resolve_channel_model
from harness2.demo import BASE_SCRIPT, DemoAdapter, DemoProposer


class HaikuImproverTests(unittest.TestCase):
    def test_haiku_55_uses_adaptive_effort_without_legacy_variant(self):
        for model_id in ("us.anthropic.claude-haiku-5-5", "global.anthropic.claude-haiku-5-5"):
            with self.subTest(model_id=model_id):
                self._check_options(
                    f"amazon-bedrock/{model_id}",
                    {"type": "adaptive", "maxReasoningEffort": "high"},
                    variant=False,
                )

    def test_older_bedrock_claude_keeps_legacy_variant(self):
        self._check_options(
            "amazon-bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            None,
            variant=True,
        )

    def _check_options(self, model: str, expected: dict | None, *, variant: bool):
        """Inspect the request we build for OpenCode without an OpenCode checkout present.

        `_opencode_dir` is stubbed because this asserts what harness2 writes and spawns,
        not that the pinned dependency is installed, which preflight covers.
        """
        proposer = Proposer(resolve_channel_model("opencode", model, effort="high"))
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("harness2.core.proposer._opencode_dir") as opencode_dir,
            patch("harness2.core.proposer.subprocess.run") as run,
        ):
            opencode_dir.return_value = Path(tmp) / "opencode"
            run.return_value = Mock(
                stdout='{"type":"text","part":{"text":"done"}}', stderr="", returncode=0
            )
            proposer._opencode(Path(tmp), "mission", "system", 5)
            oc_config = json.loads((Path(tmp) / "opencode.json").read_text())
            cmd = run.call_args.args[0]
        self.assertEqual("--variant" in cmd, variant)
        if variant:
            self.assertEqual(cmd[-2:], ["--variant", "high"])
            self.assertNotIn("provider", oc_config)
        else:
            self.assertEqual(
                oc_config["provider"]["amazon-bedrock"]["models"][model.split("/", 1)[1]][
                    "options"
                ]["reasoningConfig"],
                expected,
            )


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.adapter = DemoAdapter(self.root)
        self.proposer = DemoProposer(self.adapter.spec.channels["proposer"])

    def run_stream(self, k=2, proposer=None):
        return schedule.run_domain(
            "accounting",
            self.adapter.tasks(),
            k,
            adapter=self.adapter,
            prop=proposer or self.proposer,
            runs_root=self.adapter.spec.runs_root,
            library_root=self.root / "library" / str(k),
        )

    def test_recursion_budget_and_resume(self):
        with patch.dict(config.PROPOSER_MODEL_BY_CHANNEL, {"opencode": "", "claude_cli": ""}):
            unresolved = resolve_channel_model(None, None)
            self.assertEqual(unresolved.model, "")
            with self.assertRaises(ValueError):
                Proposer(unresolved)
            with self.assertRaises(ValueError):
                resolve_channel_model("opencode", "")
        for k in (1, 2):
            with self.subTest(k=k):
                result = self.run_stream(k)
                self.assertEqual((result["done"], result["failed"]), (2, 0))
                for task_id in self.adapter.tasks():
                    summary = json.loads(
                        (
                            self.adapter.spec.runs_root
                            / f"demo-oc-agg{k}"
                            / task_id
                            / "summary.json"
                        ).read_text()
                    )
                    for mode in ("agg", "seq"):
                        ledger = summary["modes"][mode]["ledger"]
                        self.assertEqual((ledger["R"], ledger["P"]), (4 * k, 3 * k))
                        selection = summary["selections"][mode]
                        expected = "Total: 85.15\n" if task_id == "ledger-1" else "Total: 20.25\n"
                        self.assertEqual(
                            self.adapter.deliverable_text(Path(selection["chosen_run_dir"])),
                            expected,
                        )
                executions, calls = self.adapter.execution_count, len(self.proposer._calls)
                resumed = self.run_stream(k)
                self.assertEqual(
                    (self.adapter.execution_count, len(self.proposer._calls)), (executions, calls)
                )
                self.assertEqual(
                    [row["status"] for row in resumed["tasks"]], ["skipped", "skipped"]
                )
        self.assertFalse(
            any(
                reply.label.endswith(".agg1") or reply.label.endswith(".seq1")
                for reply in self.proposer._calls
                if reply.label.startswith("select.")
            )
        )
        work = self.adapter.spec.runs_root / "demo-oc-seq2" / "ledger-1"
        summary_path = self.adapter.spec.runs_root / "demo-oc-agg2/ledger-1/summary.json"
        chosen = json.loads(summary_path.read_text())["selections"]["seq"]
        (work / "harnesses" / chosen["chosen_label"] / "scripts/summarize.py").write_text(
            "changed after selection\n"
        )
        repaired = self.run_stream()
        self.assertEqual([row["status"] for row in repaired["tasks"]], ["ok", "skipped"])
        chosen = json.loads(summary_path.read_text())["selections"]["agg"]
        (Path(chosen["chosen_run_dir"]) / "answer.txt").unlink()
        repaired = self.run_stream()
        self.assertEqual([row["status"] for row in repaired["tasks"]], ["ok", "skipped"])

    def test_complete_evidence_and_library_prefix(self):
        (self.adapter.task_files("ledger-1") / "reference_answer.txt").write_text('"pass_rate": 1')
        self.run_stream()
        work = self.adapter.spec.runs_root / "demo-oc-seq2" / "ledger-1"
        reflection = work / "reflection/seq2"
        required = [
            "source/ledger.csv",
            "rollouts/base1.md",
            "rollouts/c2.md",
            "pool/inv1/harness.diff",
            "decisions/c1.md",
            "adoption.json",
            "base_harness/scripts/summarize.py",
            "harness/scripts/summarize.py",
        ]
        self.assertEqual([name for name in required if not (reflection / name).is_file()], [])
        self.assertTrue((work / "context/inv2/pool/c1/rollout.md").is_file())
        self.assertEqual(list(work.rglob("reference_answer.txt")), [])
        next_context = work.parent / "ledger-2/context/inv1/library/harnesses"
        self.assertEqual([path.name for path in next_context.iterdir()], ["ledger-1"])
        with self.assertRaises(ValueError):
            context.assert_tree_has_no_grader_verdicts({"EXPERIENCE.md": '"pass_rate": 1'})
        previous_sessions = []

        def run_opencode(command, **kwargs):
            database = Path(kwargs["env"]["XDG_DATA_HOME"]) / "opencode.db"
            previous_sessions.append(database.exists())
            database.parent.mkdir(parents=True, exist_ok=True)
            database.write_text("private evidence from this session")
            return Mock(stdout='{"type":"text","part":{"text":"done"}}', stderr="", returncode=0)

        proposer = Proposer(self.proposer.spec)
        with (
            patch("harness2.core.proposer._opencode_dir", return_value=self.root),
            patch("harness2.core.proposer.subprocess.run", side_effect=run_opencode),
        ):
            for index in range(2):
                proposer.call("Read this task only", None, f"isolation{index}")
        self.assertEqual(previous_sessions, [False, False])

    def test_invalid_proposal_retries_with_feedback_and_accumulates_cost(self):
        class RepairProposer(DemoProposer):
            feedback = ""

            def call(self, mission, context_dir, label, **kwargs):
                script = Path(context_dir) / "harness/scripts/summarize.py"
                if label.endswith(".a1"):
                    script.write_text("def broken(:\n")
                    return Reply(
                        label,
                        "invalid edit",
                        True,
                        cost_usd=1,
                        attempts=1,
                        workspace=script.parent.parent,
                    )
                self.feedback = mission
                script.write_text(BASE_SCRIPT)
                reply = super().call(mission, context_dir, label, **kwargs)
                reply.cost_usd = 2
                return reply

        proposer = RepairProposer(self.proposer.spec)
        stage = recursion._run_stage(
            label="inv1",
            dim="INV",
            job="Repair the script",
            adapter=self.adapter,
            prop=proposer,
            ctx_dir=self.root / "context",
            task_id="ledger-1",
            rollouts=[],
            fallback=tree.read_tree(self.adapter.substrate.surface(), self.adapter.base_harness()),
            harness_dir=self.adapter.base_harness(),
            library_dir=None,
        )
        self.assertEqual((stage.fell_back, stage.attempts, stage.usd), (False, 2, 3))
        self.assertIn("PREVIOUS ATTEMPT WAS REJECTED", proposer.feedback)

    def test_no_entry_stays_out_of_library_on_resume(self):
        class NoExperience(DemoProposer):
            def call(self, mission, context_dir, label, **kwargs):
                if label.startswith("experience."):
                    if label.endswith("seq1"):
                        raise ProposerError("reflection failed", attempts=2, cost_usd=2)
                    return Reply(label, "NO_ENTRY", True, cost_usd=1)
                return super().call(mission, context_dir, label, **kwargs)

        proposer = NoExperience(self.proposer.spec)
        self.run_stream(k=1, proposer=proposer)
        self.run_stream(k=1, proposer=proposer)
        for mode in ("agg", "seq"):
            library = Library(
                self.root / "library/1" / mode, "accounting", self.adapter.substrate.surface()
            )
            self.assertEqual(library.entries(), [])
        summary_path = self.adapter.spec.runs_root / "demo-oc-agg1/ledger-1/summary.json"
        selections = json.loads(summary_path.read_text())["selections"]
        self.assertEqual([selections[mode]["usd"] for mode in ("agg", "seq")], [1, 2])

    def test_failed_selection_is_never_judged(self):
        class FailedSelector(DemoProposer):
            def call(self, mission, context_dir, label, **kwargs):
                if label.startswith("select."):
                    raise ProposerError("selection failed", attempts=2, cost_usd=3)
                return super().call(mission, context_dir, label, **kwargs)

        self.run_stream(proposer=FailedSelector(self.proposer.spec))
        summary_path = self.adapter.spec.runs_root / "demo-oc-agg2/ledger-1/summary.json"
        summary = json.loads(summary_path.read_text())
        self.assertEqual([summary["selections"][mode]["usd"] for mode in ("agg", "seq")], [3, 3])
        self.adapter.score = Mock(return_value={"n_passed": 1, "n_criteria": 1})
        rows = judge.judge_selections(
            self.adapter,
            list(summary["selections"].values()),
        )
        self.adapter.score.assert_not_called()
        self.assertEqual((rows[0].graded, rows[0].n_passed), (False, None))


class AwsCredentialPreflightTests(unittest.TestCase):
    """Preflight is the operational credential check; the SDK chain does the resolving."""

    def run_check(self, *, env=None, returncode=0, stderr="", aws="/usr/bin/aws", raises=None):
        calls = []

        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            if raises is not None:
                raise raises
            return Mock(
                returncode=returncode, stdout='{"Arn": "arn:aws:iam::1:user/u"}', stderr=stderr
            )

        printed = io.StringIO()
        with (
            patch.dict(os.environ, env or {}, clear=True),
            patch.object(preflight.shutil, "which", return_value=aws),
            patch.object(preflight.subprocess, "run", side_effect=fake_run),
            redirect_stdout(printed),
        ):
            errors = preflight.aws_credential_errors()
        return errors, calls, printed.getvalue()

    def test_resolved_identity_passes_with_one_bounded_probe(self):
        errors, calls, printed = self.run_check()
        self.assertEqual((errors, printed), ([], ""))
        self.assertEqual(len(calls), 1)
        command, kwargs = calls[0]
        self.assertEqual(
            command,
            [
                "aws",
                "sts",
                "get-caller-identity",
                "--region",
                config.AWS_REGION,
                "--no-cli-pager",
                "--output",
                "json",
            ],
        )
        self.assertEqual(kwargs["timeout"], preflight.STS_TIMEOUT)

    def test_every_chain_source_takes_the_same_path(self):
        """Nothing branches on which source the chain will use: the CLI resolves it."""
        for name, env in (
            ("default profile", {}),
            ("named profile", {"AWS_PROFILE": "bedrock"}),
            ("instance role", {"AWS_EC2_METADATA_DISABLED": "false"}),
            ("task role", {"AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "/v2/creds"}),
            (
                "web identity",
                {
                    "AWS_WEB_IDENTITY_TOKEN_FILE": "/var/run/token",
                    "AWS_ROLE_ARN": "arn:aws:iam::1:role/eks",
                },
            ),
            ("static keys", {"AWS_ACCESS_KEY_ID": "AKIA", "AWS_SECRET_ACCESS_KEY": "s"}),
        ):
            with self.subTest(source=name):
                errors, calls, _ = self.run_check(env=env)
                self.assertEqual(errors, [])
                self.assertEqual(len(calls), 1)
                self.assertNotIn("--profile", calls[0][0])

    def test_unusable_credentials_fail_with_the_cli_reason(self):
        for name, env, stderr in (
            (
                "missing profile",
                {"AWS_PROFILE": "absent"},
                "The config profile (absent) could not be found",
            ),
            ("expired sso", {}, "Error loading SSO Token: Token has expired"),
            ("no source", {}, "Unable to locate credentials"),
        ):
            with self.subTest(case=name):
                errors, calls, _ = self.run_check(env=env, returncode=255, stderr=stderr)
                self.assertEqual(len(errors), 1)
                self.assertIn("resolves no usable identity", errors[0])
                self.assertIn(stderr.splitlines()[-1][:40], errors[0])
                self.assertEqual(len(calls), 1)

    def test_a_hung_probe_is_reported_not_awaited_forever(self):
        errors, _, _ = self.run_check(
            raises=subprocess.TimeoutExpired(cmd="aws", timeout=preflight.STS_TIMEOUT)
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("TimeoutExpired", errors[0])

    def test_missing_cli_is_reported_as_a_check_gap(self):
        errors, calls, _ = self.run_check(aws=None)
        self.assertEqual(calls, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("aws CLI missing", errors[0])

    def test_bearer_token_warns_and_never_blocks(self):
        """A Bedrock API key has no STS identity, and OpenCode prefers it over the chain."""
        for name, env in (
            ("key alone", {"AWS_BEARER_TOKEN_BEDROCK": "key"}),
            ("key beside a profile", {"AWS_BEARER_TOKEN_BEDROCK": "key", "AWS_PROFILE": "p"}),
        ):
            with self.subTest(case=name):
                errors, calls, printed = self.run_check(env=env)
                self.assertEqual((errors, calls), ([], []))
                self.assertIn("WARNING", printed)
                self.assertIn("unverified", printed)
                self.assertIn("no AWS identity", printed)


if __name__ == "__main__":
    unittest.main()
