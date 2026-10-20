"""Interrupted improver work must never become a successful harness decision."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from harness2 import config
from harness2.cli import run_sweep
from harness2.core import proposer, recursion, tree
from harness2.core.proposer import Proposer, ProposerError, Reply
from harness2.core.spec import resolve_channel_model
from harness2.demo import DemoAdapter


def event(kind, message="final", **fields):
    return json.dumps({"type": kind, "part": {"messageID": message, **fields}})


def completed(text="<notes>KEEP_BASE: no supported edit.</notes>"):
    return "\n".join([event("text", text=text), event("step_finish", reason="stop", cost=0.25)])


class CompletionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.prop = Proposer(
            resolve_channel_model("opencode", "amazon-bedrock/test"),
            audit_dir=self.root / "audit",
            attempts=1,
        )

    def test_only_final_message_is_returned(self):
        stream = "\n".join(
            [
                event("text", "earlier", text="I will inspect the candidates."),
                event("step_finish", "earlier", reason="tool-calls"),
                completed(),
            ]
        )
        text, error = proposer._opencode_stdout(stream)
        self.assertEqual(error, "")
        self.assertNotIn("inspect", text)
        self.assertTrue(text.startswith("<notes>"))

    def test_incomplete_finish_reasons_are_rejected(self):
        for reason in ("tool-calls", "length", "error", "unknown", ""):
            with self.subTest(reason=reason):
                stream = (
                    event("text", text="I will decide later.")
                    + "\n"
                    + event("step_finish", reason=reason)
                )
                text, error = proposer._opencode_stdout(stream)
                self.assertEqual(text, "")
                self.assertTrue(error)

    def test_missing_or_later_unfinished_step_is_rejected(self):
        for stream in (
            "not a JSON event",
            event("text", text="progress"),
            completed() + "\n" + event("step_start", "next"),
            completed() + "\n" + event("text", "next", text="more work"),
            completed() + '\n{"type":',
        ):
            with self.subTest(stream=stream):
                text, error = proposer._opencode_stdout(stream)
                self.assertEqual(text, "")
                self.assertTrue(error)

    def test_error_overrides_prior_success(self):
        stream = completed() + '\n{"type":"error","error":{"data":{"message":"failed"}}}'
        self.assertEqual(proposer._opencode_stdout(stream), ("", "failed"))

    def test_missing_message_identity_is_rejected(self):
        text, error = proposer._opencode_stdout(
            '{"type":"text","part":{"text":"done"}}\n'
            '{"type":"step_finish","part":{"reason":"stop"}}'
        )
        self.assertEqual(text, "")
        self.assertTrue(error)

    def test_answer_file_cannot_hide_failure(self):
        for code, stream in (
            (1, completed()),
            (0, event("text", text="working")),
            (0, completed() + '\n{"type":"error","error":"failed"}'),
        ):
            with self.subTest(code=code, stream=stream), tempfile.TemporaryDirectory() as tmp:

                def run(*args, **kwargs):
                    (Path(tmp) / "output/answer.md").write_text("<notes>KEEP_BASE</notes>")
                    return Mock(returncode=code, stdout=stream, stderr="failure")

                with (
                    patch.object(proposer, "_opencode_dir", return_value=self.root),
                    patch.object(proposer.subprocess, "run", side_effect=run),
                ):
                    text, _, error = self.prop._opencode(Path(tmp), "mission", "", 5)
                self.assertEqual(text, "")
                self.assertTrue(error)

    def test_successful_final_stream_is_accepted(self):
        with (
            patch.object(proposer, "_opencode_dir", return_value=self.root),
            patch.object(
                proposer.subprocess,
                "run",
                return_value=Mock(returncode=0, stdout=completed(), stderr=""),
            ),
        ):
            text, cost, error = self.prop._opencode(self.root, "mission", "", 5)
        self.assertEqual(error, "")
        self.assertEqual(cost, 0.25)
        self.assertTrue(text.startswith("<notes>"))

    def test_final_text_wins_over_an_earlier_answer_file(self):
        def run(*args, **kwargs):
            (self.root / "output/answer.md").write_text("obsolete draft")
            return Mock(returncode=0, stdout=completed("final decision"), stderr="")

        with (
            patch.object(proposer, "_opencode_dir", return_value=self.root),
            patch.object(proposer.subprocess, "run", side_effect=run),
        ):
            text, _, error = self.prop._opencode(self.root, "mission", "", 5)
        self.assertEqual((text, error), ("final decision", ""))

    def test_successful_call_writes_reusable_completion_metadata(self):
        with patch.object(Proposer, "_opencode", return_value=("final answer", 0.5, "")) as run:
            first = self.prop.call("mission", None, "agg1")
            second = self.prop.call("mission", None, "agg1")
        self.assertTrue(first.ok)
        self.assertTrue(second.cached)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(second.text, first.text)

    def test_error_with_partial_text_never_becomes_reply(self):
        with patch.object(Proposer, "_opencode", return_value=("partial text", 0.5, "failed")):
            with self.assertRaises(ProposerError) as failure:
                self.prop.call("mission", None, "agg1")
        self.assertEqual(failure.exception.cost_usd, 0.5)

    def test_nonzero_claude_exit_rejects_partial_text(self):
        with patch.object(
            proposer.subprocess,
            "run",
            return_value=Mock(
                returncode=1,
                stdout='{"result":"partial text","subtype":"success"}',
                stderr="failed",
            ),
        ):
            text, _, error = self.prop._claude_cli(self.root, "mission", "", 5)
        self.assertEqual(text, "")
        self.assertTrue(error)

    def test_cache_requires_current_success_metadata(self):
        paths = self.prop._paths("agg1", "key")
        paths["response"].write_text("<notes>KEEP_BASE</notes>")
        valid = {
            "ok": True,
            "completion_version": 1,
            "channel": "opencode",
            "model": self.prop.spec.model,
            "error": "",
        }
        invalid = [
            None,
            {},
            {**valid, "ok": False},
            {**valid, "completion_version": 0},
            {**valid, "error": "failed"},
            {**valid, "model": "other"},
        ]
        for metadata in invalid:
            with self.subTest(metadata=metadata):
                if metadata is not None:
                    paths["meta"].write_text(json.dumps(metadata))
                self.assertIsNone(self.prop._cached("agg1", "key"))
        paths["meta"].write_text(json.dumps(valid))
        self.assertTrue(self.prop._cached("agg1", "key").cached)


class StageDecisionTests(unittest.TestCase):
    def run_stage(self, replies, *, changed=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = DemoAdapter(root)
            base = tree.read_tree(adapter.substrate.surface(), adapter.base_harness())
            prop = Proposer(adapter.spec.channels["proposer"])
            calls = []

            def reply(mission, context_dir, label, **kwargs):
                calls.append(mission)
                if changed:
                    (Path(context_dir) / "harness/guardrails.md").write_text("Verify the totals.\n")
                return Reply(
                    label,
                    replies[min(len(calls) - 1, len(replies) - 1)],
                    True,
                    attempts=1,
                    cost_usd=0.1,
                    workspace=Path(context_dir) / "harness",
                )

            with patch.object(prop, "call", side_effect=reply):
                stage = recursion._run_stage(
                    label="agg1",
                    dim="AGG",
                    job="compose",
                    fallback=base,
                    adapter=adapter,
                    prop=prop,
                    ctx_dir=root / "context",
                    task_id="ledger-1",
                    rollouts=[],
                    harness_dir=adapter.base_harness(),
                    library_dir=None,
                )
            return stage, calls

    def test_progress_without_final_notes_falls_back(self):
        for text in (
            "I will inspect the files.",
            "<notes>unfinished",
            "<notes> </notes>",
            "<notes>one</notes><notes>two</notes>",
        ):
            with self.subTest(text=text):
                stage, calls = self.run_stage([text], changed=True)
                self.assertTrue(stage.fell_back)
                self.assertEqual(len(calls), 2)
                self.assertAlmostEqual(stage.usd, 0.2)

    def test_unchanged_tree_requires_explicit_keep_base(self):
        stage, _ = self.run_stage(["<notes>I will decide later.</notes>"])
        self.assertTrue(stage.fell_back)
        self.assertIn("KEEP_BASE", stage.error)

    def test_explicit_keep_base_is_a_valid_decision(self):
        stage, calls = self.run_stage(["<notes>KEEP_BASE: evidence supports no change.</notes>"])
        self.assertFalse(stage.fell_back)
        self.assertTrue(stage.kept_base)
        self.assertEqual(len(calls), 1)

    def test_invalid_response_retries_with_feedback(self):
        stage, calls = self.run_stage(
            ["still working", "<notes>Added a check because totals were inconsistent.</notes>"],
            changed=True,
        )
        self.assertFalse(stage.fell_back)
        self.assertFalse(stage.kept_base)
        self.assertIn("PREVIOUS ATTEMPT WAS REJECTED", calls[1])
        self.assertEqual(stage.attempts, 2)


class MethodVersionTests(unittest.TestCase):
    def test_old_method_identity_requires_a_new_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = DemoAdapter(root / "demo")
            args = run_sweep._build_parser().parse_args(["--bench", "jb", "--k", "1"])
            stream = ({"accounting": ["ledger-1"]}, [("accounting", "ledger-1")])
            with patch.object(run_sweep, "_stream_for", return_value=stream):
                run_sweep._check_identity(adapter, args, root / "identity")
                path = root / "identity/config.json"
                current = json.loads(path.read_text())
                self.assertEqual(current["method_version"], config.METHOD_VERSION)
                current["method_version"] = 2
                path.write_text(json.dumps(current))
                with self.assertRaisesRegex(ValueError, "method version"):
                    run_sweep._check_identity(adapter, args, root / "identity")
                self.assertEqual(json.loads(path.read_text())["method_version"], 2)


if __name__ == "__main__":
    unittest.main()
