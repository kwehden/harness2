"""The JobBench judge's sampling patch: no temperature where a model refuses 0.0."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from harness2 import config, install
from harness2.benchmarks.jobbench import JobBenchAdapter

SAMPLING_PATCH = "job-bench-eval-judge-sampling.diff"

# The lines of the pinned eval/judge.py that the sampling patch rewrites, verbatim, so
# the shipped diff applies to this fixture exactly as it does to a real checkout.
JUDGE_FIXTURE = '''"""Stand-in for the pinned JobBench judge's client plumbing."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


MAX_CHARS_PER_FILE = 200_000


def resolve_api_config(api_base, api_key):
    return api_base, api_key


def get_openai_client(api_base: str, api_key: str):
    from openai import OpenAI

    if not api_key:
        raise ValueError("No judge API key provided. Set JUDGE_API_KEY.")

    return OpenAI(base_url=api_base, api_key=api_key)


def parse_judge_json(content: str) -> tuple[dict, str]:
    return {}, "direct_json"
'''

# What the patched judge sends per rubric, in the shape the upstream call uses.
CREATE_KWARGS = {
    "model": "us.anthropic.claude-haiku-5-5",
    "messages": [{"role": "user", "content": "rubric"}],
    "max_completion_tokens": 32000,
    "temperature": 0.0,
    "timeout": 240,
}


class FakeCompletions:
    def __init__(self, calls):
        self.calls = calls

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return "judged"


class FakeOpenAI:
    """Enough of the OpenAI client to record what a request carries."""

    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=FakeCompletions(self.calls))
        self.models = "models-resource"


def patched_judge(tmp: Path):
    """The fixture with the shipped sampling patch applied, imported as a module."""
    repo = tmp / "job-bench-eval"
    (repo / "eval").mkdir(parents=True)
    judge = repo / "eval" / "judge.py"
    judge.write_text(JUDGE_FIXTURE, encoding="utf-8")
    judge.chmod(0o755)  # the pinned judge is executable; the diff records that mode
    for command in (
        ["git", "init", "-q", "."],
        ["git", "add", "-A"],
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "pin"],
    ):
        subprocess.run(command, cwd=repo, check=True)
    with (
        patch.object(config, "DEPENDENCIES", tmp),
        patch.dict(install.PATCHES, {"job-bench-eval": (SAMPLING_PATCH,)}),
    ):
        install.patch("job-bench-eval")
        install.patch("job-bench-eval")  # a second setup is a no-op, not a failure
    spec = importlib.util.spec_from_file_location(f"judge_fixture_{tmp.name}", judge)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}):
        spec.loader.exec_module(module)
    return module


def judge_request(tmp: Path, **env) -> tuple[dict, object]:
    """One judge request's kwargs under `env`, with the client that sent it."""
    module = patched_judge(tmp)
    with (
        patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}),
        patch.dict("os.environ", {"JUDGE_OMIT_TEMPERATURE": "", **env}, clear=False),
    ):
        client = module.get_openai_client("http://127.0.0.1:4000/v1", "key")
        client.chat.completions.create(**CREATE_KWARGS)
    inner = getattr(client, "_client", client)
    return inner.calls[-1], client


class JudgeSamplingPatchTests(unittest.TestCase):
    """The omission ships as its own patch, so an installed checkout upgrades."""

    def test_job_bench_carries_the_sampling_patch_separately(self):
        self.assertEqual(
            install.PATCHES["job-bench-eval"],
            ("job-bench-eval-judge.diff", SAMPLING_PATCH),
        )
        paths = install.patch_files("job-bench-eval")
        self.assertEqual([path.name for path in paths if not path.is_file()], [])
        diff = paths[-1].read_text(encoding="utf-8")
        self.assertIn("eval/judge.py", diff)
        self.assertIn("JUDGE_OMIT_TEMPERATURE", diff)
        self.assertIn("+class JudgeClientWithoutTemperature:", diff)

    def test_the_sampling_patch_leaves_the_first_patch_reversible(self):
        """Neither patch touches the other's lines, so each stays detectable."""
        first = install.patch_files("job-bench-eval")[0].read_text(encoding="utf-8")
        second = install.patch_files("job-bench-eval")[-1].read_text(encoding="utf-8")
        self.assertIn("                 temperature=0.0,", first)
        removed = [line for line in second.splitlines() if line.startswith("-")]
        self.assertEqual([line for line in removed if "temperature=0.0" in line], [])
        self.assertEqual(
            removed,
            ["--- a/eval/judge.py", "-    return OpenAI(base_url=api_base, api_key=api_key)"],
        )

    def test_the_sampling_patch_is_checksummed_for_release(self):
        assets = json.loads((config.ROOT / "dependencies" / "ASSETS.json").read_text())
        self.assertIn(f"third_party/patches/{SAMPLING_PATCH}", assets)

    def test_the_patched_judge_reverse_checks_as_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "job-bench-eval"
            patched_judge(Path(tmp))
            reverse = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "apply",
                    "--reverse",
                    "--check",
                    str(install.patch_files("job-bench-eval")[-1]),
                ],
                capture_output=True,
            )
            self.assertEqual(reverse.returncode, 0, reverse.stderr.decode())


class JudgeRequestTests(unittest.TestCase):
    """What the judge actually sends, captured at the OpenAI client boundary."""

    def test_the_flag_omits_temperature_and_keeps_every_other_argument(self):
        for flag in ("1", "true", "TRUE", "yes", " 1 "):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as tmp:
                sent, _ = judge_request(Path(tmp), JUDGE_OMIT_TEMPERATURE=flag)
                self.assertNotIn("temperature", sent)
                self.assertEqual(
                    sent, {k: v for k, v in CREATE_KWARGS.items() if k != "temperature"}
                )

    def test_without_the_flag_the_judge_keeps_temperature_zero(self):
        for flag in (None, "", "0", "false", "no", " "):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as tmp:
                env = {} if flag is None else {"JUDGE_OMIT_TEMPERATURE": flag}
                sent, client = judge_request(Path(tmp), **env)
                self.assertEqual(sent["temperature"], 0.0)
                self.assertEqual(sent, CREATE_KWARGS)
                self.assertIsInstance(client, FakeOpenAI)  # no facade in the way

    def test_a_temperature_the_caller_still_passes_never_reaches_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = patched_judge(Path(tmp))
            with (
                patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}),
                patch.dict("os.environ", {"JUDGE_OMIT_TEMPERATURE": "1"}, clear=False),
            ):
                client = module.get_openai_client("http://127.0.0.1:4000/v1", "key")
                client.chat.completions.create(**{**CREATE_KWARGS, "temperature": 1.0})
            self.assertNotIn("temperature", client._client.calls[-1])

    def test_the_facade_passes_other_client_attributes_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = patched_judge(Path(tmp))
            with (
                patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}),
                patch.dict("os.environ", {"JUDGE_OMIT_TEMPERATURE": "1"}, clear=False),
            ):
                client = module.get_openai_client("http://127.0.0.1:4000/v1", "key")
            self.assertEqual(client.models, "models-resource")
            self.assertEqual(client.base_url, "http://127.0.0.1:4000/v1")


class JudgeEnvironmentTests(unittest.TestCase):
    """The adapter decides the route: the flag is set on AWS and empty elsewhere."""

    def judge_env(self, *, on_aws: bool) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "job-bench-eval" / "eval").mkdir(parents=True)
            (root / "job-bench-eval" / "eval" / "judge.py").write_text("", encoding="utf-8")
            task = root / "job-bench-eval" / "dataset" / "main" / "accountant" / "task1"
            task.mkdir(parents=True)
            (task / "RUBRICS.json").write_text('{"rubrics": []}', encoding="utf-8")
            interpreter = root / "python"
            interpreter.write_text("", encoding="utf-8")
            adapter = JobBenchAdapter.__new__(JobBenchAdapter)
            adapter.store = root / "store"
            calls: list[dict] = []

            def record(cmd, **kwargs):
                calls.append(kwargs.get("env", {}))
                return subprocess.CompletedProcess(cmd, 0, "", "")

            with (
                patch.object(config, "DEPENDENCIES", root),
                patch.object(config, "ON_AWS", on_aws),
                patch.object(config, "JB_JUDGE_PYTHON", interpreter),
                patch.object(config, "JB_JUDGE_API_BASE", "http://127.0.0.1:4000/v1"),
                patch.object(config, "JB_JUDGE_API_KEY", "key"),
                patch.object(config, "JB_JUDGE_MODEL", "us.anthropic.claude-haiku-5-5"),
                patch.object(config, "JB_REPORTING_JUDGE_DETAILS", ""),
                patch("harness2.benchmarks.jobbench.subprocess.run", record),
            ):
                self.assertTrue(adapter._run_judge("accountant/task1", "run-a", root / "run"))
            self.assertEqual(len(calls), 1)
            return calls[0]

    def test_the_aws_judge_asks_for_the_omission(self):
        self.assertEqual(self.judge_env(on_aws=True)["JUDGE_OMIT_TEMPERATURE"], "1")

    def test_the_vertex_judge_keeps_the_upstream_temperature(self):
        env = self.judge_env(on_aws=False)
        self.assertEqual(env["JUDGE_OMIT_TEMPERATURE"], "")
        # An empty value reads as unset, which is what the patched judge checks for.
        self.assertEqual(env["JUDGE_OMIT_TEMPERATURE"].strip().lower(), "")


if __name__ == "__main__":
    unittest.main()
