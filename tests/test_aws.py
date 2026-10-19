"""Amazon Bedrock wiring: settings derived from the environment, never from credentials."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

AWS_ENV = {
    "AWS_PROFILE": "bedrock-test",
    "AWS_REGION": "us-west-2",
    "HARNESS2_SOLVER_MODEL": "amazon-bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "HARNESS2_IMPROVER_MODEL": "amazon-bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "HARNESS2_JUDGE_MODEL": "us.anthropic.claude-opus-4-6-v1",
    "HARNESS2_BRIDGE_URL": "http://127.0.0.1:4000/v1",
    "HARNESS2_BRIDGE_KEY": "bridge-key",
}


def probe(code: str, **overrides) -> dict:
    """Evaluate `code` (which must print JSON) in a fresh interpreter under the given env."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HARNESS2_", "AWS_"))}
    # Set-but-empty reads as unset and stops a local .env from filling the value in.
    try:
        lines = (ROOT / ".env").read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        key = line.removeprefix("export ").partition("=")[0].strip()
        if key and not key.startswith("#"):
            env[key] = ""
    env.update(AWS_ENV)
    env.update(overrides)
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


class BedrockConfigTests(unittest.TestCase):
    def test_bedrock_executor_selects_aws(self):
        got = probe(
            "import json; from harness2 import config as c; c.validate_models(judge=True); "
            "print(json.dumps([c.CLOUD, c.SOLVER_MODEL_CX, c.LAB_JUDGE_MODEL, c.JB_JUDGE_MODEL, "
            "c.JB_JUDGE_API_BASE, c.JB_JUDGE_API_KEY, c.JB_JUDGE_REASONING_EFFORT]))"
        )
        self.assertEqual(
            got,
            [
                "aws",
                "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "bedrock/us.anthropic.claude-opus-4-6-v1",
                "us.anthropic.claude-opus-4-6-v1",
                "http://127.0.0.1:4000/v1",
                "bridge-key",
                "default",
            ],
        )

    def test_bedrock_requires_credentials_source(self):
        got = probe(
            "import json; from harness2 import config as c\n"
            "try:\n    c.validate_models()\n    print(json.dumps(''))\n"
            "except c.ConfigError as e:\n    print(json.dumps(str(e)))",
            AWS_PROFILE="",
        )
        self.assertIn("AWS_PROFILE", got)

    def test_empty_profile_is_unset(self):
        got = probe(
            "import json, os; from harness2 import config as c; "
            "print(json.dumps(['AWS_PROFILE' in os.environ, c.AWS_PROFILE]))",
            AWS_PROFILE="",
        )
        self.assertEqual(got, [False, ""])

    def test_mixed_cloud_configs_are_rejected(self):
        code = (
            "import json; from harness2 import config as c\n"
            "try:\n    c.validate_models()\n    print(json.dumps(''))\n"
            "except c.ConfigError as e:\n    print(json.dumps(str(e)))"
        )
        vertex = "google-vertex/gemini-3.5-flash"
        self.assertIn(
            "amazon-bedrock/",
            probe(code, HARNESS2_CLOUD="AWS", HARNESS2_SOLVER_MODEL=vertex),
        )
        self.assertIn("HARNESS2_CLOUD=aws", probe(code, HARNESS2_CLOUD="gcp"))

    def test_codex_on_aws_disables_web_search(self):
        got = probe(
            "import json, tempfile, pathlib; from harness2 import config, install; "
            "config.CODEX_HOME_DIR = pathlib.Path(tempfile.mkdtemp()); "
            "install.configure_codex(); "
            "text = (config.CODEX_HOME_DIR / 'config.toml').read_text(); "
            "print(json.dumps([text, install.codex_config_error()]))",
            HARNESS2_CODEX_BASE_URL="http://127.0.0.1:4000/v1",
        )
        self.assertIn('web_search = "disabled"', got[0])
        self.assertIsNone(got[1])

    def test_vertex_unchanged_without_bedrock(self):
        got = probe(
            "import json; from harness2 import config as c; "
            "print(json.dumps([c.CLOUD, c.LAB_JUDGE_MODEL, c.JB_JUDGE_MODEL]))",
            HARNESS2_SOLVER_MODEL="google-vertex/gemini-3.5-flash",
            HARNESS2_IMPROVER_MODEL="google-vertex/gemini-3.5-flash",
            HARNESS2_JUDGE_MODEL="gemini-3.5-flash",
        )
        self.assertEqual(got, ["gcp", "vertex/gemini-3.5-flash", "google/gemini-3.5-flash"])

    def test_claude_cli_improver_uses_bedrock(self):
        got = probe(
            "import json; from harness2.core.proposer import Proposer; "
            "from harness2.core.spec import resolve_channel_model; "
            "e = Proposer(resolve_channel_model('claude_cli', "
            "'us.anthropic.claude-opus-4-6-v1')).env(); "
            "print(json.dumps({k: e.get(k) for k in ('CLAUDE_CODE_USE_BEDROCK', "
            "'CLAUDE_CODE_USE_VERTEX', 'AWS_REGION', 'AWS_PROFILE', 'ANTHROPIC_MODEL')}))"
        )
        self.assertEqual(
            got,
            {
                "CLAUDE_CODE_USE_BEDROCK": "1",
                "CLAUDE_CODE_USE_VERTEX": None,
                "AWS_REGION": "us-west-2",
                "AWS_PROFILE": "bedrock-test",
                "ANTHROPIC_MODEL": "us.anthropic.claude-opus-4-6-v1",
            },
        )

    def test_bridge_serves_every_bedrock_model(self):
        got = probe(
            "import json; from harness2.cli import bridge; m = bridge.bedrock_models(); "
            "print(json.dumps([m, bridge.litellm_config(m)]))"
        )
        models, yaml = got
        self.assertEqual(
            models,
            ["us.anthropic.claude-sonnet-4-5-20250929-v1:0", "us.anthropic.claude-opus-4-6-v1"],
        )
        for model in models:
            self.assertIn(f"model: bedrock/invoke/{model}", yaml)
        self.assertIn("aws_profile_name: os.environ/AWS_PROFILE", yaml)
        self.assertNotIn("vertex", yaml)

    def test_workbuddy_routes_use_the_bridge(self):
        got = probe(
            "import json; from harness2 import install; "
            "print(json.dumps(install.workbuddy_model_name('us.anthropic.claude-opus-4-6-v1')))"
        )
        self.assertEqual(got, "us.anthropic.claude-opus-4-6-v1")


if __name__ == "__main__":
    unittest.main()
