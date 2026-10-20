"""Amazon Bedrock wiring: settings derived from the environment, never from credentials."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness2 import install

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
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("HARNESS2_", "AWS_", "OPENCODE_"))
    }
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

    def test_model_validation_needs_no_credentials(self):
        """Settings validation is offline: the SDK chain, not this code, finds credentials."""
        with tempfile.TemporaryDirectory() as empty:
            got = probe(
                "import json, os; from harness2 import config as c\n"
                "try:\n    c.validate_models(judge=True)\n    outcome = ''\n"
                "except c.ConfigError as e:\n    outcome = str(e)\n"
                "print(json.dumps([outcome, 'AWS_PROFILE' in os.environ, c.USES_BEDROCK, "
                "os.environ.get('OPENCODE_BEDROCK_CREDENTIAL_CHAIN')]))",
                AWS_PROFILE="",
                HOME=empty,
                AWS_CONFIG_FILE=str(Path(empty) / "none"),
                AWS_SHARED_CREDENTIALS_FILE=str(Path(empty) / "none"),
                AWS_EC2_METADATA_DISABLED="true",
            )
        self.assertEqual(got, ["", False, True, "1"])

    def test_every_chain_source_validates_without_extra_settings(self):
        """A default profile, an ECS task role and an EKS web identity need no allowlisting."""
        code = (
            "import json; from harness2 import config as c\n"
            "try:\n    c.validate_models()\n    print(json.dumps(''))\n"
            "except c.ConfigError as e:\n    print(json.dumps(str(e)))"
        )
        for name, extra in (
            ("default profile", {"AWS_PROFILE": ""}),
            ("instance role", {"AWS_PROFILE": "", "AWS_EC2_METADATA_DISABLED": "false"}),
            ("task role", {"AWS_PROFILE": "", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "/creds"}),
            (
                "web identity",
                {
                    "AWS_PROFILE": "",
                    "AWS_WEB_IDENTITY_TOKEN_FILE": "/var/run/token",
                    "AWS_ROLE_ARN": "arn:aws:iam::1:role/r",
                },
            ),
            ("static keys", {"AWS_PROFILE": "", "AWS_ACCESS_KEY_ID": "AKIA"}),
            ("bedrock api key", {"AWS_PROFILE": "", "AWS_BEARER_TOKEN_BEDROCK": "token"}),
        ):
            with self.subTest(source=name):
                self.assertEqual(probe(code, **extra), "")

    def test_no_credential_allowlist_remains(self):
        got = probe(
            "import json; from harness2 import config as c; "
            "print(json.dumps(hasattr(c, 'aws_credentials_configured')))"
        )
        self.assertFalse(got)

    def test_bedrock_improver_on_vertex_cloud_still_opts_in(self):
        got = probe(
            "import json, os; from harness2 import config as c; "
            "print(json.dumps([c.CLOUD, c.USES_BEDROCK, os.environ.get('AWS_REGION'), "
            "os.environ.get('OPENCODE_BEDROCK_CREDENTIAL_CHAIN')]))",
            HARNESS2_SOLVER_MODEL="google-vertex/gemini-3.5-flash",
            HARNESS2_IMPROVER_MODEL="amazon-bedrock/us.anthropic.claude-haiku-5-5",
            GOOGLE_CLOUD_PROJECT="a-project",
        )
        self.assertEqual(got, ["gcp", True, "us-west-2", "1"])

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
        """No Bedrock role means no AWS settings and no OpenCode provider opt-in."""
        got = probe(
            "import json, os; from harness2 import config as c; "
            "print(json.dumps([c.CLOUD, c.LAB_JUDGE_MODEL, c.JB_JUDGE_MODEL, c.USES_BEDROCK, "
            "'OPENCODE_BEDROCK_CREDENTIAL_CHAIN' in os.environ, os.environ.get('AWS_REGION') "
            "or None]))",
            HARNESS2_SOLVER_MODEL="google-vertex/gemini-3.5-flash",
            HARNESS2_IMPROVER_MODEL="google-vertex/gemini-3.5-flash",
            HARNESS2_JUDGE_MODEL="gemini-3.5-flash",
            AWS_REGION="",
        )
        self.assertEqual(
            got,
            [
                "gcp",
                "vertex/gemini-3.5-flash",
                "google/gemini-3.5-flash",
                False,
                False,
                None,
            ],
        )

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

    def test_bridge_leaves_credentials_to_boto3_without_a_profile(self):
        got = probe(
            "import json; from harness2.cli import bridge; "
            "print(json.dumps(bridge.litellm_config(['us.anthropic.claude-haiku-5-5'])))",
            AWS_PROFILE="",
        )
        self.assertNotIn("aws_profile_name", got)
        self.assertIn("aws_region_name: os.environ/AWS_REGION", got)

    def test_workbuddy_routes_use_the_bridge(self):
        got = probe(
            "import json; from harness2 import install; "
            "print(json.dumps(install.workbuddy_model_name('us.anthropic.claude-opus-4-6-v1')))"
        )
        self.assertEqual(got, "us.anthropic.claude-opus-4-6-v1")


FIRST_PATCH = """diff --git a/a.txt b/a.txt
--- a/a.txt
+++ b/a.txt
@@ -1,3 +1,3 @@
 one
-two
+TWO
 three
"""

SECOND_PATCH = """diff --git a/b.txt b/b.txt
--- a/b.txt
+++ b/b.txt
@@ -1,3 +1,3 @@
 one
-two
+TWO
 three
"""


class OpencodePatchTests(unittest.TestCase):
    """The Bedrock credential-chain change ships as its own patch, so checkouts upgrade."""

    def test_opencode_carries_the_credential_chain_patch(self):
        self.assertEqual(
            install.PATCHES["opencode"],
            ("opencode-compat.diff", "opencode-bedrock-credential-chain.diff"),
        )
        paths = install.patch_files("opencode")
        self.assertEqual([path.name for path in paths if not path.is_file()], [])
        diff = paths[-1].read_text(encoding="utf-8")
        self.assertIn("packages/opencode/src/provider/provider.ts", diff)
        self.assertIn("+      const credentialChainOptIn", diff)
        self.assertIn("OPENCODE_BEDROCK_CREDENTIAL_CHAIN", diff)
        # The upstream hints survive: the opt-in is one more way in, not a replacement.
        for hint in (
            "!profile",
            "!awsAccessKeyId",
            "!awsBearerToken",
            "!awsWebIdentityTokenFile",
            "!containerCreds",
        ):
            self.assertIn(f"+        {hint} &&", diff)

    def test_patches_apply_one_by_one_on_an_already_patched_checkout(self):
        from harness2 import config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patches = root / "third_party" / "patches"
            patches.mkdir(parents=True)
            (patches / "first.diff").write_text(FIRST_PATCH, encoding="utf-8")
            (patches / "second.diff").write_text(SECOND_PATCH, encoding="utf-8")
            repo = root / "dependencies" / "opencode"
            repo.mkdir(parents=True)
            for name in ("a.txt", "b.txt"):
                (repo / name).write_text("one\ntwo\nthree\n", encoding="utf-8")
            for command in (
                ["git", "init", "-q", "."],
                ["git", "add", "-A"],
                ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "pin"],
                ["git", "apply", str(patches / "first.diff")],
            ):
                subprocess.run(command, cwd=repo, check=True)

            with (
                patch.object(config, "THIRD_PARTY", root / "third_party"),
                patch.object(config, "DEPENDENCIES", root / "dependencies"),
                patch.dict(install.PATCHES, {"opencode": ("first.diff", "second.diff")}),
            ):
                install.patch("opencode")
                self.assertEqual((repo / "a.txt").read_text(), "one\nTWO\nthree\n")
                self.assertEqual((repo / "b.txt").read_text(), "one\nTWO\nthree\n")
                install.patch("opencode")  # applying twice is a no-op, not a failure
                self.assertEqual((repo / "b.txt").read_text(), "one\nTWO\nthree\n")


if __name__ == "__main__":
    unittest.main()
