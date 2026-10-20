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

"""Benchmark dependency and runtime readiness checks."""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request

from . import config
from .install import missing_runners, patch_files, workbuddy_model_name
from .substrate.opencode import OpencodeSubstrate


def check(adapter, *, judge=False) -> list[str]:
    missing = missing_runners()
    if missing:
        return [missing]
    errors = []

    def probe(label, cmd, *, cwd=None, remedy="run harness2-setup for this benchmark"):
        try:
            r = subprocess.run(
                [str(c) for c in cmd], cwd=cwd, capture_output=True, text=True, timeout=45
            )
            if r.returncode:
                errors.append(f"{label} failed; {remedy}")
            return r.returncode == 0
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{label}: {type(exc).__name__}; see README.md")
            return False

    for validate in (
        lambda: config.validate_models(judge=judge),
        adapter.tasks,
        OpencodeSubstrate().preflight,
        adapter.preflight,
    ):
        try:
            validate()
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(str(exc))

    bench = adapter.spec.name.split("-", 1)[0]
    name = {"jb": "job-bench-eval", "lab": "harvey-labs", "wb": "workbuddy-bench"}[bench]
    repo = config.DEPENDENCIES / name
    for dependency in (name, "opencode"):
        target = config.DEPENDENCIES / dependency
        if (target / ".git").exists():
            try:
                head = subprocess.check_output(
                    ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
                ).strip()
                if head != config.DEPENDENCY_PINS[dependency]:
                    errors.append(f"{dependency}: checkout differs from the supported pin")
            except subprocess.CalledProcessError:
                errors.append(f"{dependency}: cannot read checkout revision")
        for patch in patch_files(dependency):
            probe(
                f"{dependency} patch {patch.name}",
                ["git", "-C", target, "apply", "--reverse", "--check", patch],
            )
    if not (config.OPENCODE_DIR / "node_modules").is_dir():
        errors.append(f"OpenCode dependencies missing; run harness2-setup {bench}")

    if config.USES_BEDROCK:
        errors.extend(aws_credential_errors())
    if config.ON_AWS:
        solver = config.SOLVER_MODEL_OC.removeprefix(config.BEDROCK_PREFIX)
        jb_bridged = judge and config.JB_JUDGE_API_BASE == config.BRIDGE_URL
        served = {
            "jb": [config.JUDGE_MODEL] if jb_bridged else [],
            "wb": [solver, config.JUDGE_MODEL],
        }
        problem = _bridge_problem(served.get(bench, []))
        if problem:
            errors.append(problem)
        if judge and bench == "jb" and not jb_bridged and not config.JB_JUDGE_API_KEY:
            errors.append("HARNESS2_JUDGE_BASE_URL is not the bridge; set HARNESS2_JUDGE_API_KEY")
    if config.IMPROVER_MODEL.startswith("google-vertex") or (
        not config.ON_AWS
        and (
            config.SOLVER_MODEL_OC.startswith("google-vertex")
            or (judge and (bench == "wb" or config.JUDGE_MODEL.startswith(("claude", "gemini"))))
        )
    ):
        probe(
            "Google Application Default Credentials",
            ["gcloud", "auth", "application-default", "print-access-token"],
        )
    if bench == "lab":
        for binary in ("pandoc", "pdftoppm"):
            if not shutil.which(binary):
                errors.append(f"{binary} missing; install pandoc and poppler-utils (README.md)")
        probe(
            "LAB document and judge dependencies",
            [
                config.LAB_PYTHON,
                "-c",
                "import evaluation.scoring; import anthropic, "
                + ("boto3" if config.ON_AWS else "google.auth"),
            ],
            cwd=repo,
        )
    elif bench == "jb" and judge:
        probe(
            "JobBench document and judge dependencies",
            [
                config.JB_JUDGE_PYTHON,
                "-c",
                "import pandas, openpyxl, docx, mammoth, pdfplumber, pptx, openai",
            ],
        )
        if not config.ON_AWS and not config.JB_JUDGE_API_BASE:
            errors.append(
                "Set GOOGLE_CLOUD_PROJECT or HARNESS2_JUDGE_BASE_URL for JobBench judging"
            )
        if (
            not config.ON_AWS
            and config.JUDGE_MODEL
            and not config.JB_JUDGE_API_KEY
            and not config.JUDGE_MODEL.startswith("gemini")
        ):
            errors.append(
                "A non-Gemini JobBench judge requires HARNESS2_JUDGE_BASE_URL and HARNESS2_JUDGE_API_KEY"
            )
    elif bench == "wb":
        probe("Docker daemon", ["docker", "info", "--format", "{{.ServerVersion}}"])
        if adapter.spec.name == "wb-office":
            probe(
                "WorkBuddy document rendering",
                [config.WB_RENDER_PY, "-c", "import openpyxl, pptx"],
                remedy="install 'harness2[render]' in the main environment",
            )
        probe(
            "WorkBuddy runtime",
            [
                repo / ".venv/bin/python",
                "-c",
                "import harbor; from workbuddy_bench.runner.harness_adapters import HARNESS_ADAPTERS; "
                "assert all(k in HARNESS_ADAPTERS for k in ('opencode', 'oc-harness', 'codex-harness'))",
            ],
            cwd=repo,
        )
        for role, expected in (
            ("solver", config.SOLVER_MODEL_OC.split("/", 1)[-1]),
            ("judge", config.JUDGE_MODEL),
        ):
            path = repo / "configs/models/harness2" / f"{role}.yaml"
            try:
                if json.loads(path.read_text())["model"]["name"] != workbuddy_model_name(expected):
                    errors.append(f"WorkBuddy {role} configuration is stale; rerun setup")
            except (OSError, ValueError, KeyError, TypeError):
                errors.append(f"WorkBuddy {role} configuration missing; rerun setup")
    return errors


STS_TIMEOUT = 45
BEARER_TOKEN = "AWS_BEARER_TOKEN_BEDROCK"


def aws_credential_errors() -> list[str]:
    """Check for a usable AWS identity, and say plainly what that leaves unproved.

    Nothing here mirrors a credential chain: one `aws sts get-caller-identity` resolves
    the CLI's chain, so a named profile, the default profile, an SSO session, an EC2
    instance role, an ECS task role and an EKS web identity are all resolved by an SDK
    rather than guessed from the environment. The model clients (OpenCode's JavaScript
    provider, LiteLLM, the claude CLI) each resolve their own chain, which is not the same
    implementation, so this is evidence that the account and credential files are usable,
    not proof that those clients will resolve the same identity. It is not Bedrock access
    either: bedrock:InvokeModel on the configured model is still first exercised by the
    first model call.

    A Bedrock API key (AWS_BEARER_TOKEN_BEDROCK) signs no AWS request, has no STS identity
    and takes precedence over the chain inside OpenCode's Bedrock provider, so an STS call
    beside it would prove nothing about the key. That case warns and never blocks: only a
    billable model call separates a live key from a revoked one.
    """
    if config.env(BEARER_TOKEN):
        print(
            f"WARNING: {BEARER_TOKEN} is set and OpenCode's Bedrock provider prefers it over "
            "the AWS credential chain. Its validity is unverified here and no AWS identity "
            "is proved; benchmark steps that call AWS APIs other than Bedrock runtime still "
            "need standard IAM credentials.",
            flush=True,
        )
        return []
    if not shutil.which("aws"):
        return [
            "aws CLI missing; install it so preflight can resolve the AWS credential chain "
            "(a run itself resolves credentials through its own SDK)"
        ]
    command = [
        "aws",
        "sts",
        "get-caller-identity",
        "--region",
        config.AWS_REGION,
        "--no-cli-pager",
        "--output",
        "json",
    ]
    try:
        r = subprocess.run(command, capture_output=True, text=True, timeout=STS_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"AWS credential check did not finish ({type(exc).__name__}); see docs/SETUP.md"]
    if r.returncode == 0:
        return []
    detail = (r.stderr or r.stdout or "").strip().splitlines()
    return [
        "the AWS credential chain resolves no usable identity"
        + (f": {detail[-1][:200]}" if detail else "")
        + "; refresh the credentials the SDK would use (profile, SSO session, or role)"
    ]


def _bridge_problem(models: list[str]) -> str:
    """Why harness2-bridge cannot serve `models` from Bedrock, or "" when it can."""
    models = [m for m in models if m]
    if not models:
        return ""
    if not config.BRIDGE_URL or not config.BRIDGE_KEY:
        return "Start harness2-bridge in another session; Bedrock judging and WorkBuddy need it"
    request = urllib.request.Request(
        config.BRIDGE_URL.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {config.BRIDGE_KEY}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            served = {m.get("id") for m in json.load(response).get("data", [])}
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return f"harness2-bridge unreachable at {config.BRIDGE_URL} ({exc}); start it"
    missing = sorted(set(models) - served)
    if missing:
        return f"harness2-bridge does not serve {', '.join(missing)}; restart it"
    return ""


def require_ready(adapter, *, judge=False):
    """Raise before a caller spends anything when this benchmark is not ready to run.

    `judge=True` also demands what grading will need: `HARNESS2_JUDGE_MODEL` set, and,
    where a benchmark grades in its own environment, that environment installed and
    pointed at an endpoint. Those are offline presence and import checks — they catch an
    unset or unbuilt judge, not an account that cannot serve the judge model, which is
    still first learned when the judge actually calls it.

    The one exception is the AWS credential check, which does reach the network to resolve
    the credential chain; see `aws_credential_errors` for what it proves.
    """
    errors = check(adapter, judge=judge)
    if errors:
        raise config.ConfigError(
            "Setup incomplete; nothing has run yet:\n  - " + "\n  - ".join(errors)
        )
