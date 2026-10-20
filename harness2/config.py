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

"""Deployment settings and dependency versions. Shell values override .env."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent if (PKG.parent / "pyproject.toml").is_file() else Path.cwd()
THIRD_PARTY = ROOT / "third_party"
METHOD_VERSION = 2

DEPENDENCIES = ROOT / "dependencies"


class ConfigError(RuntimeError):
    """A required deployment value is missing; the message carries the remediation."""


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Load KEY=VALUE lines from `path` into os.environ with setdefault (the shell wins).

    Accepts blank lines, `#` comments, an optional `export ` prefix, and single- or
    double-quoted values. A missing or unreadable file is not an error.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or not all(c.isalnum() or c == "_" for c in key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        os.environ.setdefault(key, value)


load_dotenv()


def env(name: str, default=None):
    """Env override where a set-but-empty variable is the same as an unset one."""
    return os.environ.get(name) or default


def env_int(name: str, default: int) -> int:
    v = env(name)
    if v is None:
        return int(default)
    try:
        return int(v)
    except ValueError as e:
        raise ConfigError(f"{name} must be an integer, got {v!r}") from e


def require(name: str, *, needed_by: str) -> str:
    """The value of `name`, or ConfigError with the remediation line when it is unset/empty."""
    v = env(name)
    if not v:
        raise ConfigError(
            f"{name} is required for {needed_by}; set it in .env or the shell (see .env.example)"
        )
    return str(v)


RUNS = Path(env("HARNESS2_RUNS_DIR", ROOT / "runs")).expanduser().resolve()
VERTEX_PROJECT = env("GOOGLE_CLOUD_PROJECT", env("HARNESS2_VERTEX_PROJECT", ""))
VERTEX_REGION = env("GOOGLE_CLOUD_LOCATION", "global")
if VERTEX_PROJECT:
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", VERTEX_PROJECT)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", VERTEX_REGION)
if not os.environ.get("AWS_PROFILE", "x"):
    del os.environ["AWS_PROFILE"]  # an empty profile names no profile and breaks the AWS SDKs
AWS_PROFILE = env("AWS_PROFILE", "")
AWS_REGION = env("AWS_REGION", env("AWS_DEFAULT_REGION", "us-east-1"))


SOLVER_MODEL_OC = env("HARNESS2_SOLVER_MODEL", "")
SOLVER_MODEL_CX = env("HARNESS2_CODEX_MODEL", SOLVER_MODEL_OC.split("/", 1)[-1])
IMPROVER_MODEL = env("HARNESS2_IMPROVER_MODEL", SOLVER_MODEL_OC)
JUDGE_MODEL = env("HARNESS2_JUDGE_MODEL", "")

# Bare model IDs (judge, claude_cli improver, Codex) carry no provider, so the cloud
# that serves them is explicit, defaulting to the executor's OpenCode provider.
BEDROCK_PREFIX = "amazon-bedrock/"
CLOUD = env(
    "HARNESS2_CLOUD", "aws" if SOLVER_MODEL_OC.startswith(BEDROCK_PREFIX) else "gcp"
).lower()
ON_AWS = CLOUD == "aws"
# A GCP deployment may still run a Bedrock improver, so Bedrock settings follow the roles
# actually pointed at Bedrock rather than the cloud of the executor.
USES_BEDROCK = ON_AWS or IMPROVER_MODEL.startswith(BEDROCK_PREFIX)
if USES_BEDROCK:
    os.environ.setdefault("AWS_REGION", AWS_REGION)
    # Credentials are the AWS SDK's job, not this file's: every consumer here (OpenCode's
    # Bedrock provider, LiteLLM, the claude CLI under CLAUDE_CODE_USE_BEDROCK) resolves
    # them through its default credential chain, so no explicit AWS_PROFILE is required
    # for a default profile, an SSO session, an EC2 instance role, an ECS task role or an
    # EKS web identity -- each still needs its own setup, just not one named here. The
    # pinned OpenCode autoloads amazon-bedrock only when an environment variable hints at
    # a credential, which misses the sources that leave no trace; this opt-in, added by
    # third_party/patches/opencode-bedrock-credential-chain.diff, is set only where a role
    # is on Bedrock and hands resolution to the chain. Whether a credential works is
    # preflight's question.
    os.environ.setdefault("OPENCODE_BEDROCK_CREDENTIAL_CHAIN", "1")


# On AWS, OpenAI-protocol consumers (JobBench judge, WorkBuddy proxy, Codex) reach
# Bedrock through harness2-bridge, which saves its port and key here.
BRIDGE_FILE = DEPENDENCIES / "codex" / "bridge.json"


def _saved_bridge() -> dict:
    try:
        saved = json.loads(BRIDGE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return saved if isinstance(saved, dict) else {}


_bridge = _saved_bridge()
BRIDGE_URL = env(
    "HARNESS2_BRIDGE_URL", f"http://127.0.0.1:{_bridge['port']}/v1" if _bridge.get("port") else ""
)
BRIDGE_KEY = env("HARNESS2_BRIDGE_KEY", _bridge.get("key", ""))

DEPENDENCY_PINS = {
    "job-bench-eval": "fa03a17f9ff662f136798a274e10011c3aa00fd0",
    "workbuddy-bench": "b516950be5b56eb3be406c2f76ee1c5111dcb57f",
    "harvey-labs": "b58a28ae9fde5ee4b810e53b4fd0a1915325a35d",
    "opencode": "23fb5e0516c99ac04a1aa46c193efda2e1b9bb24",
}
DEPENDENCY_ORIGINS = {
    "job-bench-eval": "https://github.com/Job-Bench/job-bench-eval.git",
    "workbuddy-bench": "https://github.com/Tencent/workbuddy-bench.git",
    "harvey-labs": "https://github.com/harveyai/harvey-labs.git",
    "opencode": "https://github.com/anomalyco/opencode.git",
}


def jb_repo() -> Path:
    return DEPENDENCIES / "job-bench-eval"


def wb_repo() -> Path:
    return DEPENDENCIES / "workbuddy-bench"


def lab_repo() -> Path:
    return DEPENDENCIES / "harvey-labs"


OPENCODE_DIR = DEPENDENCIES / "opencode"
BUN = DEPENDENCIES / "tools" / "node_modules" / ".bin" / "bun"
LAB_PYTHON = lab_repo() / ".venv" / "bin" / "python"
if ON_AWS:
    LAB_JUDGE_MODEL = (
        "bedrock/" + JUDGE_MODEL if JUDGE_MODEL and "/" not in JUDGE_MODEL else JUDGE_MODEL
    )
else:
    LAB_JUDGE_MODEL = (
        "vertex/" + JUDGE_MODEL if JUDGE_MODEL.startswith(("claude", "gemini")) else JUDGE_MODEL
    )
LAB_DELIV_CAP = 60000
JB_JUDGE_PYTHON = jb_repo() / ".venv" / "bin" / "python"
JB_JUDGE_MODEL = (
    "google/" + JUDGE_MODEL
    if not ON_AWS and JUDGE_MODEL.startswith("gemini") and not env("HARNESS2_JUDGE_BASE_URL")
    else JUDGE_MODEL
)
_vertex_host = (
    "aiplatform.googleapis.com"
    if VERTEX_REGION == "global"
    else f"{VERTEX_REGION}-aiplatform.googleapis.com"
)
VERTEX_OPENAI_URL = (
    f"https://{_vertex_host}/v1/projects/{VERTEX_PROJECT}"
    f"/locations/{VERTEX_REGION}/endpoints/openapi"
)
if ON_AWS:
    JB_JUDGE_API_BASE = env("HARNESS2_JUDGE_BASE_URL", BRIDGE_URL)
    JB_JUDGE_API_KEY = env("HARNESS2_JUDGE_API_KEY", BRIDGE_KEY)
else:
    JB_JUDGE_API_BASE = env("HARNESS2_JUDGE_BASE_URL", VERTEX_OPENAI_URL if VERTEX_PROJECT else "")
    JB_JUDGE_API_KEY = env("HARNESS2_JUDGE_API_KEY", "")
# Claude on Bedrock rejects temperature 0 alongside extended thinking, and caps output
# below the upstream 64k default, so the AWS judge runs without a reasoning effort.
JB_JUDGE_REASONING_EFFORT = env("HARNESS2_JUDGE_REASONING_EFFORT", "default" if ON_AWS else "low")
JB_JUDGE_MAX_TOKENS = env("HARNESS2_JUDGE_MAX_TOKENS", "32000" if ON_AWS else "")
JB_REPORTING_JUDGE_DETAILS = ""
JB_OBS_TRUNC = 50000

CODEX_VERSION = "0.144.6"
CODEX_DIST = str(DEPENDENCIES / "codex" / "dist")
CODEX_BASE_URL = env("HARNESS2_CODEX_BASE_URL", "")
CODEX_CONTAINER_URL = env("HARNESS2_CODEX_CONTAINER_URL", CODEX_BASE_URL)
VERTEX_PROXY_KEY = env("HARNESS2_CODEX_API_KEY", "")
CODEX_HOME_DIR = DEPENDENCIES / "codex" / "home"

WB_SOLVER_MODEL = "harness2/solver"
WB_WEB_JUDGE = "harness2/judge"
WB_RENDER_PY = sys.executable
WB_OC_HARNESS_VER = "1.14.18"
WB_CX_HARNESS_VER = CODEX_VERSION
WB_MAX_JOBS = {"oc": 1, "cx": 1}
WB_MAX_TRIALS = {"oc": 2, "cx": 2}
WB_MODEL_CONNECTION = "local_proxy"

DEFAULT_PROPOSER_CHANNEL = "opencode"
PROPOSER_MODEL_BY_CHANNEL = {"opencode": IMPROVER_MODEL, "claude_cli": IMPROVER_MODEL}
PHASE_WIDTH = 2
TRANSPORT_ATTEMPTS = 2
PROPOSER_CONCURRENCY = 2


def validate_models(*, judge: bool = False) -> None:
    require("HARNESS2_SOLVER_MODEL", needed_by="executor rollouts")
    if "/" not in SOLVER_MODEL_OC or "/" not in IMPROVER_MODEL:
        raise ConfigError("Executor and improver models must use OpenCode's provider/model syntax")
    if CLOUD not in ("aws", "gcp"):
        raise ConfigError(f"HARNESS2_CLOUD must be 'aws' or 'gcp', got {CLOUD!r}")
    if ON_AWS and not SOLVER_MODEL_OC.startswith(BEDROCK_PREFIX):
        raise ConfigError(f"HARNESS2_CLOUD=aws needs a {BEDROCK_PREFIX}... HARNESS2_SOLVER_MODEL")
    if not ON_AWS and SOLVER_MODEL_OC.startswith(BEDROCK_PREFIX):
        raise ConfigError("A Bedrock executor needs HARNESS2_CLOUD=aws (or unset)")
    if any(m.startswith("google-vertex") for m in (SOLVER_MODEL_OC, IMPROVER_MODEL)):
        if not VERTEX_PROJECT:
            raise ConfigError("Set GOOGLE_CLOUD_PROJECT for Vertex AI")
    if judge:
        require("HARNESS2_JUDGE_MODEL", needed_by="benchmark judging")
