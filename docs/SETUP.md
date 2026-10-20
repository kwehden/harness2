# Setup guide

Start with the [quick start](../README.md#install). This guide covers benchmark
dependencies, model credentials, and the optional Codex executor. For individual
benchmarks, see the [benchmark guides](BENCHMARKS.md).

## Requirements

Use Linux and Python 3.12 for benchmark runs; the Python package also supports
3.13. On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y git curl unzip ca-certificates pandoc poppler-utils
```

Pandoc and Poppler are needed by LAB. Also install:

- [Node.js](https://nodejs.org/en/download) 20 or newer.
- [uv](https://docs.astral.sh/uv/getting-started/installation/).
- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
  for Bedrock, or [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
  for Vertex AI. You do not need both for a single-cloud configuration.
- [Docker Engine](https://docs.docker.com/engine/install/) 28 or newer for
  WorkBuddy-Bench, with permission to run `docker info`, plus the Compose v2+
  and buildx 0.17+ plugins. WorkBuddy attaches the agent CLI to task containers
  as an image mount, which needs Docker Engine API 1.48. On distributions that
  package an older Engine (Amazon Linux 2023 ships 25.x), use Docker's
  [static binaries](https://docs.docker.com/engine/install/binaries/).

Benchmark setup installs its own pinned Bun runtime under `dependencies/`.
Keep the main Harness² environment active when running the commands below.

## Models and credentials

Copy `.env.example` to `.env` if needed. This fork's example selects Amazon
Bedrock. Shell environment variables take precedence over `.env`; keep secrets
out of tracked files. Your account must have access to every configured model.

| Setting | Meaning |
|---|---|
| `HARNESS2_CLOUD` | `aws` or `gcp`; inferred from the executor provider when omitted |
| `HARNESS2_SOLVER_MODEL` | Executor model in OpenCode's `provider/model` format |
| `HARNESS2_IMPROVER_MODEL` | Improver model in the same format; defaults to the executor |
| `HARNESS2_JUDGE_MODEL` | Bare model ID for the chosen cloud; adapters add their routing prefix |
| `HARNESS2_RUNS_DIR` | Output root; defaults to `runs/`; change it when changing models or tasks |

### Amazon Bedrock

Harness² can run every model role on Amazon Bedrock instead of Vertex AI. It
selects AWS when the executor uses OpenCode's `amazon-bedrock/` provider, or
when `HARNESS2_CLOUD=aws`. Bare model IDs (the judge, a `claude_cli` improver,
and Codex) then also resolve on Bedrock. Install the
[AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
so that `harness2-check` can verify credentials.

```bash
# Optional: set only when using a named profile.
# AWS_PROFILE=bedrock
HARNESS2_CLOUD=aws
AWS_REGION=us-west-2
HARNESS2_SOLVER_MODEL=amazon-bedrock/us.anthropic.claude-haiku-5-5
HARNESS2_IMPROVER_MODEL=amazon-bedrock/us.anthropic.claude-haiku-5-5
HARNESS2_JUDGE_MODEL=us.anthropic.claude-haiku-5-5
```

Use inference-profile IDs (`us.`, `eu.`, `global.`, ...) that your account can
invoke in `AWS_REGION`; `aws bedrock list-inference-profiles` lists available
profiles. The example IDs are not a guarantee of model access. Use
least-privilege permissions for Bedrock invocation, including streaming and the
resources required by your inference profile, rather than an administrator role.

`AWS_PROFILE` is optional. Leave it unset to use the default shared profile or
the SDK credential chain, including EC2 instance roles, ECS task roles, and EKS
web identity. Use your normal AWS authentication process to refresh credentials.
The readiness check uses `aws sts get-caller-identity`; configuration loading
does not contact AWS. STS success verifies AWS identity, not model access.
Use standard AWS credentials for the full benchmark workflow. A Bedrock bearer
token has no STS identity, so the no-model-call check cannot verify it. OpenCode
prefers that token when present, even alongside a profile; unset it when you
intend to use the profile or role.

| Role | How it reaches Bedrock |
|---|---|
| OpenCode executor and improver | OpenCode's `amazon-bedrock` provider |
| `claude_cli` improver | Claude Code with `CLAUDE_CODE_USE_BEDROCK=1` |
| LAB judge | `anthropic.AnthropicBedrock`, as `bedrock/<judge model>` |
| JobBench judge, WorkBuddy solver and judge, Codex | `harness2-bridge` |

JobBench judging, WorkBuddy, and Codex speak the OpenAI protocol, so on AWS
they need `harness2-bridge` running; see [Codex executor](#codex-executor). On
AWS it serves the Codex, judge, and WorkBuddy solver models from one endpoint,
and the judge and WorkBuddy routes find it automatically. Start the bridge
before `harness2-setup wb` and `harness2-check`. To use another
OpenAI-compatible endpoint, set `HARNESS2_BRIDGE_URL` and `HARNESS2_BRIDGE_KEY`.

On AWS, setup turns off Codex's built-in web search. Through the bridge it would
become Anthropic's server-side web search tool, which Bedrock does not host, and
every request would fail. Codex can still reach the web through its shell.

The Haiku 5.5 OpenCode improver uses adaptive thinking with the configured
effort instead of OpenCode's legacy fixed-budget variant. Older Bedrock Claude
models retain their existing variant behavior. The LAB judge also uses adaptive
thinking, so choose a compatible Claude judge, such as the Haiku 5.5 example.
The JobBench judge runs on AWS without a reasoning effort and with a
32,000-token completion cap. Override these with
`HARNESS2_JUDGE_REASONING_EFFORT` and `HARNESS2_JUDGE_MAX_TOKENS`.

### Google Vertex AI

Comment out the Bedrock model settings in `.env` and enable its Vertex example.
Set `GOOGLE_CLOUD_PROJECT`, the executor and improver in OpenCode's
`google-vertex/model` format, and a bare judge ID. `GOOGLE_CLOUD_LOCATION`
defaults to `global`. Authenticate with:

```bash
gcloud auth application-default login
```

The Vertex configuration supports Gemini executors and judges, with either
Vertex Claude or Gemini as the improver. LAB also supports the upstream judge's
direct providers when their credentials are configured. WorkBuddy's Vertex
routes require Gemini for both solver and judge; rerun setup after changing
these models.

On a machine with an attached service account, Application Default Credentials
can use that account. Other options include:

- `gcloud auth application-default login --no-browser` for a remote machine.
- `GOOGLE_APPLICATION_CREDENTIALS` pointing at a service account key file.
- `gcloud auth application-default login --impersonate-service-account=<account>`.

Keep `gcloud` on `PATH`, including for the key-file route: the installation check
runs `gcloud auth application-default print-access-token`.

## Benchmark installation

Install only the benchmark you intend to run:

```bash
harness2-setup lab
harness2-check --bench lab
```

Use `jb` for JobBench or `wb` for WorkBuddy-Bench. See the
[benchmark guides](BENCHMARKS.md) for subset selection and licensing.

The runner overlays and compatibility patches are included in
[`third_party/`](../third_party/README.md). Setup clones supported upstream
revisions under `dependencies/`, applies the patches, creates the benchmark
environments, and downloads task data.

Each benchmark environment lives inside its upstream checkout:

- `dependencies/harvey-labs/.venv`
- `dependencies/job-bench-eval/.venv`
- `dependencies/workbuddy-bench/.venv`

Harness² selects these interpreters automatically; do not activate them for the
main commands.

### Installation checks

`harness2-setup <bench> --dry-run` previews setup without making changes.
`harness2-check` checks installed dependencies, credentials, and configuration
without calling a model. Refinement runs perform the same checks, including
judge configuration, before starting.

These checks do not test model availability. Access to an executor, improver, or
judge model is tested when that model is first called. Refinement and judging
return a nonzero exit status when incomplete.

## Developer checks

From a source checkout with the main virtual environment active:

```bash
python -m unittest discover -s tests -v
python scripts/check_provenance.py
python -m compileall -q harness2 tests scripts
harness2-demo --output runs/dev-check --k 2
```

These checks need no model credentials or downloaded OpenCode checkout. The
GitHub workflow runs them on Python 3.12 and 3.13, then builds the source
archive and wheel and verifies the archive contains the benchmark assets.
They do not replace a live smoke run for model access or provider compatibility.

## Codex executor

The Codex executor connects to a Responses API endpoint. For Gemini on Vertex
AI, the included `harness2-bridge` provides this interface through a local
LiteLLM server, using the same Application Default Credentials and
`GOOGLE_CLOUD_PROJECT`. On [Amazon Bedrock](#amazon-bedrock) it uses
the AWS credential chain and `AWS_REGION`.

Install the optional dependencies and start the bridge in a separate session:

```bash
uv pip install -e '.[render,bridge]'
harness2-bridge                 # for WorkBuddy: harness2-bridge --host 0.0.0.0
```

The bridge prints the `HARNESS2_CODEX_*` settings to add to `.env`. Its port and
generated access key are stored under `dependencies/codex/` and reused on
restart. Pass `--substrate cx` to setup, checks, refinement, judging, and
reporting:

```bash
harness2-setup lab --substrate cx
harness2-check --bench lab --substrate cx
harness2 --bench lab --substrate cx --k 1 --mode parallel \
  --only-domains antitrust-competition --max-tasks 1
```

WorkBuddy executes Codex inside Docker containers, so the bridge must listen on
`--host 0.0.0.0`; it also prints `HARNESS2_CODEX_CONTAINER_URL`. This binds the
bridge to all network interfaces; restrict access to the intended clients.

To use another Responses API endpoint, set `HARNESS2_CODEX_BASE_URL`,
`HARNESS2_CODEX_API_KEY`, and `HARNESS2_CODEX_MODEL` directly. Setup records the
endpoint and model in `dependencies/codex/home/config.toml`; rerun setup after
changing either. `harness2-check` reports stale configuration until then.

Codex runs with its sandbox disabled. Review the
[executor network restrictions](EXPERIMENTS.md#web-access) before evaluation.

## Network access during setup

Setup downloads dependencies from the following hosts:

| Host | Used for |
|---|---|
| `github.com` | Pinned upstream checkouts |
| `registry.npmjs.org` | Pinned Bun runtime and Codex CLI, plus dependencies resolved by `bun install` |
| `huggingface.co` and its CDN | JobBench data and WorkBuddy subset archives |
| A Docker registry | WorkBuddy container image |
| `astral.sh` | CPython 3.12 when the host has no system Python 3.12 |

Use `harness2-setup <bench> --dry-run` to review setup commands when configuring
network access. Experiments also access the model endpoints configured in
`.env`. Setup downloads and executor web access have different requirements;
see [web access during evaluation](EXPERIMENTS.md#web-access).

## Wheel installations

Use a source checkout or the source distribution for benchmark setup. They
include the required `third_party/` runners, patches, licenses, and asset
checksums. Run `python scripts/check_provenance.py` to verify those assets.

A wheel ships the Python package only, without benchmark runners and patches.
It can run the deterministic demo, but installing the wheel alone is not enough
to set up a benchmark. The task partitioner, `python scripts/split_tasks.py`,
also requires the source tree. Outside a source checkout, `.env`,
`dependencies/`, and `runs/` are resolved from the working directory.
