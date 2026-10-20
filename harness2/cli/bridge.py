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

"""Serve an OpenAI-compatible endpoint on Vertex AI or Amazon Bedrock.

On Vertex it backs the Codex executor. On AWS it also serves the JobBench judge and
WorkBuddy's model routes, which speak the OpenAI protocol that Bedrock does not.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import sys
from pathlib import Path

from harness2 import config

LITELLM_CONFIG = """model_list:
  - model_name: {model}
    litellm_params:
      model: vertex_ai/{model}
      vertex_project: os.environ/GOOGLE_CLOUD_PROJECT
      vertex_location: os.environ/GOOGLE_CLOUD_LOCATION
litellm_settings:
  drop_params: true
general_settings:
  master_key: os.environ/HARNESS2_BRIDGE_KEY
"""

BEDROCK_MODEL = """  - model_name: {model}
    litellm_params:
      model: bedrock/{route}{model}
      aws_region_name: os.environ/AWS_REGION
{profile}"""


def bedrock_models() -> list[str]:
    """The bare Bedrock IDs a run asks the bridge for: Codex, judge, and WorkBuddy solver."""
    wanted = [config.SOLVER_MODEL_CX, config.JUDGE_MODEL]
    if config.SOLVER_MODEL_OC.startswith(config.BEDROCK_PREFIX):
        wanted.append(config.SOLVER_MODEL_OC.removeprefix(config.BEDROCK_PREFIX))
    return list(dict.fromkeys(m for m in wanted if m))


def litellm_config(models: list[str]) -> str:
    if not config.ON_AWS:
        return LITELLM_CONFIG.format(model=models[0])
    # Name the profile only when the deployment named one; with no AWS_PROFILE, LiteLLM
    # falls back to boto3's default chain, which is what a role or default profile needs.
    profile = "      aws_profile_name: os.environ/AWS_PROFILE\n" if config.AWS_PROFILE else ""
    # Claude goes through Bedrock's native Anthropic invoke API: LiteLLM's Converse
    # translation of Codex's Responses requests sends tool_choice twice, which Bedrock
    # rejects. Other models keep the default Converse route.
    entries = "".join(
        BEDROCK_MODEL.format(model=m, route="invoke/" if "anthropic." in m else "", profile=profile)
        for m in models
    )
    tail = LITELLM_CONFIG.partition("litellm_settings:")[2]
    return "model_list:\n" + entries + "litellm_settings:" + tail


def _free_port(host: str) -> int:
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _endpoint(path: Path, host: str, port: int) -> dict:
    """Reuse the saved port and key, so the Codex config written by setup stays valid."""
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    endpoint = {
        "port": port or saved.get("port") or _free_port(host),
        "key": saved.get("key") or secrets.token_urlsafe(24),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(endpoint) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return endpoint


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        action="append",
        help="model ID to serve; repeatable. Default: the Codex model, plus on AWS the "
        "judge and WorkBuddy solver",
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to bind; use 0.0.0.0 so WorkBuddy containers can reach it",
    )
    p.add_argument("--port", type=int, default=0, help="default: the saved port, else a free one")
    args = p.parse_args(argv)

    litellm = Path(sys.executable).with_name("litellm")
    if not litellm.is_file():
        p.exit(1, "LiteLLM is not installed; run: uv pip install -e '.[render,bridge]'\n")
    if config.ON_AWS:
        # No credential gate: LiteLLM resolves Bedrock credentials through boto3's chain,
        # which also covers the default profile, SSO, and instance, task and web identity
        # roles. An unusable credential surfaces when the proxy serves a request, and
        # `harness2-check` resolves a chain before a run.
        models = args.model or bedrock_models()
    else:
        models = args.model or [config.SOLVER_MODEL_CX]
        if len(models) > 1:
            p.exit(1, "The Vertex bridge serves one model; pass --model once\n")
        if not config.VERTEX_PROJECT:
            p.exit(1, "Set GOOGLE_CLOUD_PROJECT in .env\n")
    if not any(models):
        p.exit(1, "No model to serve; set HARNESS2_SOLVER_MODEL or pass --model\n")

    root = config.BRIDGE_FILE.parent
    endpoint = _endpoint(config.BRIDGE_FILE, args.host, args.port)
    port = endpoint["port"]
    bridge_yaml = root / "bridge.yaml"
    bridge_yaml.write_text(litellm_config(models), encoding="utf-8")

    settings = {
        "HARNESS2_CODEX_BASE_URL": f"http://127.0.0.1:{port}/v1",
        "HARNESS2_CODEX_API_KEY": endpoint["key"],
        "HARNESS2_CODEX_MODEL": models[0],
    }
    if args.host != "127.0.0.1":
        settings["HARNESS2_CODEX_CONTAINER_URL"] = f"http://host.docker.internal:{port}/v1"
    if config.ON_AWS:
        print(f"Serving Bedrock models: {', '.join(models)}")
        print("The JobBench judge and WorkBuddy routes find this bridge automatically.")
    print("For Codex, add these to .env, then run harness2-setup <bench> --substrate cx:")
    print("\n".join(f"{name}={value}" for name, value in settings.items()), flush=True)

    env = {
        **os.environ,
        "HARNESS2_BRIDGE_KEY": endpoint["key"],
        **(
            {"AWS_REGION": config.AWS_REGION}
            if config.ON_AWS
            else {
                "GOOGLE_CLOUD_PROJECT": config.VERTEX_PROJECT,
                "GOOGLE_CLOUD_LOCATION": config.VERTEX_REGION,
            }
        ),
    }
    command = [str(litellm), "--config", str(bridge_yaml), "--host", args.host]
    os.execve(str(litellm), [*command, "--port", str(port)], env)


if __name__ == "__main__":
    raise SystemExit(main())
