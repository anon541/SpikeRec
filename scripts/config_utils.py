"""Shared helpers for loading YAML/JSON configs and standard log paths."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Set

DEFAULT_LOG_TXT = "docs/analysis/logs.txt"
DEFAULT_LOG_JSONL = "docs/analysis/logs.jsonl"


def load_config_dict(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pyyaml is required to load YAML configs") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping of arg names to values")
    return data


def _get_explicitly_provided_args() -> Set[str]:
    """
    Extract argument names that were explicitly provided via CLI.
    Returns set of argument names (dest names, not CLI flags).
    
    This function checks sys.argv directly to determine which arguments
    were explicitly provided, ensuring CLI arguments always take precedence
    over config file values.
    """
    from scripts import arguments as arguments_cli
    
    parser = arguments_cli.build_parser()
    explicitly_provided = set()
    
    # Map CLI flags to their dest names
    cli_flag_to_dest = {}
    for action in parser._actions:
        if action.dest and action.option_strings:
            for opt in action.option_strings:
                cli_flag_to_dest[opt] = action.dest
    
    # Check sys.argv for explicitly provided flags
    argv_list = sys.argv[1:]
    i = 0
    while i < len(argv_list):
        arg = argv_list[i]
        if arg in cli_flag_to_dest:
            dest_name = cli_flag_to_dest[arg]
            explicitly_provided.add(dest_name)
            # Skip the value if it's not a flag
            if i + 1 < len(argv_list) and not argv_list[i + 1].startswith('-'):
                i += 1
        i += 1
    
    return explicitly_provided


def apply_config_overrides(args, overrides: Dict[str, Any]) -> None:
    """
    Apply config defaults but preserve explicit CLI overrides.
    
    Rule: 
    - CLI에서 명시적으로 전달된 인자는 항상 우선 (config 무시)
    - CLI에서 전달되지 않은 인자만 config 파일 값 적용
    """
    from scripts import arguments as arguments_cli  # local import to avoid cycle

    parser = arguments_cli.build_parser()
    defaults = parser.parse_args([])  # empty CLI -> pure defaults
    default_dict = vars(defaults)
    
    # CLI에서 명시적으로 전달된 인자 추적
    explicitly_provided = _get_explicitly_provided_args()

    for key, value in overrides.items():
        current = getattr(args, key, None)
        default_val = default_dict.get(key, None)
        
        # CLI에서 명시적으로 전달된 인자는 config 무시
        if key in explicitly_provided:
            continue
        
        # Only override if current still at default or None (i.e., CLI didn't change it)
        if current == default_val or current is None:
            setattr(args, key, value)


def ensure_log_paths(args) -> None:
    if not getattr(args, "log_txt", ""):
        args.log_txt = DEFAULT_LOG_TXT
    if not getattr(args, "log_jsonl", ""):
        args.log_jsonl = DEFAULT_LOG_JSONL
    for attr in ("log_txt", "log_jsonl"):
        path = Path(getattr(args, attr))
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
