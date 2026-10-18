#!/usr/bin/env bash
# Load a small, allow-listed set of simple KEY=value settings from a project .env file.
# Values are treated literally (no shell evaluation or variable expansion).
load_project_env() {
  local env_file="${1:-.env}"
  local line key value
  [ -f "$env_file" ] || return 0

  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    [[ "$line" =~ ^[[:space:]]*$ || "$line" =~ ^[[:space:]]*# ]] && continue
    if [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      key="${BASH_REMATCH[2]}"
      value="${BASH_REMATCH[3]}"
      case "$key" in
        AI_STUDIO_HOST|AI_STUDIO_PORT|OPENAI_BASE_URL|SERVED_MODEL_NAME|VLLM_API_KEY|QWEN_MODEL|VLLM_PORT|MAX_MODEL_LEN|MAX_NUM_SEQS|GPU_MEMORY_UTILIZATION|WAN22_REPO|WAN22_CKPT) ;;
        *) continue ;;
      esac
      if [[ ${#value} -ge 2 && "$value" == \"*\" ]]; then
        value="${value:1:${#value}-2}"
      elif [[ ${#value} -ge 2 && "$value" == \'*\' ]]; then
        value="${value:1:${#value}-2}"
      fi
      # An explicitly supplied shell variable takes precedence over .env.
      if [[ ! -v $key ]]; then
        printf -v "$key" '%s' "$value"
        export "$key"
      fi
    fi
  done < "$env_file"
}
