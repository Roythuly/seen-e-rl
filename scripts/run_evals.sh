#!/usr/bin/env bash
set -euo pipefail

configs=("$@")
if [ ${#configs[@]} -eq 0 ]; then
  configs=(
    "configs/experiment/ppo_humanoid_v5.yaml"
    "configs/experiment/sac_humanoid_v5.yaml"
    "configs/experiment/td3_humanoid_v5.yaml"
  )
fi

runtime_report="$(python scripts/validate_runtime_env.py)"
printf '%s\n' "${runtime_report}"

validated_backend="$(printf '%s\n' "${runtime_report}" | awk -F= '/^mujoco_gl=/{print $2; exit}')"
if [ "${validated_backend:-}" = "auto" ]; then
  unset MUJOCO_GL
elif [ -n "${validated_backend:-}" ]; then
  export MUJOCO_GL="${validated_backend}"
fi

for config in "${configs[@]}"; do
  echo "=== train ${config} ==="
  python main.py train --config "${config}"
  echo "=== evaluate ${config} ==="
  python main.py evaluate --config "${config}" --selector latest
done
