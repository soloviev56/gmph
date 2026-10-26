#!/usr/bin/env bash
# Установить таймер systemd пользователя для ежедневного запуска.
#   scripts/daily/install.sh            — установить и включить
#   scripts/daily/install.sh --remove   — выключить и удалить
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
UNITS="$HOME/.config/systemd/user"

if [[ "${1:-}" == "--remove" ]]; then
    systemctl --user disable --now gmph-daily.timer || true
    rm -f "$UNITS/gmph-daily.service" "$UNITS/gmph-daily.timer"
    systemctl --user daemon-reload
    echo "таймер удалён"
    exit 0
fi

mkdir -p "$UNITS"
cp "$HERE/gmph-daily.service" "$HERE/gmph-daily.timer" "$UNITS/"
chmod +x "$HERE/run.sh"
systemctl --user daemon-reload
systemctl --user enable --now gmph-daily.timer
systemctl --user list-timers gmph-daily.timer --no-pager
