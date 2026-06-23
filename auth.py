from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import settings


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class AuthStore:
    """
    Хранилище авторизованных пользователей.

    Админы берутся из ADMIN_USER_IDS.
    Обычные пользователи хранятся в AUTH_USERS_FILE.

    На Amvera:
    AUTH_USERS_FILE=/data/auth_users.json
    """

    def __init__(self) -> None:
        self.path = Path(settings.auth_users_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _empty_data(self) -> dict[str, Any]:
        return {
            "users": {},
            "pending": {},
        }

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty_data()

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return self._empty_data()

        if "users" not in data:
            data["users"] = {}

        if "pending" not in data:
            data["pending"] = {}

        return data

    def _save(self) -> None:
        temp_path = self.path.with_suffix(".tmp")

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

        temp_path.replace(self.path)

    def reload(self) -> None:
        self.data = self._load()

    def is_admin(self, user_id: int | None) -> bool:
        if user_id is None:
            return False

        return user_id in settings.admin_user_ids

    def is_static_allowed(self, user_id: int | None) -> bool:
        if user_id is None:
            return False

        return user_id in settings.allowed_user_ids

    def is_allowed(self, user_id: int | None) -> bool:
        if user_id is None:
            return False

        if self.is_admin(user_id):
            return True

        if self.is_static_allowed(user_id):
            return True

        return str(user_id) in self.data["users"]

    def add_pending(
        self,
        user_id: int,
        username: str | None,
        full_name: str | None,
    ) -> bool:
        """
        Добавляет заявку.

        Возвращает True, если заявка новая.
        Возвращает False, если заявка уже была.
        """
        if self.is_allowed(user_id):
            return False

        key = str(user_id)
        is_new = key not in self.data["pending"]

        old_requested_at = self.data["pending"].get(key, {}).get("requested_at")

        self.data["pending"][key] = {
            "user_id": user_id,
            "username": username,
            "full_name": full_name,
            "requested_at": old_requested_at or now_str(),
            "updated_at": now_str(),
        }

        self._save()

        return is_new

    def get_pending(self, user_id: int) -> dict[str, Any] | None:
        return self.data["pending"].get(str(user_id))

    def list_pending(self) -> list[dict[str, Any]]:
        items = list(self.data["pending"].values())
        return sorted(items, key=lambda x: x.get("requested_at", ""))

    def approve_user(self, user_id: int, admin_id: int) -> dict[str, Any]:
        key = str(user_id)

        record = self.data["pending"].pop(
            key,
            {
                "user_id": user_id,
                "username": None,
                "full_name": None,
                "requested_at": None,
            },
        )

        record["approved_by"] = admin_id
        record["approved_at"] = now_str()
        record["status"] = "approved"

        self.data["users"][key] = record
        self._save()

        return record

    def deny_user(self, user_id: int, admin_id: int) -> dict[str, Any] | None:
        key = str(user_id)

        record = self.data["pending"].pop(key, None)

        if record:
            record["denied_by"] = admin_id
            record["denied_at"] = now_str()

        self._save()

        return record

    def allow_user(
        self,
        user_id: int,
        admin_id: int,
        username: str | None = None,
        full_name: str | None = None,
    ) -> dict[str, Any]:
        key = str(user_id)

        record = {
            "user_id": user_id,
            "username": username,
            "full_name": full_name,
            "approved_by": admin_id,
            "approved_at": now_str(),
            "status": "approved",
            "source": "manual_admin_command",
        }

        self.data["pending"].pop(key, None)
        self.data["users"][key] = record
        self._save()

        return record

    def revoke_user(self, user_id: int) -> tuple[bool, str]:
        """
        Удаляет пользователя из динамического списка.

        Нельзя удалить:
        - админа из ADMIN_USER_IDS
        - пользователя из ALLOWED_USER_IDS

        Их нужно удалять из переменных Amvera.
        """
        if self.is_admin(user_id):
            return (
                False,
                "Этот пользователь является администратором. "
                "Удалите его из ADMIN_USER_IDS в Amvera и перезапустите контейнер.",
            )

        if self.is_static_allowed(user_id):
            return (
                False,
                "Этот пользователь задан в ALLOWED_USER_IDS. "
                "Удалите его из переменных Amvera и перезапустите контейнер.",
            )

        key = str(user_id)

        removed_user = self.data["users"].pop(key, None)
        removed_pending = self.data["pending"].pop(key, None)

        if removed_user is None and removed_pending is None:
            return False, "Пользователь не найден в списке доступа или заявок."

        self._save()

        return True, "Пользователь удалён."

    def list_users(self) -> list[dict[str, Any]]:
        items = list(self.data["users"].values())
        return sorted(items, key=lambda x: x.get("approved_at", ""))