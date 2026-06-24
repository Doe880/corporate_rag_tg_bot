from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from config import settings
from db import get_connection, init_db


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def row_to_dict(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None

    return dict(row)


class AuthStore:
    """
    Авторизация через SQLite.

    Админы берутся из ADMIN_USER_IDS.
    Обычные пользователи и заявки хранятся в SQLite.

    На Amvera:
    DB_PATH=/data/bot.db
    """

    def __init__(self) -> None:
        init_db()

    def reload(self) -> None:
        """
        Оставлено для совместимости.
        Для SQLite отдельная перезагрузка не нужна.
        """
        init_db()

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

        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT user_id
                FROM users
                WHERE user_id = ?
                  AND status = 'active'
                """,
                (user_id,),
            ).fetchone()

        return row is not None

    def add_pending(
        self,
        user_id: int,
        username: str | None,
        full_name: str | None,
    ) -> bool:
        """
        Добавляет заявку на доступ.

        Возвращает True, если заявка новая.
        Возвращает False, если pending-заявка уже была.
        """
        if self.is_allowed(user_id):
            return False

        now = now_str()

        with get_connection() as conn:
            existing = conn.execute(
                """
                SELECT id
                FROM access_requests
                WHERE user_id = ?
                  AND status = 'pending'
                ORDER BY id DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()

            if existing:
                conn.execute(
                    """
                    UPDATE access_requests
                    SET username = ?,
                        full_name = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (username, full_name, now, existing["id"]),
                )
                return False

            conn.execute(
                """
                INSERT INTO access_requests (
                    user_id,
                    username,
                    full_name,
                    status,
                    requested_at,
                    updated_at
                )
                VALUES (?, ?, ?, 'pending', ?, ?)
                """,
                (user_id, username, full_name, now, now),
            )

        return True

    def get_pending(self, user_id: int) -> dict[str, Any] | None:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT
                    user_id,
                    username,
                    full_name,
                    requested_at,
                    updated_at
                FROM access_requests
                WHERE user_id = ?
                  AND status = 'pending'
                ORDER BY id DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()

        return row_to_dict(row)

    def list_pending(self) -> list[dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    user_id,
                    username,
                    full_name,
                    requested_at,
                    updated_at
                FROM access_requests
                WHERE status = 'pending'
                ORDER BY requested_at ASC
                """
            ).fetchall()

        return [dict(row) for row in rows]

    def approve_user(self, user_id: int, admin_id: int) -> dict[str, Any]:
        now = now_str()
        pending = self.get_pending(user_id)

        username = pending.get("username") if pending else None
        full_name = pending.get("full_name") if pending else None
        requested_at = pending.get("requested_at") if pending else None

        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO users (
                    user_id,
                    username,
                    full_name,
                    status,
                    source,
                    approved_by,
                    approved_at,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, 'active', 'admin_approval', ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    username = excluded.username,
                    full_name = excluded.full_name,
                    status = 'active',
                    source = 'admin_approval',
                    approved_by = excluded.approved_by,
                    approved_at = excluded.approved_at,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    username,
                    full_name,
                    admin_id,
                    now,
                    now,
                    now,
                ),
            )

            conn.execute(
                """
                UPDATE access_requests
                SET status = 'approved',
                    decided_by = ?,
                    decided_at = ?,
                    updated_at = ?
                WHERE user_id = ?
                  AND status = 'pending'
                """,
                (admin_id, now, now, user_id),
            )

            self._log_admin_action(
                conn=conn,
                admin_id=admin_id,
                action="approve_user",
                target_user_id=user_id,
                details={
                    "username": username,
                    "full_name": full_name,
                },
            )

        return {
            "user_id": user_id,
            "username": username,
            "full_name": full_name,
            "requested_at": requested_at,
            "approved_by": admin_id,
            "approved_at": now,
            "status": "approved",
        }

    def deny_user(self, user_id: int, admin_id: int) -> dict[str, Any] | None:
        now = now_str()
        pending = self.get_pending(user_id)

        with get_connection() as conn:
            conn.execute(
                """
                UPDATE access_requests
                SET status = 'denied',
                    decided_by = ?,
                    decided_at = ?,
                    updated_at = ?
                WHERE user_id = ?
                  AND status = 'pending'
                """,
                (admin_id, now, now, user_id),
            )

            self._log_admin_action(
                conn=conn,
                admin_id=admin_id,
                action="deny_user",
                target_user_id=user_id,
                details=pending or {},
            )

        if not pending:
            return None

        pending["denied_by"] = admin_id
        pending["denied_at"] = now

        return pending

    def allow_user(
        self,
        user_id: int,
        admin_id: int,
        username: str | None = None,
        full_name: str | None = None,
    ) -> dict[str, Any]:
        now = now_str()

        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO users (
                    user_id,
                    username,
                    full_name,
                    status,
                    source,
                    approved_by,
                    approved_at,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, 'active', 'manual_admin_command', ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    username = excluded.username,
                    full_name = excluded.full_name,
                    status = 'active',
                    source = 'manual_admin_command',
                    approved_by = excluded.approved_by,
                    approved_at = excluded.approved_at,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    username,
                    full_name,
                    admin_id,
                    now,
                    now,
                    now,
                ),
            )

            conn.execute(
                """
                UPDATE access_requests
                SET status = 'approved',
                    decided_by = ?,
                    decided_at = ?,
                    updated_at = ?
                WHERE user_id = ?
                  AND status = 'pending'
                """,
                (admin_id, now, now, user_id),
            )

            self._log_admin_action(
                conn=conn,
                admin_id=admin_id,
                action="allow_user",
                target_user_id=user_id,
                details={
                    "username": username,
                    "full_name": full_name,
                },
            )

        return {
            "user_id": user_id,
            "username": username,
            "full_name": full_name,
            "approved_by": admin_id,
            "approved_at": now,
            "status": "approved",
            "source": "manual_admin_command",
        }

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

        now = now_str()

        with get_connection() as conn:
            user_row = conn.execute(
                """
                SELECT user_id
                FROM users
                WHERE user_id = ?
                  AND status = 'active'
                """,
                (user_id,),
            ).fetchone()

            pending_row = conn.execute(
                """
                SELECT id
                FROM access_requests
                WHERE user_id = ?
                  AND status = 'pending'
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()

            if not user_row and not pending_row:
                return False, "Пользователь не найден в списке доступа или заявок."

            conn.execute(
                """
                UPDATE users
                SET status = 'revoked',
                    updated_at = ?
                WHERE user_id = ?
                """,
                (now, user_id),
            )

            conn.execute(
                """
                UPDATE access_requests
                SET status = 'revoked',
                    updated_at = ?,
                    decided_at = ?
                WHERE user_id = ?
                  AND status = 'pending'
                """,
                (now, now, user_id),
            )

            self._log_admin_action(
                conn=conn,
                admin_id=None,
                action="revoke_user",
                target_user_id=user_id,
                details={},
            )

        return True, "Пользователь удалён."

    def list_users(self) -> list[dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    user_id,
                    username,
                    full_name,
                    source,
                    approved_by,
                    approved_at,
                    created_at,
                    updated_at
                FROM users
                WHERE status = 'active'
                ORDER BY approved_at ASC, created_at ASC
                """
            ).fetchall()

        return [dict(row) for row in rows]

    def _log_admin_action(
        self,
        conn,
        admin_id: int | None,
        action: str,
        target_user_id: int | None,
        details: dict[str, Any],
    ) -> None:
        conn.execute(
            """
            INSERT INTO admin_actions (
                created_at,
                admin_id,
                action,
                target_user_id,
                details
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                now_str(),
                admin_id,
                action,
                target_user_id,
                json.dumps(details, ensure_ascii=False),
            ),
        )