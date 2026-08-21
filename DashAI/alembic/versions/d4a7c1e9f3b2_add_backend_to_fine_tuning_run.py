"""Add backend column to fine_tuning_run

Revision ID: d4a7c1e9f3b2
Revises: c3f5a9b7d2e1
Create Date: 2026-08-21 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4a7c1e9f3b2"
down_revision: Union[str, None] = "c3f5a9b7d2e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("fine_tuning_run") as batch_op:
        batch_op.add_column(
            sa.Column(
                "backend",
                sa.Enum(
                    "transformers",
                    "unsloth",
                    name="finetuningbackendtype",
                ),
                nullable=False,
                server_default="transformers",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("fine_tuning_run") as batch_op:
        batch_op.drop_column("backend")
