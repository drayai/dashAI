"""Add managed_local_model table and generative_session.local_model_id

Revision ID: c3f5a9b7d2e1
Revises: 8b1d2e3f4a50
Create Date: 2026-08-21 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3f5a9b7d2e1"
down_revision: Union[str, None] = "8b1d2e3f4a50"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "managed_local_model",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_key", sa.String(), nullable=False),
        sa.Column("base_model_revision", sa.String(), nullable=False),
        sa.Column("resolved_revision", sa.String(), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "downloading",
                "ready",
                "error",
                name="managedmodelstatus",
            ),
            nullable=False,
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("created", sa.DateTime(), nullable=False),
        sa.Column("last_modified", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_managed_local_model")),
        sa.UniqueConstraint(
            "model_key",
            "base_model_revision",
            name="uq_managed_model_key_revision",
        ),
    )
    with op.batch_alter_table("generative_session") as batch_op:
        batch_op.add_column(
            sa.Column("local_model_id", sa.Integer(), nullable=True)
        )
        batch_op.create_foreign_key(
            op.f("fk_generative_session_local_model_id_managed_local_model"),
            "managed_local_model",
            ["local_model_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index(
            op.f("ix_generative_session_local_model_id"),
            ["local_model_id"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("generative_session") as batch_op:
        batch_op.drop_index(op.f("ix_generative_session_local_model_id"))
        batch_op.drop_column("local_model_id")
    op.drop_table("managed_local_model")
