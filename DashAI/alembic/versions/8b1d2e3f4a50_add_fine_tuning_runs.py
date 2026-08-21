"""Add fine-tuning runs and generative adapter references.

Revision ID: 8b1d2e3f4a50
Revises: d5b3c8f2a041
Create Date: 2026-08-19 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "8b1d2e3f4a50"
down_revision: Union[str, None] = "d5b3c8f2a041"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "fine_tuning_run",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("base_model_id", sa.String(), nullable=False),
        sa.Column("base_model_revision", sa.String(), nullable=False),
        sa.Column("resolved_model_revision", sa.String(), nullable=True),
        sa.Column("method", sa.String(), nullable=False),
        sa.Column("dataset_mapping", sa.JSON(), nullable=False),
        sa.Column("training_parameters", sa.JSON(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "not_started",
                "queued",
                "running",
                "completed",
                "failed",
                "canceled",
                name="finetuningstatus",
            ),
            nullable=False,
        ),
        sa.Column("huey_id", sa.String(), nullable=True),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("progress_message", sa.String(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("runtime_metadata", sa.JSON(), nullable=True),
        sa.Column("artifact_path", sa.String(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("cancellation_requested", sa.Boolean(), nullable=False),
        sa.Column("created", sa.DateTime(), nullable=False),
        sa.Column("last_modified", sa.DateTime(), nullable=False),
        sa.Column("start_time", sa.DateTime(), nullable=True),
        sa.Column("end_time", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["dataset.id"],
            name=op.f("fk_fine_tuning_run_dataset_id_dataset"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_fine_tuning_run")),
        sa.UniqueConstraint("name", name=op.f("uq_fine_tuning_run_name")),
    )
    with op.batch_alter_table("generative_session") as batch_op:
        batch_op.add_column(
            sa.Column("fine_tuning_run_id", sa.Integer(), nullable=True)
        )
        batch_op.create_foreign_key(
            op.f("fk_generative_session_fine_tuning_run_id_fine_tuning_run"),
            "fine_tuning_run",
            ["fine_tuning_run_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index(
            op.f("ix_generative_session_fine_tuning_run_id"),
            ["fine_tuning_run_id"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("generative_session") as batch_op:
        batch_op.drop_index(op.f("ix_generative_session_fine_tuning_run_id"))
        batch_op.drop_constraint(
            op.f("fk_generative_session_fine_tuning_run_id_fine_tuning_run"),
            type_="foreignkey",
        )
        batch_op.drop_column("fine_tuning_run_id")
    op.drop_table("fine_tuning_run")
