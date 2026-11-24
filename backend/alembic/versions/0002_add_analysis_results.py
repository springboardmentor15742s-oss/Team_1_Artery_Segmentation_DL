"""add analysis_results table

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-13 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, Sequence[str], None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "analysis_results",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("case_id", sa.Integer(), nullable=False),
        sa.Column("image_id", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("slice_index", sa.Integer(), nullable=True),
        sa.Column("inference_duration_ms", sa.Float(), nullable=True),
        sa.Column("mask_path", sa.String(length=512), nullable=True),
        sa.Column("overlay_path", sa.String(length=512), nullable=True),
        sa.Column("foreground_pixels", sa.Integer(), nullable=True),
        sa.Column("centerline_status", sa.String(length=64), nullable=True),
        sa.Column("centerline_px", sa.Integer(), nullable=True),
        sa.Column("mean_diameter_px", sa.Float(), nullable=True),
        sa.Column("mean_diameter_mm", sa.Float(), nullable=True),
        sa.Column("mld_px", sa.Float(), nullable=True),
        sa.Column("mld_mm", sa.Float(), nullable=True),
        sa.Column("reference_diameter_px", sa.Float(), nullable=True),
        sa.Column("reference_diameter_mm", sa.Float(), nullable=True),
        sa.Column("stenosis_pct", sa.Float(), nullable=True),
        sa.Column("severity_category", sa.String(length=64), nullable=True),
        sa.Column("pixel_spacing_mm", sa.Float(), nullable=True),
        sa.Column("physical_spacing_available", sa.Boolean(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["case_id"], ["cases.id"]),
        sa.ForeignKeyConstraint(["image_id"], ["images.id"]),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_analysis_results_case_id"), "analysis_results", ["case_id"], unique=False)
    op.create_index(op.f("ix_analysis_results_image_id"), "analysis_results", ["image_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_analysis_results_image_id"), table_name="analysis_results")
    op.drop_index(op.f("ix_analysis_results_case_id"), table_name="analysis_results")
    op.drop_table("analysis_results")
