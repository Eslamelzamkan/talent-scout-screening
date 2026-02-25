"""add candidate session/final_score indexes

Revision ID: 0002_candidate_indexes
Revises: 0001_initial_schema
Create Date: 2026-02-25 00:30:00
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "0002_candidate_indexes"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("idx_candidates_session_id", "candidates", ["session_id"], unique=False)
    op.create_index("idx_candidates_final_score", "candidates", ["final_score"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_candidates_final_score", table_name="candidates")
    op.drop_index("idx_candidates_session_id", table_name="candidates")
