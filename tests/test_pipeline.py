"""
Unit tests for the data generation pipeline.

These tests verify the core functionality of the tokamak disruption
data generation pipeline.
"""

import pytest


class TestPipelineStructure:
    """Tests for verifying project structure and imports."""

    def test_src_imports(self):
        """Test that main source modules can be imported."""
        import src
        assert hasattr(src, '__version__')

    def test_submodule_imports(self):
        """Test that submodules can be imported."""
        import src.dina
        import src.dream
        import src.pipeline
        import src.utils

    def test_version_format(self):
        """Test that version follows semantic versioning."""
        import src
        version_parts = src.__version__.split('.')
        assert len(version_parts) == 3
        for part in version_parts:
            assert part.isdigit()


class TestDirectoryStructure:
    """Tests for verifying directory structure."""

    def test_configs_directory_exists(self):
        """Test that configs directory exists."""
        from pathlib import Path
        configs_dir = Path(__file__).parent.parent / 'configs'
        assert configs_dir.exists()

    def test_data_directory_exists(self):
        """Test that data directory exists."""
        from pathlib import Path
        data_dir = Path(__file__).parent.parent / 'data'
        assert data_dir.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
