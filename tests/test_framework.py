"""pytest 框架验证测试"""
import pytest


class TestFramework:
    """验证 pytest 框架配置正确"""

    def test_basic_assertion(self):
        """基础断言测试"""
        assert 1 + 1 == 2

    def test_import_src(self):
        """验证可以导入 src 模块"""
        import src
        assert hasattr(src, '__version__')
        assert src.__version__ == "0.1.0"

    @pytest.mark.unit
    def test_marker(self):
        """验证 marker 配置"""
        assert True
