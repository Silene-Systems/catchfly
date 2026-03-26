"""Tests for catchfly._parsing utilities."""

from catchfly._parsing import strip_markdown_fences


class TestStripMarkdownFences:
    def test_plain_json_unchanged(self):
        text = '{"key": "value"}'
        assert strip_markdown_fences(text) == '{"key": "value"}'

    def test_json_fence_stripped(self):
        text = '```json\n{"key": "value"}\n```'
        assert strip_markdown_fences(text) == '{"key": "value"}'

    def test_bare_fence_stripped(self):
        text = '```\n{"key": "value"}\n```'
        assert strip_markdown_fences(text) == '{"key": "value"}'

    def test_python_fence_stripped(self):
        text = '```python\nprint("hello")\n```'
        assert strip_markdown_fences(text) == 'print("hello")'

    def test_leading_trailing_whitespace_trimmed(self):
        text = '  \n  {"key": "value"}  \n  '
        assert strip_markdown_fences(text) == '{"key": "value"}'

    def test_empty_string(self):
        assert strip_markdown_fences("") == ""

    def test_multiline_content_preserved(self):
        text = '```json\n{"a": 1,\n"b": 2}\n```'
        assert strip_markdown_fences(text) == '{"a": 1,\n"b": 2}'

    def test_no_fence_with_backticks_in_content(self):
        text = 'Use `code` here'
        assert strip_markdown_fences(text) == 'Use `code` here'
