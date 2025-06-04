import re
from typing import Optional
import ast
import os
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SOURCE = os.path.join(ROOT_DIR, 'Quik_App_v2.py')

with open(SOURCE, 'r') as f:
    tree = ast.parse(f.read(), filename='Quik_App_v2.py')

extracted_body = []
for node in tree.body:
    if isinstance(node, ast.Assign):
        # capture ALLOWED_EXTENSIONS
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if 'ALLOWED_EXTENSIONS' in targets:
            extracted_body.append(node)
    if isinstance(node, ast.FunctionDef) and node.name in {'allowed_file', 'extract_url'}:
        extracted_body.append(node)

module = ast.Module(body=extracted_body, type_ignores=[])
namespace = {"re": re, "Optional": Optional}
exec(compile(module, '<extracted>', 'exec'), namespace)

allowed_file = namespace['allowed_file']
extract_url = namespace['extract_url']


@pytest.mark.parametrize(
    'filename,expected',
    [
        ('document.txt', True),
        ('report.pdf', True),
        ('notes.doc', True),
        ('slides.docx', True),
        ('CAPITAL.PDF', True),
        ('archive', False),
        ('image.png', False),
        ('malware.exe', False),
        ('noextension', False),
    ],
)
def test_allowed_file(filename, expected):
    assert allowed_file(filename) is expected


@pytest.mark.parametrize(
    'text,expected',
    [
        ('Check this http://example.com/path', 'http://example.com/path'),
        (
            'Visit https://sub.domain.com/page?query=1',
            'https://sub.domain.com/page?query=1',
        ),
        (
            'Multiple http://first.com and https://second.com',
            'http://first.com',
        ),
        ('No url here', None),
        ('Malformed www.example.com', None),
        ('https://', None),
    ],
)
def test_extract_url(text, expected):
    assert extract_url(text) == expected
