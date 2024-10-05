import pytest
import tempfile
import os
import json
from unittest.mock import patch, Mock
from io import StringIO
import requests

# Import the functions you want to test
from src.fetch_images_database import get_keys, fetch_image

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

def test_get_keys():
    mock_data = [
        '{"product": "0027e30879ce3d87f82f699f148bff7e", "scene": "cdab9160072dd1800038227960ff6467"}\n',
        '{"product": "0027e30879ce3d87f82f699f148bff7e", "scene": "14f59334af4539132981b1324a731067"}\n',
        '{"product": "0027e30879ce3d87f82f699f148bff7e", "scene": "e7d32df9f45b691afc580808750f73ca"}\n'
    ]
    mock_file = StringIO(''.join(mock_data))

    with patch('builtins.open', return_value=mock_file):
        result = get_keys('dummy_file.json', 3)

    expected = frozenset(['0027e30879ce3d87f82f699f148bff7e', 'cdab9160072dd1800038227960ff6467', 
                          '0027e30879ce3d87f82f699f148bff7e', '14f59334af4539132981b1324a731067', 
                          '0027e30879ce3d87f82f699f148bff7e', 'e7d32df9f45b691afc580808750f73ca'])
    assert result == expected

def test_get_keys_max_lines():
    mock_data = [
        '{"product": "0027e30879ce3d87f82f699f148bff7e", "scene": "cdab9160072dd1800038227960ff6467"}\n',
        '{"product": "0027e30879ce3d87f82f699f148bff7e", "scene": "14f59334af4539132981b1324a731067"}\n',
        '{"product": "0027e30879ce3d87f82f699f148bff7e", "scene": "e7d32df9f45b691afc580808750f73ca"}\n'
    ]
    mock_file = StringIO(''.join(mock_data))

    with patch('builtins.open', return_value=mock_file):
        result = get_keys('dummy_file.json', 2)

    expected = frozenset(['0027e30879ce3d87f82f699f148bff7e', 'cdab9160072dd1800038227960ff6467', 
                          '0027e30879ce3d87f82f699f148bff7e', '14f59334af4539132981b1324a731067'])
    assert result == expected

@patch('requests.get')
@patch('src.pin_util.key_to_url')
def test_fetch_image_success(mock_key_to_url, mock_get, temp_dir):
    mock_key_to_url.return_value = 'http://i.pinimg.com/400x/00/27/e3/0027e30879ce3d87f82f699f148bff7e.jpg'
    mock_response = Mock()
    mock_response.content = b'image_content'
    mock_get.return_value = mock_response

    result = fetch_image('0027e30879ce3d87f82f699f148bff7e', temp_dir, 10)

    assert result == True
    assert os.path.exists(os.path.join(temp_dir, '0027e30879ce3d87f82f699f148bff7e.jpg'))

@patch('requests.get')
@patch('src.pin_util.key_to_url')
def test_fetch_image_already_exists(mock_key_to_url, mock_get, temp_dir):
    # Create a dummy file
    with open(os.path.join(temp_dir, '0027e30879ce3d87f82f699f148bff7e.jpg'), 'w') as f:
        f.write('dummy content')

    result = fetch_image('0027e30879ce3d87f82f699f148bff7e', temp_dir, 1)

    assert result == False
    mock_get.assert_not_called()