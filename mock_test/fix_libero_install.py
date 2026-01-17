"""
LIBERO Installation Fix Script

诊断并修复LIBERO editable安装问题。

Usage:
    python mock_test/fix_libero_install.py
"""

import sys
import os
from pathlib import Path

print('=' * 60)
print('LIBERO Installation Fix')
print('=' * 60)

# Find site-packages
site_packages = None
for path in sys.path:
    if 'site-packages' in path and os.path.isdir(path):
        site_packages = Path(path)
        break

if not site_packages:
    print('❌ Could not find site-packages directory')
    sys.exit(1)

print(f'\n📦 Site-packages: {site_packages}')

# Check editable install files
pth_file = site_packages / '__editable__.libero-0.1.0.pth'
finder_file = site_packages / '__editable___libero_0_1_0_finder.py'

print(f'\n🔍 Checking editable install files:')
print(f'   .pth file: {pth_file.exists()} - {pth_file}')
print(f'   finder file: {finder_file.exists()} - {finder_file}')

if pth_file.exists():
    print(f'\n📄 Content of {pth_file.name}:')
    with open(pth_file, 'r') as f:
        content = f.read()
        print('   ' + content.replace('\n', '\n   '))

    # Check if paths in .pth file actually exist
    print(f'\n🔍 Checking if paths exist:')
    for line in content.strip().split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            exists = os.path.exists(line)
            print(f'   {line}: {"✅ Exists" if exists else "❌ NOT FOUND"}')

            if exists and os.path.isdir(line):
                # Check if libero package exists in that directory
                libero_path = Path(line) / 'libero'
                if libero_path.exists():
                    print(f'      → Found libero package at: {libero_path}')
                else:
                    print(f'      → ❌ No libero package found in this directory')

print('\n' + '=' * 60)
print('🔧 Suggested fix:')
print('=' * 60)

print('''
If the paths in .pth file don't exist, you need to reinstall LIBERO:

1. Uninstall current broken installation:
   pip uninstall libero -y

2. Clone LIBERO to a permanent location:
   cd /workspace
   git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

3. Install in editable mode:
   cd LIBERO
   pip install -e .

4. Verify installation:
   python -c "from libero.libero import benchmark; print('✅ Success!')"
''')

print('\n💡 Quick fix command (copy-paste):')
print('=' * 60)
print('''
pip uninstall libero -y && \\
cd /workspace && \\
rm -rf LIBERO && \\
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && \\
cd LIBERO && \\
pip install -e . && \\
python -c "from libero.libero import benchmark; print('✅ LIBERO installed successfully!')"
''')
