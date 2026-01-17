"""
LIBERO Import Diagnostics Script

用于诊断LIBERO导入问题。

Usage:
    python mock_test/diagnose_libero.py
"""

import sys
import os

print('=' * 60)
print('LIBERO Import Diagnostics')
print('=' * 60)

# Test 1: Try import LIBERO (uppercase)
print('\n1️⃣ Testing: import LIBERO')
try:
    import LIBERO
    print('   ✅ Success!')
    print(f'   Location: {LIBERO.__file__ if hasattr(LIBERO, "__file__") else "No __file__ attribute"}')
    print(f'   Type: {type(LIBERO)}')
    print(f'   Dir: {dir(LIBERO)[:10]}')
except ImportError as e:
    print(f'   ❌ Failed: {e}')

# Test 2: Try import libero (lowercase)
print('\n2️⃣ Testing: import libero')
try:
    import libero
    print('   ✅ Success!')
    print(f'   Location: {libero.__file__ if hasattr(libero, "__file__") else "No __file__ attribute"}')
    print(f'   Type: {type(libero)}')
    print(f'   Dir: {dir(libero)[:10]}')
except ImportError as e:
    print(f'   ❌ Failed: {e}')

# Test 3: Try import libero.libero
print('\n3️⃣ Testing: from libero.libero import benchmark')
try:
    from libero.libero import benchmark
    print('   ✅ Success!')
    print(f'   benchmark type: {type(benchmark)}')
except ImportError as e:
    print(f'   ❌ Failed: {e}')

# Test 4: Check sys.path
print('\n4️⃣ Python sys.path:')
for i, p in enumerate(sys.path[:8]):
    print(f'   [{i}] {p}')

# Test 5: Check if LIBERO package exists in site-packages
print('\n5️⃣ Searching for libero packages:')
for path in sys.path:
    if os.path.isdir(path):
        try:
            items = [f for f in os.listdir(path) if 'libero' in f.lower()]
            if items:
                print(f'   In {path}:')
                for item in items:
                    print(f'      - {item}')
        except PermissionError:
            pass

# Test 6: Check environment variables
print('\n6️⃣ Environment variables:')
libero_env_vars = {k: v for k, v in os.environ.items() if 'LIBERO' in k.upper()}
if libero_env_vars:
    for k, v in libero_env_vars.items():
        print(f'   {k} = {v}')
else:
    print('   No LIBERO-related environment variables found')

print('\n' + '=' * 60)
print('Diagnostics complete!')
print('=' * 60)
