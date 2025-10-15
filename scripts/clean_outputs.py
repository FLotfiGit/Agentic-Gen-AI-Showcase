#!/usr/bin/env python3
import os
ROOT = os.path.dirname(os.path.dirname(__file__))
out = os.path.join(ROOT, 'outputs')
for f in os.listdir(out):
    if f == '.gitkeep':
        continue
    p = os.path.join(out, f)
    try:
        if os.path.isfile(p):
            os.remove(p)
    except Exception as e:
        print('Failed to remove', p, e)
print('Cleaned outputs/.')
