from utils import *

# Normal characters
m = dict()
m[' '] = 0xA0
for x in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    m[x] = ord(x) - ord('A') + 0xC1

# Flashing characters
m = dict()
m[' '] = 0x60
for x in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    m[x] = ord(x) - ord('A') + 0x41

print( hex_byte( [m[x] for x in "FLIP DISK AND HIT A KEY"]))
