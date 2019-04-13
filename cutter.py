import sys
from utils import *

# 0,1,2 logical
# 0000,0700,0e00, ?, $7700, $7600, ?, $7400

# Mem                            Disk
# 0000 : 01 38 B0 03             0000
# 0100 : 03 00 05 00             0700
# 0200 : 3F 09 26 50             0e00
# 0300 : 00 00 00 00
# 0400 : 00 00 00 00
# 0500 : 04 00 00 00             0500
# 0600 : 00 00 00 00
# 0700 : 00 00 00 00
# 0800 : 00 00 03 00 f3 41...    0B00

DISK_SIZE = 143360
# Allow some wrapping so that the ASM code is simpler
extra_bytes = 3

DOS_ORDER    = [0x0, 0x7, 0xe, 0x6, 0xd, 0x5, 0xc, 0x4,
                0xb, 0x3, 0xa, 0x2, 0x9, 0x1, 0x8, 0xf]

REV_DOS = [ DOS_ORDER.index(i) for i in range(16)]

PRODOS_ORDER = [0x0, 0x8, 0x1, 0x9, 0x2, 0xa, 0x3, 0xb,
                0x4, 0xc, 0x5, 0xd, 0x6, 0xe, 0x7, 0xf]

REV_PRODOS = [ PRODOS_ORDER.index(i) for i in range(16)]

def fit_in_disk( a : bytearray):
    if len( a) < DISK_SIZE:
        # Pad with zeros
        a.extend( bytearray(  DISK_SIZE - len( a)))
        assert len(a) == DISK_SIZE
        return a
    elif len( a) > DISK_SIZE:
        return a[0:DISK_SIZE]
    else:
        return a

def at_page( a : bytearray):

    if type(a) != bytearray:
        a = bytearray(a)

    if len(a) % 256:
        a.extend( bytearray( 256 - (len(a) % 256)))
    return a

def offset_to_track_sector( o):
    assert 0 <= o <= DISK_SIZE

    t = (o // 256) // 16
    s = (o // 256) % 16
    return t,s

def reorder_disk( org_disk : bytearray, sector_order):
    assert len(org_disk) == 143360, "Expected something else than {}".format(len(org_disk))

    generic_disk = bytearray(143360)

    for track in range(35):
        for sector in range(16):
            ndx_src = sector * 256 + track * 16 * 256
            ndx = sector_order[sector] * 256 + track * 16 * 256

            generic_disk[ndx:ndx+256] = org_disk[ndx_src:ndx_src+256]

    return generic_disk


b_mem = []

print( sys.argv)

with open('STARTUP','rb') as fi:
    f = fi.read(65536)

    if sys.argv[1] == 'cut':
        code = f[0: f.index('DATADATA'.encode('ASCII')) +1]
        assert len(code) < 0x2000 - 0xC00, "Code starting at C00 won't fit in memory"
        print("Code is {} bytes long".format( len( code)))
        with open('WIZ4','wb') as fo:
            fo.write( code)

    data = at_page( f[(0x4000 - 0xC00):])

    print("Data start at ${:X}, len={}".format(0x4000 - 0xC00, len(data)))


with open("cstripes.data", "rb") as fi:
    b = bytearray( fi.read())

    if len(b) > DISK_SIZE:
        print("Need additional RAM to fit stream data : {} bytes".format(len(b)-DISK_SIZE))
        too_much = min( 8192+2048+512, len(b)-DISK_SIZE)
        too_much = len(b)-DISK_SIZE

        if too_much % 256:
            too_much = ((too_much // 256) + 1) * 256
            print("Aligned 'too much' on {}".format(too_much))

        # Cut bytes at the beginning of the stream

        b_mem = bytearray()
        b_mem.extend( b[ 0 : too_much])

        extra = bytearray(256)
        extra[0:extra_bytes] = b[ too_much : too_much + extra_bytes]
        b_mem.extend( extra)

        b_disk = fit_in_disk( b[too_much : ])

    else:
        b_disk = b
        b_mem = bytearray()

    print("Writing BAD_APPLE_DATA.DSK")
    with open("BAD_APPLE_DATA.DSK","bw") as fo:
        fo.write( reorder_disk( fit_in_disk( b_disk), DOS_ORDER))


if sys.argv[1] == 'disk':
    print("Reading NEW.DSK")
    with open( 'NEW.DSK','rb') as fi:
        orig = bytearray( fi.read())
        disk = reorder_disk( orig, REV_DOS)

        print("stripes dictionary = {} bytes ({} sectors); data from disk = {} bytes ({} sectors)".format( len(data), len(data)//256, len(b_mem), len(b_mem)//256))

        assert len(data) % 256 == 0
        assert len(b_mem) % 256 == 0

        print("stripes data : {}".format( hex_byte( list( data[0:64]))))
        print("disk data    : {}".format( hex_byte( list( b_mem[0:64]))))

        disk[ DISK_SIZE - (len(data) + len(b_mem)): DISK_SIZE] = (data + b_mem)[:]

        print( offset_to_track_sector(DISK_SIZE - (len(data) + len(b_mem))))
        print( offset_to_track_sector(DISK_SIZE - (len(b_mem))))


        disk2 = reorder_disk( disk, DOS_ORDER)

        print("Writing BAD_APPLE.DSK")
        with open( 'BAD_APPLE.DSK','wb') as fo:
            fo.write(disk2)
