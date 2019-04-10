import utils

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
DOS_ORDER    = [0x0, 0x7, 0xe, 0x6, 0xd, 0x5, 0xc, 0x4,
                0xb, 0x3, 0xa, 0x2, 0x9, 0x1, 0x8, 0xf]

REV_DOS = [ DOS_ORDER.index(i) for i in range(16)]

PRODOS_ORDER = [0x0, 0x8, 0x1, 0x9, 0x2, 0xa, 0x3, 0xb,
                0x4, 0xc, 0x5, 0xd, 0x6, 0xe, 0x7, 0xf]

REV_PRODOS = [ PRODOS_ORDER.index(i) for i in range(16)]


def reorder_disk( org_disk : bytearray, sector_order):
    generic_disk = bytearray(143360)

    for track in range(35):
        for sector in range(16):
            ndx_src = sector * 256 + track * 16 * 256
            ndx = sector_order[sector] * 256 + track * 16 * 256

            generic_disk[ndx:ndx+256] = org_disk[ndx_src:ndx_src+256]

    return generic_disk



with open('STARTUP','rb') as fi:
    f = fi.read(65536)

    code = f[0:4096]

    data = f[(0x4000 - 0xC00):]

    print("Data start at ${:X}, len={}".format(0x4000 - 0xC00, len(data)))

    with open('WIZ4','wb') as fo:
        fo.write( code)

    with open('BADATA','wb') as fo:
        fo.write( data)


print("Reading NEW.DSK")
with open( 'NEW.DSK','rb') as fi:
    orig = bytearray( fi.read())
    disk = reorder_disk( orig, REV_DOS)

    s = len(data) // 256
    if len(data) % 256 != 0:
        s += 1

    t = s // 16
    if s % 16 != 0:
        t += 1

    ofs = (35 - t)*256*16

    print("Data starts on track {}, it is {} sectors long".format(35-t, s))

    disk[ ofs:ofs+len(data) ] = data[:]

    disk2 = reorder_disk( disk, DOS_ORDER)

    print("Writing NEW2.DSK")
    with open( 'NEW2.DSK','wb') as fo:
        fo.write(disk2)
