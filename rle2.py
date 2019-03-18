
""" C64 : 320*200 with 8x8 chars => 40 * 25
    Apple : 40x48

128 bytes pour 6 lignes = 40+40+40 + 8 bytes lost every 6 lines. There are 48 lines => (48/6)*8=64 bytes lost out of 48*24=1152 => 5%.


Je pense qu'un stripe moyen n'est pas plus fréquent qu'un strip long : les deux sont en fait rares (donc ne se répètent pas, donc ne compresse rien).
Par conséquent, diminuer la taille des stripes n'a pas bcp d'importanvce pour la taille du codebook.
Donc, de toute façon, on stocke bcp de strip uniques dans le codebook.
Donc optimiser la structure du codeboook n'est pas très utile.
Il vaut mieux optimiser les stripes.
Ce qui est semblable à optimiser des tiles qui auraient la lnogueur de stripes :-)

ffmpeg -i bad_apple_original.mp4 -an  -vf fps=10,scale=36:46  /tmp/bad_apple%05d.png

1 side x 35 Tracks/side x 16 Sectors/Trk x 256 Bytes/Sec = 143,360 Bytes.
prodos data disk : 139776

"""
import platform
import collections
import bitstring
import math
import pprint
import hashlib
import time
import os
from scipy.cluster import vq
import numpy as np
from PIL import Image
import glob
import curses
import huffman
from array import array
from bidict import bidict

MAX_PICTURES=15000 # 275 maximum one can put in a normal BIN file.
SKIP_PICTURES=0
MAX_DISK_SIZE = 280*512 - 4096 # bytes, 4096 = "catalog track" = 139264-
MAX_DISK_SIZE = 16*35*256 # 143360 if I manage the whole disk myself; else :  137216  According to Beneath Apple Prodos, for a ProDos data disk.

# prodos : 0c00 => 9600 => 9600 - 0c00 = 08A00 = 35328 bytes max binary BRUN'able by ProDos
# => have to write my own loader
MAXIMUM_RAM=(0xC000 - 0x800) + (0xFFFF - 0xD000) + (0x400-0x200) # = 59903
MUSIC_SIZE = 4000 #bytes

WIDTH=36 # In pixels, width of the original video
HEIGHT=46

"""
See Apple Prodos source code page 280 for RWTS source code (or something that looks like it)
"""


def sound_table():
    cpu_freq = 1000000 # cycles / seconds
    base_frequencies = [261, 277, 293, 311, 329, 349, 369, 392, 415, 440, 466, 493] # Hertz
    frequencies = [f/2 for f in base_frequencies]
    frequencies.extend( base_frequencies)
    frequencies.extend( [f*2 for f in base_frequencies])

    cycles_per_loop = 35
    note_duration = 0.0125 # seconds

    inner_loop_data = []
    outer_loop_data = []

    for freq in frequencies:
        cycles = cpu_freq / freq

        # Time between two clicks
        inner_loops = cycles / cycles_per_loop

        # How many loops to reach the duration

        outer_loops = (cpu_freq * note_duration) / cycles

        inner_loop_data.append( inner_loops)
        outer_loop_data.append( outer_loops)

        print("f:{} il:{} ol:{} - {}".format( freq, inner_loops, outer_loops, inner_loops*outer_loops))

    print( "inner_loops !byte " + ",".join( [str( int(x)) for x in inner_loop_data]))
    print( "outer_loops !byte " + ",".join( [str( int(x)) for x in outer_loop_data]))

sound_table()
#exit()


def compress_code( code):
    if code < 2 ** 3:
        # 0xxxb
        return bitstring.BitArray(length=4, uint=code)
    elif code < 2 ** 6:
        # 10yy yyyyb 8 bits
        return bitstring.BitArray(length=8, uint=0b10000000 + code)
    elif code < 2 ** 9:
        # 110z zzzz zzzz 12 bits
        return bitstring.BitArray(length=12, uint=0b110000000000 + code)
    elif code < 2 ** 13:
        # 111z zzzz zzzz zzzz 16 bits
        return bitstring.BitArray(length=16, uint=0b1110000000000000 + code)
    else:
        raise Exception("argh")



def blank_disk():
    disk = []
    for i in range(256):
        block = [i]* 512

        s = "HELLO {}".format(i).encode('ascii')

        for i in range( len( s)):
            block[i] = s[i]
        block[i+1] = 0xD
        block[i+2] = 0x0

        disk.extend( block)
    disk.extend( [0] * (143360 - len(disk)))
    with open("gdata.po","bw") as fo:
        fo.write( bytearray(disk))


def disk_2_dos( disk):
    print("disk_2_dos")
    dos_sector= [0x0, 0xd, 0xb, 0x9, 0x7, 0x5, 0x3, 0x1,
                 0xe, 0xc, 0xa, 0x8, 0x6, 0x4, 0x2, 0xf]

    dos_sector= [0x0, 0x7, 0xe, 0x6, 0xd, 0x5, 0xc, 0x4,
                 0xb, 0x3, 0xa, 0x2, 0x9, 0x1, 0x8, 0xf]

    DISK_SIZE = 143360

    if len(disk) > DISK_SIZE:
        print("Disk image too big by {} bytes, truncating...".format(len(disk) - DISK_SIZE))
        disk = disk[0:DISK_SIZE]
    elif len(disk) < DISK_SIZE:
        print("Disk image too small, extending...")
        disk.extend( bytearray( DISK_SIZE - len(disk)))

    # dos_sector = list( range( 16))
    disk_dos = bytearray( DISK_SIZE)

    for track in range(35):
        for sector in range(16):
            track_offset = track * 16 * 256

            dos_ofs = track_offset + dos_sector[sector]*256
            dsk_ofs = track_offset + sector*256
            disk_dos[ dos_ofs:dos_ofs+256] =  disk[ dsk_ofs:dsk_ofs+256] # [sector for i in range(256)]

    return disk_dos

def stripes_to_disk( stripes):

    disk = bytearray()

    for s in stripes[0:min( (len(stripes) // 2) - 1, (143360//2) - 1)]:
        sid = (s.stripe_id2 - 1) * 2
        assert sid < 65536
        disk.append( sid & 0xFF)
        disk.append( sid >> 8)

    disk.append( 0xFF)
    disk.append( 0xFF)
    disk.extend( bytearray( 143360 - len(disk)))

    with open("stripes.dsk","bw") as fo:
        fo.write( disk_2_dos( disk))

"""
bad apple resolution
562 x 386

je veux préserver aspect ratio :
bad_apple 217 à 777 = 560
applewin 217 à 817 = 600
=> 2.6 pixels GR de différence
=> je peux faire le resize de bad_apple vers => 37.4x48 ou pour fair eplus facile : 36*46


"""

# COMPRESSOR=1 # No packing (for test purposes)
# PIX_PER_BYTE=1 # Only horizontally
# LINES_PER_BYTE=1
# BYTES_PER_ROW = WIDTH  # 1 pixel per byte
# DECIMATE_ROWS_WITH_STRICT_MATCHER=True

# COMPRESSOR=2 # Regular Apple memory layout
# PIX_PER_BYTE=1 # Only horizontally
# LINES_PER_BYTE=2
# BYTES_PER_ROW = WIDTH  # because we store two vertical pixels in one byte.
# DECIMATE_ROWS_WITH_STRICT_MATCHER=False

COMPRESSOR=3 # Pack 4 pixels per byte
PIX_PER_BYTE=2 # Only horizontally
LINES_PER_BYTE=2
# From 2 lines of 40 pixels, I compress them to two lines of 20 bytes each, then I merge them into one of 20 pixels.
BYTES_PER_ROW = WIDTH // 2 # We store 4 pixels per byte, two lines at a time. (40/4 pix per bytes)*2 pix lines == 40/2
DECIMATE_ROWS_WITH_STRICT_MATCHER=False

# COMPRESSOR=4 # Testing
# PIX_PER_BYTE=2 # Only horizontally
# LINES_PER_BYTE=2
# BYTES_PER_ROW = WIDTH // 2  # 1 pixel per byte
# DECIMATE_ROWS_WITH_STRICT_MATCHER=True

# Decreasing the size of the stripes improve the codebook size significantly
# while increasing the size of the compressed data not significantly (so it's not symetric loss)
MAX_STRIPE_LENGTH = BYTES_PER_ROW // 2
print(MAX_STRIPE_LENGTH)

# we compress to rows of bytes. Each row of bytes corresponds
# to an integer number of pixel lines.

LINES_PER_ROW=LINES_PER_BYTE
PIXELS_PER_ROW = LINES_PER_BYTE * WIDTH
BYTES_PER_FRAME= (HEIGHT * WIDTH  // PIXELS_PER_ROW) * BYTES_PER_ROW

if COMPRESSOR == 1:
    ORIGINAL_LINE_ZAP_THRESHOLD = 0 # 0.3/PIXELS_PER_ROW # 1.0/WIDTH # 8.0, expressed in pixels / pixels per row
    DECIMATE_STRIPS_THRESHOLD_FACTOR = 0 # 1/40 # portion of stripe to allow to differ
elif COMPRESSOR == 4:
    ORIGINAL_LINE_ZAP_THRESHOLD = 0 # 0.3/PIXELS_PER_ROW # 1.0/WIDTH # 8.0, expressed in pixels / pixels per row
    DECIMATE_STRIPS_THRESHOLD_FACTOR = 0/40 # 1/40 # portion of stripe to allow to differ
elif COMPRESSOR == 3:
    ORIGINAL_LINE_ZAP_THRESHOLD = 0.3/PIXELS_PER_ROW # 1.0/WIDTH # 8.0, expressed in pixels / pixels per row
    DECIMATE_STRIPS_THRESHOLD_FACTOR = 1/40 # portion of stripe to allow to differ
else:
    ORIGINAL_LINE_ZAP_THRESHOLD = 0.3/PIXELS_PER_ROW # 1.0/WIDTH # 8.0, expressed in pixels / pixels per row
    DECIMATE_STRIPS_THRESHOLD_FACTOR = 0.5/40 # portion of stripe to allow to differ

colors = [ ' '*2, '\u2591'*2,'\u2592'*2,'\u2588'*2, '\u2588'*2 ]
#colors = [ ' {}'.format(i) for i in range(4)]

PIXEL_MATCHER = [ [ True, True, False, False ],
                  [ True, True,  True,  False ],
                  [ False, True,  True, False ],
                  [ False, False, False, True] ]


def ffmpeg():
    if not os.path.exists("/tmp/bad_apple00009.png"):
        if platform.system() == 'Linux':
            os.system("ffmpeg -i bad_apple_original.mp4 -an  -vf fps=10,scale={}:{}  /tmp/bad_apple%05d.png".format(WIDTH, HEIGHT))

def replace( packed_frame):
    d = [[0] * WIDTH, [0] * WIDTH]

    s = ".ggW"
    r = "..gW"


def compare_row( old_data, new_data):
    old_lines = split_bytes(old_data)
    new_lines = split_bytes(new_data)

    for i in range( len( old_lines)):
        for x in range( len( old_lines[i])):
            if not PIXEL_MATCHER[ old_lines[i][x]][ new_lines[i][x]]:
                return False

    return True

def decimate_rows( old_frame, new_frame):
    # The idea here is to give white and black more weight
    # when comparing pixels.

    m = [ [ True,  False, False, False ],
          [ False, True,  True,  False ],
          [ False, True,  True,  False ],
          [ False, False, False, True] ]

    decimated_rows = []

    for l in range(0, HEIGHT, LINES_PER_ROW):
        old_row = old_frame[ l*WIDTH : (l+LINES_PER_ROW)*WIDTH]
        new_row = new_frame[ l*WIDTH : (l+LINES_PER_ROW)*WIDTH]

        good = True
        for x in range(LINES_PER_ROW*WIDTH):
            if not m[old_row[x]][new_row[x]]:
                good=False
                break

        if good:
            decimated_rows.append(l // LINES_PER_ROW)
            new_frame[ l*WIDTH : (l+LINES_PER_ROW)*WIDTH] = old_row

    return decimated_rows


def curses_loop():
    stdscr.refresh()

    # codebook_ram = TILE_SIZE*TILE_SIZE*CODEBOOK_SIZE
    # print("Codebook size = {}".format( codebook_ram))
    # tile_ram = 40*48/(TILE_SIZE*TILE_SIZE)
    # print("Frame size = {}".format( tile_ram))
    # print("Frame in memeory size = {}".format( (34000 - codebook_ram) // tile_ram ))


    time.sleep(0.1)

    c = stdscr.getch()

    if c == ord('p'):
        c = ''
        while c != ord('p') and c != ord('q'):
            c = stdscr.getch()

    if c == ord('q'):
        return False

    return True


def show_pictures( decompressed, p = None):
    console_height = 44

    # assert len(decompressed) == HEIGHT , "Bad size : {}, expected {}".format(len(decompressed), HEIGHT)
    # assert len(decompressed[0]) == WIDTH, "Bad size : {}, expected {}".format(len(decompressed[0]), WIDTH)

    for y in range( min( console_height, len( decompressed))):
        l = ""

        data = decompressed[ y*WIDTH:y*WIDTH + WIDTH ]
        for x in range(WIDTH):
            l += colors[data[x]]

        if p is not None:

            l += " | "
            data = p[ y*WIDTH:y*WIDTH + WIDTH ]
            for x in range(WIDTH):
                l += colors[int(data[x])]

        l += " |"
        #print(l)

        stdscr.addstr(y,1,l)

def pack_2_pixels( p1, p2):
    return (p1 << 2) + p2

def unpack_2_pixels( b):
    b = b & 15
    return [b >> 2, b & 3]

def pack_4_pixels( pixels):
    return (pack_2_pixels( pixels[0][0], pixels[0][1]) << 4) + pack_2_pixels( pixels[1][0], pixels[1][1])

def pack_bytes( l1):
    if PIX_PER_BYTE == 2 and LINES_PER_BYTE == 2:

        assert len(l1) == PIXELS_PER_ROW, "Bad length {}, expected {}".format(len(l1), PIX_PER_BYTE*LINES_PER_BYTE*WIDTH)

        return [ (pack_2_pixels( l1[z], l1[z+1]) << 4) +
                  pack_2_pixels( l1[z+WIDTH], l1[z+WIDTH+1])  for z in range( 0,WIDTH,2) ]

    elif LINES_PER_BYTE == 2:
        assert len(l1) == WIDTH*2, "Bad length {}".format(len(l1))
        return [ pack_2_pixels( l1[z], l1[WIDTH+z]) for z in range( WIDTH) ]
    else:
        return [ l1[z] for z in range( len( l1)) ]

def split_bytes( data):
    if PIX_PER_BYTE == 2 and LINES_PER_BYTE == 2:
        # assert len(data) == BYTES_PER_ROW, "Expected {}, got {}".format(BYTES_PER_ROW, len(data))
        top = []
        bottom = []
        for b in data:
            top.extend( unpack_2_pixels( b >> 4))
            bottom.extend( unpack_2_pixels( b))

        return [ top, bottom ]

    elif LINES_PER_BYTE == 2:
        top = []
        bottom = []
        for byte in data:
            t,b = unpack_2_pixels(byte)
            top.append(t)
            bottom.append(b)
        return [ top, bottom ]

    else:
        return [ [ x for x in data] ]

def decrunch_2bytes( data):

    assert len(data) == BYTES_PER_FRAME, "Expected {}, got {}".format(BYTES_PER_FRAME, len(data))

    if type(data) == str:
        data = [int(x) for x in data]

    out = []
    for l in range(0,BYTES_PER_FRAME,BYTES_PER_ROW):
        decrunched = split_bytes( data[ l : l+BYTES_PER_ROW])
        assert len(decrunched) == LINES_PER_BYTE
        for pixel_line in decrunched:
            assert len( pixel_line) == WIDTH, "Expected {}, got {}. Bytes per line = {}".format( WIDTH, len( pixel_line), BYTES_PER_ROW)
            out.append( pixel_line )

    assert len(out) == HEIGHT, "Got {}, but BYTES_PER_FRAME={} and BYTES_PER_ROW={}".format( len(out), BYTES_PER_FRAME, BYTES_PER_ROW)
    return out


def array_to_asm( fo, a, line_prefix, label = ""):

    if type(a[0]) == str:
        fmt = "{}"
    elif line_prefix == '!word':
        fmt = "${:04x}"
    elif line_prefix == '!byte':
        fmt = "${:02x}"
    else:
        raise Exception("Unknown format {}".format( line_prefix))

    if label:
        label = "\t{}:".format(label)
    else:
        label = ""

    fo.write("{}; {} values\n".format(label, len(a)))
    for i in range( 0, len( a), 10):
        end = min( i + 10, len( a))
        fo.write("\t{} {}\n".format( line_prefix, ", ".join( [ fmt.format(x) for x in a[i:end]])))


def hex_word( b_or_list, prefix='!dword '):
        return prefix + ",".join( ["${:04x}".format(b).upper() for b in b_or_list] )


def hex_byte(b_or_list, prefix="!byte "):

    if type(b_or_list) in (list, bytes):
        return prefix + ",".join( ['$'+format(b,'02X') for b in b_or_list] )
    else:
        return '$'+format(b_or_list,'02X')

def compress_file( bin_name, bdata):
    bfile = open( bin_name,'bw')
    bfile.write( bdata)
    bfile.close()
    os.system('lz4x.exe -9 -f {} {}'.format(bin_name, bin_name+'lz4'))
    return os.path.getsize( bin_name+'lz4')

def stripe_as_str( stripe):
    return ",".join( [format(b,'02d') for b in stripe] )

def decode_strip_str( s):
    return [int(x) for x in s.split(",")]


def peek( data, i, scan_value, strip_max_len):
    cnt = 0
    while i+cnt < len(data) and data[i+cnt] == scan_value and (cnt < strip_max_len):
        cnt += 1

    return cnt

def pack_line( data, i, scan_values, max_len = None):

    if not max_len:
        max_len = MAX_STRIPE_LENGTH
    cnt = 0
    stripe = []
    while i < len(data) and (data[i] in scan_values)  and (cnt < max_len): # and (data[i] in scan_values)
        stripe.append(data[i])
        i += 1
        cnt += 1

    return stripe, i

def pack_line_one_pixel_stop( data, i, scan_values, stop_values, max_i):
    """ Somehow, picking an additional, different, pixel after a long run
    is really efficient (ie a 8% increase in compression).

    Picking 1 more is 10% more efficient.
    Picking 2 more is 30% less efficient.
    Picking n more is totally not efficient (like 100% less efficient)
    """
    cnt = 0
    stripe = []
    while i < len(data) and data[i] == scan_values and (cnt < strip_max_len) and i < max_i: # and (data[i] in scan_values)
        stripe.append(data[i])
        i += 1
        cnt += 1


    stop_value_cnt = 0
    while i < len(data) and (cnt < strip_max_len) and i < max_i: # and (data[i] in scan_values)
        stripe.append(data[i])
        i += 1
        cnt += 1
        stop_value_cnt += 1
        if stop_value_cnt == 1:
            break

    return stripe, i


def super_pack_line( data, i, scan_value):

    shorter_len = MAX_STRIPE_LENGTH // 2
    shorter_len = 4
    bigger_len = MAX_STRIPE_LENGTH*4
    if bigger_len > 31:
        bigger_len = 31

    # 31 : gives a few bytes better compression than 32.

    assert shorter_len < bigger_len

    cnt = peek( data, i, scan_value, bigger_len)
    #print(cnt)

    if cnt > shorter_len:
        # Simple tile repetition
        stripe, i = pack_line( data, i, [scan_value], bigger_len)
    else:
        others = set( range(256))
        others.remove( scan_value)
        stripe, i = pack_line_one_pixel_stop( data, i, scan_value, others, i+shorter_len )

    #print("{} {}".format( scan_value, len(stripe)))

    return stripe, i






def pack_line_one_pixel_stop2( data, i, scan_value, stop_values, max_i):
    cnt = 0
    stripe = []
    while i < len(data) and data[i] == scan_value and (cnt < strip_max_len) and i < max_i: # and (data[i] in scan_values)
        stripe.append(data[i])
        i += 1
        cnt += 1


    stop_value_cnt = 0
    while i < len(data) and (cnt < strip_max_len) and i < max_i: # and (data[i] in scan_values)
        stripe.append(data[i])
        i += 1
        cnt += 1
        stop_value_cnt += 1
        if stop_value_cnt == 1:
            break

    return stripe, i


def MSE(data1, data2):
    l = len(data1)

    assert l == len(data2)
    error = 0
    for i in range( l):
        error += (data1[i] - data2[i])**2
    error = error / len(data1)
    if error < (1/3)*(1/3) / l:
        if error > 0:
            print(data1)
            print(data2)
        error = 0
    return error


def compare_stripe( a,b):
    if compare_row(a,b) == False:
        return 1234

    #clip = [0,0,0,1]
    clip = [0/3,1/3,2/3,3/3]

    a = split_bytes(a)
    b = split_bytes(b)

    wa = []
    wb = []

    for l in range(len(a)):
        wa.extend(  [ clip[x] for x in a[l]])
        wb.extend(  [ clip[x] for x in b[l]])

    mse = MSE(wa, wb)
    assert 0 <= mse <= 1
    # print("MSE")
    # print(wa)
    # print(wb)
    # print(mse)

    # 0000111110000
    # 0010111100001
    # __1_____1___1

    return mse




def simplify_tiles( data):
    stats = sorted( collections.Counter( data).items(), key=lambda p:p[1], reverse=True)
    replacements = [0 for i in range(256)]

    ndx = 0
    for k,cnt in list(reversed(stats)):
        pixels = split_bytes( [ k ] )
        ndx += 1

        grays = []
        whites = blacks = 0
        for xx in range(len( pixels[0])):
            for yy in range(len( pixels)):
                if pixels[yy][xx] == 0:
                    blacks += 1
                elif pixels[yy][xx] == 3:
                    whites += 1
                elif pixels[yy][xx] in (1,2):
                    grays.append(pixels[yy][xx])

        if cnt <= 20:
            replacements[k] = pack_4_pixels([ [0,0], [0,0] ])

        elif blacks >= 3 and whites == 0:
            pixels = [ [0,0], [0,0] ]
            b = pack_4_pixels(pixels)
            replacements[k] = b

        elif whites >= 3 and blacks == 0:
            pixels = [ [3,3], [3,3] ]
            b = pack_4_pixels(pixels)
            replacements[k] = b

        elif grays:
            avg = int(round(sum(grays) / len(grays)))
            for xx in range(len( pixels[0])):
                for yy in range(len( pixels)):
                    if pixels[yy][xx] in (1,2):
                        pixels[yy][xx] = avg

            b = pack_4_pixels(pixels)
            replacements[k] = b

    # print( len( stats))
    # print( len(list(filter( lambda k:k[1] >= 10, stats))))
    # print( len( set( replacements)))
    # print(stats)
    # print(replacements)

    tile_map = bidict() # Maps tile numbers 0,1,2,3,4,... to actual tiles 0,12,34,..,255
    for i in range( len(replacements)):
        if replacements[i] not in tile_map.inv:
            tile_map[ len(tile_map) ] =  replacements[i]
        replacements[i] = tile_map.inv[ replacements[i]]
    return replacements, tile_map



def print_tiles_stats( y, stats):
    x=1
    for k,cnt in stats:
        pixels = split_bytes( [ k ] )

        #print(len( pixels[0]))
        #print(len( pixels))

        for xx in range(len( pixels[0])):
            stdscr.addstr(y*3 + 3 - 1 , 5, "----+"*32)
            for yy in range(len( pixels)):
                stdscr.addstr(y*3 + yy, x*5 + xx*2, colors[pixels[yy][xx]])
                stdscr.addstr(y*3 + yy, x*5 - 1, "|")

        stdscr.addstr(y*3-1, x*5, "{:4}".format(min(9999, cnt)))

        x += 1
        if x > 32:
            x = 1
            y += 1

    while curses_loop():
        pass




def make_delta_frames_stream( frames, special_tiles, bytes_per_frame):

    assert len( frames) % bytes_per_frame == 0

    delta_frames_stream = []
    delta_frames_stream.extend( frames[0:bytes_per_frame] )

    for i in range(bytes_per_frame, len( frames), bytes_per_frame):
        old_f = frames[i-bytes_per_frame:i]
        f = frames[i:i+bytes_per_frame]

        delta_frame = []
        for j in range( len( f)):
            if f[j] == old_f[j]:
                delta_frame.append( special_tiles.transparent)
            else:
                delta_frame.append( f[j])

        delta_frames_stream.extend( delta_frame)

    return delta_frames_stream



class Stripe:
    def __init__( self, data, special_tiles):
        assert len(data) <= MAX_STRIPE_LENGTH * 4, "Bad data {}".format(data)
        self.data = data
        self._hash = hash(array('B',data).tobytes())
        self.cycles = None # Number of cycles needed to decompress the stripe

        self.stripe_id = None
        self.compressed = self._compress_stripe2( self.data, special_tiles)
        self.stripe_id2 = None

        decomp = self._decompress_stripe( self.compressed, special_tiles)
        assert data == decomp, "{} != {}, compressed={}".format( data, decomp, hex_byte(self.compressed))

        #self.compressed = self._compress_stripe2( self.data, transparent_tile)
        self.label = None
        self.frequency = 0


    def __str__(self):
        return "Stripe freq:{}x {} [hash:{}]".format( self.frequency, ",".join( ['$'+format(b,'02X') for b in self.data] ) ,self._hash)

    def __hash__(self):
        return self._hash

    def has_no_count(self):
        return self.compressed[0] & 128 == 128

    def _decompress_stripe( self, data, special_tiles):

        self.cycles = 80

        r = []
        if data[0] & 128 == 128:

            self.cycles += len(data) * (63+26)

            #print("decompress raw bytes : {}".format(data))
            r.append( data[0] & 127 )

            if data[1] == 255:
                return r

            i = 1
            while True:
                r.append( data[i])
                if data[i] & 128:
                    r[-1] = r[-1] & 127
                    return r
                i += 1

            return r
        else:
            self.cycles += 111

            #print("decompress byte run")
            cmd = data[0] >> 5
            cnt = (data[0] & 31) + 1

            if cmd == 0:
                color = special_tiles.white
                self.cycles += cnt * 19
            elif cmd == 1:
                color = special_tiles.black
                self.cycles += cnt * 19
            elif cmd == 2:
                color = special_tiles.transparent
                self.cycles += cnt * 19

            r = []
            r.extend( [color] * cnt)
            r.append( data[1])
            return r

    def _compress_stripe2( self, values, special_tiles):
        # Some calculations :
        # There are 2200 frames
        # There are 3760 different stripes.
        # If want to know how many stripes there are in a frame at run time, I can :
        # 1/ Have a map frame -> nb stripes; which optimistically is 2200 bytes
        # 2/ I can count the length of the stripes until a I reach a frame, but this is colstly => I need to be able to tell the size
        #    of a stripe => I add a length byte => 3288 stripes need one => cost 3288 bytes
        # 3/ I can add a special strip to mark the end of a frame, but if Huffman compression
        #    doesn't work well enough (less than 8 bits for that stripe), it might be bigger than 2200.

        if len(values) <= 2 or values[0] not in special_tiles.all():
            r = [v for v in values ]
            r[0] = r[0] | 128

            # There are two ways of marking the end of a stream of tiles (see below).
            # Optimizing it this way let me spare +/- 5 kb out of 29 kb in the
            # stripe dictionary.

            if len(values) == 1:
                r.append(255)
            else:
                r[-1] = r[-1] | 128 # BAsically : data byte | MARK, data_byte, ..., data_byte | MARK => I spare a counter byte.
            return r
        else:
            # We always encode : a repetition of one tile followed by a single tile.
            # I cannot use the topmost bit because it's used for stream of stripes
            # (cf above)
            if values[0] == special_tiles.white:
                cmd = 0b000 << 5
            elif values[0] == special_tiles.black:
                cmd = 0b001 << 5
            elif values[0] == special_tiles.transparent:
                cmd = 0b010 << 5

            assert cmd & 128 == 0
            # Run over black or white bytes
            i = 0
            while i+1 < len(values) and values[i+1] == values[0]:
                i += 1
            assert len(values) > 2, "shorter run should be compressed differently"
            assert values[i] == values[0], "this is not a run of bytes that are all the same"
            assert i in (len(values)-1, len(values)-2), "{} not in {}".format( i, (len(values)-2, len(values)-1))

            # -1 because the last tile is put apart, -1 because the cnt of repetition is never 0 (so we save 1 increment)
            repeat_byte = cmd | (len(values) - 1 - 1)
            additional_tile_byte = values[ len(values) - 1]
            return [ repeat_byte, additional_tile_byte]


def make_stripes(data_stream, special_tiles, bytes_per_frame=512):

    assert len(data_stream) % bytes_per_frame == 0

    all_stripes_codes = []
    others = set( range(256))
    others.remove( special_tiles.white)
    others.remove( special_tiles.black)
    others.remove( special_tiles.transparent)

    for ndx in range( 0, len(data_stream), bytes_per_frame):
        #print(ndx)
        data = data_stream[ ndx:ndx+bytes_per_frame]
        i = 0
        while i < len(data):
            if data[i] == special_tiles.transparent:

                stripe, i = super_pack_line( data, i, special_tiles.transparent)
                #stripe, i = pack_line( data, i, [transparent_tile])

            elif data[i] == special_tiles.white:
                #stripe, i = pack_line( data, i, OTHERS + [WHITE], BLACK)
                #stripe, i = pack_line( data, i, WHITE, [])
                #stripe, i = pack_line_one_pixel_stop( data, i, WHITE, OTHERS, i+MAX_STRIPE_LENGTH )
                stripe, i = super_pack_line( data, i, special_tiles.white)

            elif data[i] == special_tiles.black:
                #stripe, i = pack_line( data, i, OTHERS + [BLACK], WHITE)
                #stripe, i = pack_line( data, i, BLACK, [])
                #stripe, i = pack_line_one_pixel_stop( data, i, BLACK, OTHERS, i+MAX_STRIPE_LENGTH)
                stripe, i = super_pack_line( data, i, special_tiles.black)

            else:
                #stripe, i = pack_line( data, i, OTHERS, [BLACK[0], WHITE[0]])
                stripe, i = pack_line( data, i, others, 4)


            all_stripes_codes.append( Stripe(stripe, special_tiles))

    return all_stripes_codes

# 95 = 01 01 11 11
# pixels = [1,2]*20 + [3,0]*20 # 2 lines of 40 pixels each
# print(pixels)
# b = pack_bytes( pixels)
# print(format(b[0],'08b'))

# print(b)
# assert( len(b) == 20)
# print( split_bytes( b))
#exit()

TILE_SIZE = 4

#WHITE=[3 << 2 + 3] # lucky bug


if LINES_PER_BYTE*PIX_PER_BYTE == 4:
    WHITE=255
    BLACK=(0 << 0) + 0
elif LINES_PER_BYTE*PIX_PER_BYTE == 2:
    WHITE=[(3 << 2) + 3]
    BLACK=[(0 << 0) + 0]
else:
    WHITE=[3]
    BLACK=[0]

OTHERS=list( range( BLACK + 1, WHITE))


ffmpeg()


white_stripes = dict()
black_stripes = dict()
gray_stripes = dict()

gs_data= dict()

all_original_frames = []
all_decimated_frames = []

all_stripes_codes = []

strip_max_len = BYTES_PER_ROW

img_ndx = 0
decimated_frames = set()

ERROR_THRESHOLD=0

min_error = 10000
decimated_rows = set()

row_ndx = 0
old_frame_data = None
old_packed_data = None

data_size_after_decimation = 0

packed_stream = []

APPLE_LO = [ i - 1024 for i in [1024,1152,1280,1408,1536,1664,1792,1920,
	                        1064,1192,1320,1448,1576,1704,1832,1960,
	                        1104,1232,1360,1488,1616,1744,1872,2000]]

def appleize( data, w, h):
    assert len(data) == w*h

    """ A block of 128 bytes stores three rows of 40 characters each, with
    a remainder of eight bytes left after the third row is stored.

    00 03 06 pp 09 12 15 pp ...
    01 04 07 pp 10 13 16 pp ...
    02 05 08 pp 11 14 17 pp ...

    https://retrocomputing.stackexchange.com/questions/2534/what-are-the-screen-holes-in-apple-ii-graphics


          TOP/         MIDDLE/      BOTTOM/      (SCREEN HOLES)
    BASE  FIRST 40     SECOND 40    THIRD 40     UNUSED 8
    ADDR  #  RANGE     #  RANGE     #  RANGE     RANGE
    $400  00 $400-427  08 $428-44F  16 $450-477  $478-47F
    $480  01 $480-4A7  09 $4A8-4CF  17 $4D0-4F7  $4F8-4FF
    $500  02 $500-527  10 $528-54F  18 $550-577  $578-57F
    $580  03 $580-5A7  11 $5A8-5CF  19 $5D0-5F7  $5F8-5FF
    $600  04 $600-627  12 $628-64F  20 $650-677  $678-67F
    $680  05 $680-6A7  13 $6A8-6CF  21 $6D0-6F7  $6F8-6FF
    $700  06 $700-727  14 $728-74F  22 $750-777  $778-77F
    $780  07 $780-7A7  15 $7A8-7CF  23 $7D0-7F7  $7F8-7FF

    """
    apple_mem = [ 0]*1024

    for y in range( 0,h*w,w*2):
        b = [ data[ y + x] + (data[ y + w + x] << 4) for x in range(w) ]

        p = APPLE_LO[ y // (w*2)]
        apple_mem[ p : p+len(b)] = b

        #print( data[y:y+2*w])
        # print("{} : {}".format(p, b))

    return apple_mem


def deappleize( apple_mem, w, h):

    data = []

    for y in range( 0, h, 2):
        p = APPLE_LO[ y // 2]

        l1, l2 = [], []
        for x in range(w):
            b = apple_mem[p + x]
            l1.append( b & 15)
            l2.append( b >> 4)

        data.extend(l1)
        data.extend(l2)

        # print("{}: {} {}".format(y, l1, l2))


    assert len(data) == w*h
    return data


def apple_2_pack4( a_data):
    packed_data = []
    for i in range( 0, len(a_data), 2):

        p1 = a_data[i] & 15
        p2 = a_data[i] >> 4
        p3 = a_data[i+1] & 15
        p4 = a_data[i+1] >> 4

        # On screen :
        #     p1 p3
        #     p2 p4
        # In apple memory
        #     (p2,p1) (p4,p3)
        # In packed memory
        #     (p1,p2,p3,p4)

        packed_data.append( (p1 << 6) + (p3 << 4) + (p2 << 2) + p4 )
    return packed_data


def pack4_2_apple( data):
    unpacked_data = []
    for b in data:
        p1 = (b >> 6) & 3
        p3 = (b >> 4) & 3
        p2 = (b >> 2) & 3
        p4 = b & 3

        # p1 p3
        # p2 p4
        unpacked_data.append( (p2 << 4) + p1 )
        unpacked_data.append( (p4 << 4) + p3 )
    return unpacked_data



def decompress_one_frame( all_stripes, stripe_ndx):
    decompressed = []
    used_stripes = []
    while len( decompressed) != 512:
        stripe = all_stripes[ stripe_ndx]
        used_stripes.append( stripe)
        decompressed.extend( stripe.data)
        stripe_ndx += 1

    assert len(decompressed) == 512, "Unepxected length = {}".format(len(decompressed))

    return decompressed, used_stripes, stripe_ndx


if os.name == 'nt':
    IMG_PREFIX = 'c:/PORT-STC/PRIVATE/tmp'
    FFMPEG = 'c:\PORT-STC\opt\ffmpeg-20181221-53d3a1c-win64-static\bin\ffmpeg -i bad_apple_original.mp4 -an  -vf fps=10,scale=36:46  c:\port-stc\private\tmp\bad_apple%05d.png'
else:
    IMG_PREFIX = '/tmp'

for filename in sorted(glob.glob('{}/bad_apple0*.png'.format(IMG_PREFIX))):

    if img_ndx < SKIP_PICTURES:
        img_ndx += 1
        continue

    if img_ndx % 100 == 0:
        print("{} ".format(img_ndx, end='\r', flush=True))

    if img_ndx == MAX_PICTURES: # 139125
        break

    img = Image.open(filename)
    data = img.convert('L').tobytes()
    #data = [ int(3*round(x/256)) for x in data] # From grayscale to 1 bits per pixel
    data = [ int(round(3.499 * x /256)) for x in data] # From grayscale to 2 bits per pixel


    # Testing appleize functions
    # a_data = deappleize( appleize( data, WIDTH, HEIGHT), WIDTH, HEIGHT)
    # assert data == a_data, "{} {}".format(data, a_data)
    # TEsting pack4
    # assert a_data == pack4_2_apple( apple_2_pack4( a_data)), "{}\n{}".format( pack4_2_apple( apple_2_pack4( a_data)), a_data)

    packed_data = apple_2_pack4( appleize( data, WIDTH, HEIGHT))

    # packed_data = []

    # for l in range( 0, WIDTH*HEIGHT, PIXELS_PER_ROW):
    #     pb = pack_bytes( data[l:l+PIXELS_PER_ROW])
    #     assert len(pb) == BYTES_PER_ROW, "Got {}".format( str(pb))
    #     packed_data.extend( pb)

    # #packed_data = appleize( packed_data)
    # assert len(packed_data) == BYTES_PER_FRAME, "Got {}, expected {}".format( len(packed_data), BYTES_PER_FRAME)

    all_original_frames.append(packed_data)
    packed_stream.extend( packed_data)

    img_ndx += 1

assert img_ndx >= 1, "No images found ?"

replacements, tile_map = simplify_tiles( packed_stream)
for n in range(len(packed_stream)):
    packed_stream[n] = replacements[ packed_stream[n]]
    assert packed_stream[n] < 128

# print( tile_map)

#stats = sorted( collections.Counter( packed_stream).items(), key=lambda p:p[1], reverse=True)

# print("BLACK={}, WHITE={}, tranapernt={}".format( BLACK, WHITE, transparent_tile))

class SpecialTiles:
    def __init__( self, tile_map):
        self.black = tile_map.inv[0]
        self.white = tile_map.inv[255]
        self.transparent = 0x7E

        self.others = [x for x in range(self.transparent)]
        self.others.remove( self.white)
        self.others.remove( self.black)


    def all(self):
        return (self.black, self.white, self.transparent)


special_tiles = SpecialTiles( tile_map)

# stdscr = curses.initscr()
# print_tiles_stats(1, stats)


# for ndx in [0x78, 0xF8, 0x178, 0x1F8, 0x278, 0x2F8, 0x378, 0x3F8]:
#     data[ (ndx //2) : (ndx // 2) + 4] = transparent_tile


delta_frames = make_delta_frames_stream( packed_stream, special_tiles, 1024//2)
all_stripes = make_stripes(delta_frames, special_tiles, 512)

print("{} delta frames".format( len(delta_frames) // 512))

unique_stripes = dict()
stripe_id = 1
for s in all_stripes:
    h = hash(s)
    if h not in unique_stripes:
        unique_stripes[ h] = s
        s.stripe_id = stripe_id
        stripe_id += 1

for i in range( len( all_stripes)):
    all_stripes[i] = unique_stripes[ hash(all_stripes[i])]


ndx = 0
for s, freq in collections.Counter( all_stripes ).items():
    s.frequency = freq
    s.label = ndx
    ndx += 1


f1 = [s for s in filter( lambda s:s.frequency == 1, sorted( unique_stripes.values(), key=lambda s:s.frequency)) ]

print( "{} length-1 stripes, totalling {} bytes".format( len(f1), sum( [ len(s.data) for s in f1 ])))
#exit()

# cum = 0
# with open('freq.gplot', 'w') as fo:
#     for s in sorted( unique_stripes.values(), key=lambda s:s.frequency):
#         print("#{}\t{}x\t{}".format( s.label, s.frequency, len(s.data)))
#         cum += s.frequency
#         fo.write("{}\n".format( cum))


def simple_lz77( all_stripes):

    stream = []
    dico = {} # Maps stripe -> last position in stream

    for s in all_stripes:
        if s not in dico:
            dico[s] = len(stream)
            stream.append(s.label)
        else:

            # I should limit the number of jumps (as well as the distance of jumps)
            if len(stream) - dico[s]  > 127:
                dico[s] = len(stream) - 1
                stream.append(s.label)
            else:
                stream.append( dico[s] - len(stream) )

    s = 0
    for x in stream:
        if x >= 0:
            s += 2
        elif x > -128:
            s += 1
        else:
            raise Exception("argh")

    # print(stream[1000:10000])
    print("LZ77 Stream size : {}".format( s))



# 155296 : without d1, d2...
# 153278 : with d1, d2...

def simple_huffman( unique_stripes, all_stripes):
    sid = 1

    # Sort stripes, most frequent first

    for s in sorted( unique_stripes.values(), key=lambda s:s.frequency, reverse=True):
        s.stripe_id2 = sid
        sid += 1

    # for s in all_stripes[0:100]:
    #     print("({},{})".format( s.stripe_id, s.stripe_id2 ))

    stream = bitstring.BitArray()

    d1 = (2 ** 3)
    d2 = (2 ** 6) + d1
    d3 = (2 ** 9) + d2

    ndx = 0
    for s in all_stripes:
        sid = s.stripe_id2 - 1

        if sid < d1:
            # 0xxxb
            bits = bitstring.BitArray(length=4, uint=sid)
        elif d1 <= sid < d2:
            # 10yy yyyy
            bits = bitstring.BitArray(length=8, uint=0b10000000 + sid - d1)
        elif d2 <= sid < d3:
            # 110z zzzz zzzz 12 bits, 9 significant
            bits = bitstring.BitArray(length=12, uint=0b110000000000 + sid - d2)
        elif d3 <= sid < 2 ** 13:
            # 111z zzzz zzzz zzzz 16 bits, 13 significant
            bits = bitstring.BitArray(length=16, uint=0b1110000000000000 + sid - d3)
        else:
            raise Exception("argh {}".format( sid))

        if ndx < 300:
            print("s# {} (b: {}) -> {} / {}".format( hex(ndx), len(stream.tobytes()), hex(bits.uint), sid))

        stream.append( bits)
        ndx += 1

    print("Bit stream simple huffman : {} stripes, {} bits, {} bytes".format( len( all_stripes), len( stream), len(stream) // 8))

    b = stream.tobytes()

    extra_bytes = 3


    too_much = len(b) - MAX_DISK_SIZE

    with open("compressed.a","w") as fo:
        array_to_asm( fo, b[0:too_much + extra_bytes], '!byte')

    with open("cstripes.dsk","bw") as fo:
        fo.write( disk_2_dos( b[too_much:]))


    # print("Some stripes:")
    # for i in range(20):
    #     print( '{:04} '.format(i*16) + ' '.join([ "${:04x}".format(s.stripe_id2) for s in all_stripes[i*16:(i+1)*16]]))



    return

    # Test decompression

    #print( hex_word([s.stripe_id2 for s in all_stripes[0:500]]))
    #print( hex_byte( stream.tobytes()[0:1000]))

    decomp_stream = []
    max_l = len( stream)
    ndx = 0
    while ndx < max_l:
        half_byte = stream[ndx:ndx+4].uint

        if   half_byte & 0b1000 == 0:
            s = half_byte

        elif half_byte & 0b1100 == 0b1000:
            s = (half_byte & 0b0011)
            ndx += 4
            s = (s << 4) + stream[ndx:ndx+4].uint
            s += d1

        elif half_byte & 0b1110 == 0b1100:
            s = (half_byte & 0b0001)
            #print( hex(s))
            ndx += 4
            s = (s << 4) + stream[ndx:ndx+4].uint
            #print( hex(s))
            ndx += 4
            s = (s << 4) + stream[ndx:ndx+4].uint
            #print( hex(s))
            s += d2
            #print( hex(d2))
            #print( hex(s))

        elif half_byte & 0b1110 == 0b1110:
            s = (half_byte & 0b0001)
            ndx += 4
            s = (s << 4) + stream[ndx:ndx+4].uint
            ndx += 4
            s = (s << 4) + stream[ndx:ndx+4].uint
            ndx += 4
            s = (s << 4) + stream[ndx:ndx+4].uint
            s += d3

        decomp_stream.append(s)
        ndx += 4


    a = [s.stripe_id2 for s in all_stripes]
    b = decomp_stream

    for i in range( len(a)):
        if a[i] != b[i]:
            print(i)


simple_lz77( all_stripes)
simple_huffman( unique_stripes, all_stripes)

all_stripes_labels = dict()
for i in range( len( all_stripes)):
    all_stripes_labels[ all_stripes[i].label] = all_stripes[i]


hcb = huffman.codebook(collections.Counter( all_stripes ).items())

mini,maxi = 9999999,0
# "1"+makes the front zero significant
for k,v in sorted( hcb.items(), key=lambda k_v:int("1"+k_v[1],2)):
    mini = min(len(v), mini)
    maxi = max(len(v), maxi)
    if len(v) < 800:
        pass
        #print( "{}\t -> {}".format( v, str(k)))


print("{} different tiles; transparent tile={}; {} labels".format( len( set( replacements)), hex_byte(special_tiles.transparent), len( all_stripes_labels)))
print("min len:{}, max len:{}".format(mini, maxi))

s = ""
for l in all_stripes:
    s += hcb[ l]

huffman_compressed_size = len(bitstring.BitArray(bin=s).tobytes())
print("Movie size after Huffman compression = {} / {} bytes".format(huffman_compressed_size, MAX_DISK_SIZE))


compressed = bytearray()
for l in all_stripes:
    compressed.append( l.label & 255)
    compressed.append( l.label >> 8)

print("Movie size with just stripes (no compress) : {} bytes".format( len( compressed)))

assert len(delta_frames) % 512 == 0, "Delta frames  {}, bytes per frame = {}".format(len(delta_frames), 512)


print("Testing LZ4 compression wiwth official lz4 compressor")
with open("test","bw") as fo:
    fo.write( compressed)
os.system("lz4\\lz4.exe -f -9 test test.lz4")


try:
    import lz4.frame
    print("Movie size after LZ4 compression : {}".format( len(lz4.frame.compress( bytes( compressed), compression_level=lz4.frame.COMPRESSIONLEVEL_MAX))))
except Exception as ex:
    print("No LZ4 in python")






with open("data.a","w") as fo:

    # stripe_ndx = 0
    # frames_cycles = []
    # while stripe_ndx < len( all_stripes):
    #     decompressed, used_stripes, stripe_ndx = decompress_one_frame( all_stripes, stripe_ndx)
    #     frames_cycles.append( sum( [s.cycles for s in used_stripes]))

    # fo.write('frames_cycles:\n')
    # array_to_asm( fo, frames_cycles, "!word")

    palette = [0,5,10,15] # grays
    #palette = [0,2,6,15] # blues

    fo.write("tiles:\n")
    i = 0
    for tk in sorted(tile_map.keys()):
        t = tile_map[tk]
        p1, p2 = pack4_2_apple( [t])
        # 0,5,10,15

        def zeze( p):
            a = p & 15
            b = p >> 4
            assert 0 <= a <= 3 and 0 <= b <= 3
            #return ((b * 5) << 4) + (a * 5)
            return ( palette[b] << 4) + palette[a]

        fo.write("\t{}\t; {} - {}\n".format(  hex_byte( [ zeze(p1), zeze(p2)]), hex(i), t))
        i += 1

    def stripe_id(stripe):
        return stripe.stripe_id2

    sorted_stripes = sorted( unique_stripes.values(), key=stripe_id)
    fo.write('\n')
    for s in sorted_stripes:
        fo.write("stripe{}\t{}\t; {}\n".format( stripe_id(s), hex_byte(s.compressed), hex_byte(s.data, '')))

    # fo.write("stripes_pointers\t!word {}\n".format( ",".join( ["stripe{}".format(i+1) for i in range(len(unique_stripes))])))

    fo.write('\n')

    fo.write('stripes_indices:\n')
    array_to_asm( fo, ["stripe{}".format( stripe_id(s)) for s in sorted_stripes], "!word")

    #fo.write('stripes_pointers:\n')
    #array_to_asm( fo, [ (s.stripe_id-1)*2 for s in all_stripes], "!word") # *2 because we'll index a words array
    # Mark the end of stripes
    #fo.write('\t!word $FF,$FF,$FF,$FF\t; End of stripes mark\n')


    # for i in range( 0, len( all_stripes), 10):
    #     end = min( i + 10, len( all_stripes))
    #     fo.write("\t!word {}\n".format( ",".join( ["stripe{}".format(s.stripe_id) for s in all_stripes[i:end]])))


stripes_to_disk( all_stripes)

if 0:
    stdscr = curses.initscr()
    stdscr.nodelay(1)
    curses.cbreak()

    img_ndx = 0
    stripe_ndx = 0
    accu_frame = [0] * 512

    cycles = 0
    sector_cycles = 0
    sector = 0
    sectors_read = [False] * 16
    tracks_read = 0

    for ndx in range( 0, len(delta_frames), 512):
        #oframe = decrunch_2bytes(all_original_frames[img_ndx])
        oframe = deappleize( pack4_2_apple( all_original_frames[img_ndx]), WIDTH, HEIGHT)

        # Decompress a frame.
        # A frame is 1024 bytes. But since we store two bytes in one (with color reduction)
        # this becomes 512 bytes.

        frame_cycles = 0
        done_sector_stuff_this_frame = False

        decompressed, used_stripes, stripe_ndx = decompress_one_frame( all_stripes, stripe_ndx)

        # Sectors skipped while decrunching the stripes
        pass_cycles = frame_cycles
        while pass_cycles > 12500:
            pass_cycles -= 12500
            sector = (sector + 1) % 16

        # We use the time to wait for the frame end to load stuff.

        remaining_cycles = 100000 - frame_cycles

        if remaining_cycles > 12500*2: # * 2 is just a security
            if sectors_read.count(True) < 16:

                max_sect = 0
                while remaining_cycles > 12500 * 2:
                    sectors_read[sector] = True
                    sector_cycles += 12500
                    remaining_cycles -= 12500
                    sector = (sector + 1) % 16

        while remaining_cycles > 12500:
            remaining_cycles -= 12500
            sector = (sector + 1) % 16

        for i in range( len( decompressed)):
            if decompressed[i] != special_tiles.transparent:
                accu_frame[i] = tile_map[decompressed[i]]

        a = deappleize( pack4_2_apple( accu_frame), WIDTH, HEIGHT)

        #a = decrunch_2bytes( accu_frame)
        #a = decrunch_2bytes( delta_frames[ndx:ndx+BYTES_PER_FRAME])

        show_pictures( a, oframe)
        stdscr.addstr(1,1,"Cycles = {} / sector = {}".format(sum( [s.cycles for s in used_stripes]), sector))
        stdscr.addstr(2,1,"".join( ["#" if x else "."  for x in sectors_read]) + " Tracks : {} / {:.1f} frame per track".format(tracks_read, img_ndx / (tracks_read or 1)))

        if sectors_read.count(True) == len( sectors_read):
            sectors_read = [False] * 16
            tracks_read += 1

        img_ndx += 1
        if not curses_loop():
            break

    if ndx >= len(delta_frames) - 512:
        while curses_loop():
            pass

    curses.endwin()

print("Replacement tiles: We keep {} out of 256".format( len( set( replacements))))
print("{} stripes, {} unique totalling {} bytes (incl 2 bytes for index table), covering {} bytes".format(
    len( all_stripes),
    len( unique_stripes),
    sum( [ (2+len(s.compressed)) for s in unique_stripes.values()]),
    len( packed_stream)))

# for s in all_stripes:
#     print(s)



lengths = dict( [ (i,0) for i in range(256)])
for s in all_stripes:
    lengths[ len( s.data)] += 1
for i in range(256):
    if lengths[i] == 0:
        del lengths[i]

print(lengths)

no_count = 0
for s in unique_stripes.values():
    if s.has_no_count():
        no_count += 1

print("{} stripes out of {} have no count".format( no_count, len( unique_stripes)))
