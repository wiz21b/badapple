import bitstring
import collections
import math
from array import array
import os

""" HGR   = 280 * 192
    C64   = 40*24 chars => 320 * 192 ; display 3min 20 (2000 images) sec instead of 3min 40 (2200)
    Video = 192 * 160 (24*20)
"""

if os.name == 'nt':
    IMG_PREFIX = r'c:/PORT-STC/PRIVATE/tmp'
    FFMPEG = r'c:\PORT-STC\opt\ffmpeg-20181221-53d3a1c-win64-static\bin\ffmpeg' # -i bad_apple_original.mp4 -an  -vf fps=10,scale=36:46  c:\port-stc\private\tmp\bad_apple%05d.png'
else:
    IMG_PREFIX = '/tmp'
    FFMPEG = 'ffmpeg'

DISK_SIZE = 143360

class SpecialTiles:
    def __init__( self, black, white, transparent):
        self.black = black
        self.white = white
        self.transparent = transparent


        # self.others = [x for x in range(self.transparent)]
        # self.others.remove( self.white)
        # self.others.remove( self.black)


    def all(self):
        return (self.black, self.white, self.transparent)


class Stripe:
    def __init__( self, data, special_tiles):
        self.data = data
        self._hash = hash(array('B',data).tobytes())
        self.cycles = None # Number of cycles needed to decompress the stripe

        self.stripe_id = None
        self.compressed = self._compress_stripe2( self.data, special_tiles)
        self.stripe_id2 = None

        decomp = self._decompress_stripe( self.compressed, special_tiles)
        assert data == decomp, "{} != {}, compressed={}".format( hex_byte(data), decomp, hex_byte(self.compressed))

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
                cmd = 0b000 << 5 # 0x00 = 0
            elif values[0] == special_tiles.black:
                cmd = 0b001 << 5 # 100000 = 0x20
            elif values[0] == special_tiles.transparent:
                cmd = 0b010 << 5 # 1000000 = 0x40

            assert cmd & 128 == 0
            # Run over black or white bytes
            i = 0
            while i+1 < len(values) and values[i+1] == values[0]:
                i += 1
            assert len(values) > 2, "shorter run should be compressed differently"
            assert values[i] == values[0], "this is not a run of bytes that are all the same"
            assert i in (len(values)-1, len(values)-2), "{} not in {}".format( i, (len(values)-2, len(values)-1))
            assert len(values) - 1 - 1 < 2 ** 5

            # -1 because the last tile is put apart, -1 because the cnt of repetition is never 0 (so we save 1 increment)
            repeat_byte = cmd | (len(values) - 1 - 1)
            additional_tile_byte = values[ len(values) - 1]
            return [ repeat_byte, additional_tile_byte]


def ffmpeg( params):
    print(params)
    os.system("{} {}".format( FFMPEG, params))



def make_delta_frames_stream( frames, special_tiles, bytes_per_frame):

    assert len( frames) % bytes_per_frame == 0

    stats_change = []

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

        # Compute some stats

        unchanged = 0
        for i in delta_frame:
            if i == special_tiles.transparent:
                unchanged += 1

        stats_change.append(100.0 * unchanged/len(delta_frame))
        delta_frames_stream.extend( delta_frame)

    avg = sum(stats_change)/len(stats_change)
    stddev = sum( [ math.fabs( i - avg) for i in stats_change ])/len(stats_change)
    print( "unchanged avg:{}, stddev:{}".format( avg, stddev))

    return delta_frames_stream




def peek( data, i, scan_value, strip_max_len):
    cnt = 0
    while i+cnt < len(data) and data[i+cnt] == scan_value and (cnt < strip_max_len):
        cnt += 1

    return cnt

def pack_line( data, i, predicate, max_len):

    cnt = 0
    stripe = []
    while i < len(data) and predicate(data[i])  and (cnt < max_len): # and (data[i] in scan_values)
        stripe.append(data[i])
        i += 1
        cnt += 1

    return stripe, i

def pack_line_one_pixel_stop( data, i, scan_values, stop_values, max_i, strip_max_len):
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


def super_pack_line( data, i, scan_value, max_stripe_length):

    shorter_len = max_stripe_length // 2
    shorter_len = 4
    bigger_len = max_stripe_length*4
    if bigger_len > 31:
        bigger_len = 31

    # 31 : gives a few bytes better compression than 32.

    assert shorter_len < bigger_len

    cnt = peek( data, i, scan_value, bigger_len)
    #print(cnt)

    if cnt > shorter_len:
        # Simple tile repetition
        stripe, i = pack_line( data, i, lambda d:d == scan_value, bigger_len)
    else:
        others = set( range(256))
        others.remove( scan_value)
        stripe, i = pack_line_one_pixel_stop( data, i, scan_value, others, i+shorter_len, max_stripe_length )

    #print("{} {}".format( scan_value, len(stripe)))

    return stripe, i


def make_stripes(data_stream, special_tiles, bytes_per_frame, max_stripe_length):

    assert len(data_stream) % bytes_per_frame == 0

    all_stripes_codes = []

    # others = set( range(256))
    # others.remove( special_tiles.white)
    # others.remove( special_tiles.black)
    # others.remove( special_tiles.transparent)

    for ndx in range( 0, len(data_stream), bytes_per_frame):
        #print(ndx)
        data = data_stream[ ndx:ndx+bytes_per_frame]
        i = 0
        while i < len(data):
            if data[i] == special_tiles.transparent:
                #print("transparent")

                stripe, i = super_pack_line( data, i, special_tiles.transparent, max_stripe_length)
                #stripe, i = pack_line( data, i, [transparent_tile])

            elif data[i] == special_tiles.white:
                #print("white")
                #stripe, i = pack_line( data, i, OTHERS + [WHITE], BLACK)
                #stripe, i = pack_line( data, i, WHITE, [])
                #stripe, i = pack_line_one_pixel_stop( data, i, WHITE, OTHERS, i+MAX_STRIPE_LENGTH )
                stripe, i = super_pack_line( data, i, special_tiles.white, max_stripe_length)

            elif data[i] == special_tiles.black:
                #print("black")
                #stripe, i = pack_line( data, i, OTHERS + [BLACK], WHITE)
                #stripe, i = pack_line( data, i, BLACK, [])
                #stripe, i = pack_line_one_pixel_stop( data, i, BLACK, OTHERS, i+MAX_STRIPE_LENGTH)
                stripe, i = super_pack_line( data, i, special_tiles.black, max_stripe_length)

            else:
                #stripe, i = pack_line( data, i, OTHERS, [BLACK[0], WHITE[0]])
                stripe, i = pack_line( data, i, lambda d : d not in special_tiles.all(), 4)


            all_stripes_codes.append( Stripe(stripe, special_tiles))

    return all_stripes_codes


def simplify_stripes( all_stripes):
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

    return unique_stripes


def compute_stripes_frequencies( all_stripes):
    ndx = 0
    for s, freq in collections.Counter( all_stripes ).items():
        s.frequency = freq
        s.label = ndx
        ndx += 1


def hex_byte(b_or_list, prefix="!byte "):

    if type(b_or_list) in (list, bytes):
        return prefix + ",".join( ['$'+format(b,'02X') for b in b_or_list] )
    else:
        return '$'+format(b_or_list,'02X')


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


def stats_unique_stipes( unique_stripes):
    print("{} unique stripes, stored with {} bytes, representing {} stripes".format(
        len(unique_stripes),
        sum( [ len(s.compressed) for s in unique_stripes]),
        sum( [ s.frequency for s in unique_stripes])))

    f1 = [s for s in filter( lambda s:s.frequency == 1, sorted( unique_stripes, key=lambda s:s.frequency)) ]
    f1_size = sum( [ len(s.data) for s in f1 ])
    print( "{} frequency-1 stripes, totalling {} bytes. Other stripes + index table = {} bytes => total {} bytes".format(
        len(f1), f1_size,
        sum( [ len(s.compressed) for s in unique_stripes]) - f1_size + 2*(len(unique_stripes) - len(f1)),
        f1_size + sum( [ len(s.compressed) for s in unique_stripes]) - f1_size + 2*(len(unique_stripes) - len(f1))))

    f2 = [s for s in filter( lambda s:s.frequency == 2, sorted( unique_stripes, key=lambda s:s.frequency)) ]
    print( "{} frequency-2 stripes, totalling {} bytes".format( len(f2), sum( [ len(s.data) for s in f2 ])))

    with open("stats.csv","w") as fo:
        for s in sorted( unique_stripes, key=lambda s:s.frequency * 100000 + len(s.compressed)):
            fo.write("{};{};{};{};\"{}\"\n".format( s.frequency, len(s.compressed), len(s.data), s.has_no_count(), (hex_byte(s.data))))

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

    d1_count = d2_count = d3_count = d4_count = 0
    d1_len = d2_len = d3_len = d4_len = 0

    ndx = 0
    warn = False
    for s in all_stripes:
        sid = s.stripe_id2 - 1

        if sid < d1:
            # 0xxxb => 8 values
            bits = bitstring.BitArray(length=4, uint=sid)
            d1_count += 1
            d1_len += len(s.compressed)
        elif d1 <= sid < d2:
            # 10yy yyyy => 64 values
            bits = bitstring.BitArray(length=8, uint=0b10000000 + sid - d1)
            d2_count += 1
            d2_len += len(s.compressed)
        elif d2 <= sid < d3:
            # 110z zzzz zzzz 12 bits, 9 significant => 512 values
            bits = bitstring.BitArray(length=12, uint=0b110000000000 + sid - d2)
            d3_count += 1
            d3_len += len(s.compressed)
        elif d3 <= sid < 2 ** 13:
            # 111z zzzz zzzz zzzz 16 bits, 13 significant => 8192 values
            bits = bitstring.BitArray(length=16, uint=0b1110000000000000 + sid - d3)
            d4_count += 1
            d4_len += len(s.compressed)
        else:
            # Error !
            warn = True
            bits = bitstring.BitArray(length=16, uint=0b1111111111111111)

        # if ndx < 300:
        #     print("s# {} (b: {}) -> {} / {}".format( hex(ndx), len(stream.tobytes()), hex(bits.uint), sid))

        stream.append( bits)
        ndx += 1

    if warn:
        print("Too many stripes for the compressor ! (8192) {}".format( len(unique_stripes)))

    print("{} * 4 bits for {} bytes, {} * 8 bits for {} bytes, {} * 12 bits for {} bytes, {} * 16 bits for {} bytes".format(d1_count,d1_len,d2_count,d2_len,d3_count,d3_len,d4_count,d4_len))
    b = stream.tobytes()
    print("Bit stream simple huffman : {} stripes, {} bits, {} bytes".format( len( all_stripes), len( stream), len(b)))


    # Allow some wrapping so that the ASM code is simpler
    extra_bytes = 3


    too_much = len(b) - DISK_SIZE

    MAX = 1024

    if too_much <= 0:
        too_much = MAX

    if too_much > MAX:
        too_much = MAX

    with open("compressed.a","w") as fo:
        array_to_asm( fo, b[0:too_much + extra_bytes], '!byte')

    with open("cstripes.data","bw") as fo:
        fo.write( b)

    with open("cstripes.dsk","bw") as fo:
        fo.write( disk_2_dos( b[too_much:]))


    print("Some stripes:")
    for i in range(20):
        print( '{:04} '.format(i*16) + ' '.join([ "${:04x}".format(s.stripe_id2 - 1) for s in all_stripes[i*16:(i+1)*16]]))



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



def unique_stripes_to_asm( fo, unique_stripes):
    def stripe_id(stripe):
        return stripe.stripe_id2

    sorted_stripes = sorted( unique_stripes.values(), key=stripe_id)
    fo.write('\n')
    for s in sorted_stripes:
        fo.write("stripe{}\t{}\t; [${:X}] {}\n".format( stripe_id(s), hex_byte(s.compressed), stripe_id(s) - 1, hex_byte(s.data, '')))

    fo.write('stripes_indices:\n')
    array_to_asm( fo, ["stripe{}".format( stripe_id(s)) for s in sorted_stripes], "!word")


def stripes_to_disk( stripes):

    disk = bytearray()

    for s in stripes[0:min( (len(stripes) // 2) - 1, (DISK_SIZE//2) - 1)]:
        sid = (s.stripe_id2 - 1) * 2
        assert sid < 65536
        disk.append( sid & 0xFF)
        disk.append( sid >> 8)

    disk.append( 0xFF)
    disk.append( 0xFF)
    #disk.extend( bytearray( 143360 - len(disk)))

    with open("stripes.dsk","bw") as fo:
        fo.write( disk_2_dos( disk))



def disk_2_dos( disk):

    disk = bytearray( disk)

    dos_sector= [0x0, 0xd, 0xb, 0x9, 0x7, 0x5, 0x3, 0x1,
                 0xe, 0xc, 0xa, 0x8, 0x6, 0x4, 0x2, 0xf]

    prodos_sector = [0x0, 0x8, 0x1, 0x9, 0x2, 0xa, 0x3, 0xb,
                     0x4, 0xc, 0x5, 0xd, 0x6, 0xe, 0x7, 0xf]

    # Dos order : https://en.wikipedia.org/wiki/Apple_DOS

    dos_sector= [0x0, 0x7, 0xe, 0x6, 0xd, 0x5, 0xc, 0x4,
                 0xb, 0x3, 0xa, 0x2, 0x9, 0x1, 0x8, 0xf]


    if len(disk) > DISK_SIZE:
        print("Disk image too big by {} bytes, truncating...".format(len(disk) - DISK_SIZE))
        disk = disk[0:DISK_SIZE]
    elif len(disk) < DISK_SIZE:
        print("Disk image too small ({}), extending to disk size...".format(len(disk)))
        disk.extend( bytearray( DISK_SIZE - len(disk)))
    else:
        print("disk_2_dos : putting {} bytes on a disk of {}".format(len(disk), DISK_SIZE))

    # dos_sector = list( range( 16))
    disk_dos = bytearray( DISK_SIZE)

    for track in range(35):
        for sector in range(16):
            track_offset = track * 16 * 256

            dos_ofs = track_offset + dos_sector[sector]*256
            dsk_ofs = track_offset + sector*256
            disk_dos[ dos_ofs:dos_ofs+256] =  disk[ dsk_ofs:dsk_ofs+256] # [sector for i in range(256)]

    return disk_dos
