"""

FF7 : process_a_frame_loop2:
1077 : load_current_stripe
1080 : data == stripe number

2F40 : strip_indices



85 centorids
unchanged avg:82.32714992389649, stddev:9.068949250432643
11593 unique stripes, stored with 37064 bytes
8635 length-1 stripes, totalling 29388 bytes
163081 stripes were built, 11593 are unique

10668 unique stripes, stored with 34784 bytes
7823 frequency-1 stripes, totalling 27188 bytes
63 base tiles
161613 stripes were built, 10668 are unique

unchanged avg:81.73848934550979, stddev:9.162873654288592
15035 unique stripes, stored with 47217 bytes
11699 frequency-1 stripes, totalling 38842 bytes
127 base tiles
164866 stripes were built, 15035 are unique


"""
import time
import os
import glob
import curses

import xxhash
from scipy.cluster import vq
import numpy as np
from PIL import Image

from utils import *

# Onslaught like parameters
# CODEBOOK_SIZE = 125 # black, white, transparent included
# TILE_SIZE = 16 # square tiles of NxN pixels
# WIDTH = 20 # 24
# HEIGHT = 12 # 20

# I reached 170K with these settings
CODEBOOK_SIZE = 41 # black, white, transparent included
TILE_SIZE = 8 # square tiles of NxN pixels
WIDTH = 24
HEIGHT = 20

IMG_FROM_TO = 00,1000

COLORS = 2


print("Breaking {}x{} frames, from {} to {}.".format( WIDTH*TILE_SIZE, HEIGHT*TILE_SIZE, IMG_FROM_TO[0], IMG_FROM_TO[1]))

PNG_PREFIX="bad_apple_hgr_{}x{}x{}".format( WIDTH, HEIGHT, TILE_SIZE)

if False or not os.path.exists("{}/{}00009.png".format(IMG_PREFIX, PNG_PREFIX)):
    ffmpeg( "-i data/bad_apple_original.mp4 -an  -vf fps=10,scale={}:{}  {}/{}%05d.png".format( WIDTH*TILE_SIZE, HEIGHT*TILE_SIZE, IMG_PREFIX, PNG_PREFIX))

# MUL_VEC = np.zeros( (TILE_SIZE ** 2, ), dtype=np.uint64).transpose()
# for i in range(TILE_SIZE ** 2):
#     MUL_VEC[i] = 2 ** i


def hash_npa( a):
    return xxhash.xxh64( a.view(np.uint8)).hexdigest()
    #return bitstring.Bits( a.astype( int)).tobytes()
    return int(a.dot( MUL_VEC))


#print( hash_npa( np.zeros( (TILE_SIZE**2, ), dtype=np.bool)))
#print( MUL_VEC)
#exit()

def image_to_tiles( filename, tiles):
    # ndx = 0
    # tmap = dict()
    # for tile in tiles:
    #     tmap[ hash_npa( tile.flatten()) ] = ndx
    #     ndx += 1

    img = Image.open(filename)
    data1 = img.convert('L').tobytes()
    img.close()
    data = [x >> 7 for x in data1] # From grayscale to 2 bits per pixel


    a = np.asarray(data, dtype=np.bool_).reshape( (HEIGHT*TILE_SIZE, WIDTH*TILE_SIZE,) )

    pic = np.zeros( ( HEIGHT, WIDTH ) )

    for y in range(0, HEIGHT*TILE_SIZE, TILE_SIZE):
        for x in range(0, WIDTH*TILE_SIZE, TILE_SIZE):
            tile = a[y:y+TILE_SIZE,x:x+TILE_SIZE]

            s = tile.sum() # number of True

            if  0 < s < TILE_SIZE**2: # forget white and blacks
                tiles.append( tile )
                pic[y // TILE_SIZE, x // TILE_SIZE] = len( tiles)
            elif s == 0:
                pic[y // TILE_SIZE, x // TILE_SIZE] = 0
            elif s == 1:
                pic[y // TILE_SIZE, x // TILE_SIZE] = 1

    return a, pic

# 3:41 = 221 sec = 2210 pictures

tiles = [ np.zeros( ( TILE_SIZE, TILE_SIZE, ), dtype=np.bool_ ),
          np.ones( ( TILE_SIZE, TILE_SIZE, ), dtype=np.bool_ ) ]

pictures= []
pictures_as_tiles = []

img_ndx = 0
for filename in sorted(glob.glob('{}/{}0*.png'.format(IMG_PREFIX, PNG_PREFIX))):
    img_ndx += 1

    if img_ndx % 200 == 0:
        print( img_ndx, end="\r", flush=True)

    if img_ndx % 10 == 0:
        continue

    from_, to = IMG_FROM_TO
    if not (from_ <= img_ndx <= to):
        continue

    original_pic, pic = image_to_tiles( filename, tiles)
    pictures.append( original_pic)
    pictures_as_tiles.append( pic)

print("Read {} frames".format(len(pictures)))

# p = pictures[ 80 ]
# print(p)
# print(p.astype(np.byte)*255)
# img = Image.fromarray( p.astype(np.uint8) * 255, mode='L')
# img.save("zulu.png", "png")
# exit()

# Tiles to vectors. There will be duplicates of course.
tiles = np.asarray( [ t.flatten() for t in tiles])

print("There are {} tiles".format(len(tiles)))

print("Finding unique tiles")
unique_tiles = dict()
for tile in tiles:
    h = hash_npa(tile)
    # print(h)
    if h not in unique_tiles:
        unique_tiles[h] = tile
print("There are {} unique tiles".format(len(unique_tiles)))


print("Computing centroids")

# centroids = np.append( np.ones( (1,4*4), dtype=float), np.zeros( (CODEBOOK_SIZE-1,4*4)), axis=0 )
# centroids, labels = vq.kmeans2( tw, centroids, minit='matrix')

cb_size = min( len( tiles), CODEBOOK_SIZE)
centroids, labels = vq.kmeans2( vq.whiten( tiles), cb_size, iter=15, minit='points') # ++ takes at least 6 minutes, never went to the end
centroids = np.round_( (centroids * (0.5))).astype(np.bool_)

# Remove white and black that may have been found by kmean
centroids = [x for x in filter( lambda t: 0 < t.sum() < TILE_SIZE**2, centroids)]
# centroids = np.insert( centroids, 0, tiles[0], 0)
# centroids = np.insert( centroids, 1, tiles[1], 0)
z = [ tiles[0], tiles[1] ]
z.extend( centroids)
centroids = z
print("{} base tiles + transparent".format(len(centroids)))


# for c in centroids:
#     print(c)

#centroids = [c.reshape( (TILE_SIZE, TILE_SIZE,)) for c in centroids]

# centroids_hashes = dict()
# for centroid in centroids:
#     c_hash = centroid.dot( MUL_VEC)
#     centroids_hashes[ c_hash ] = centroid.reshape( (TILE_SIZE, TILE_SIZE, ) )

#     #print( type(c_hash))

# for c in centroids:
#     print(c.reshape( (1, TILE_SIZE, TILE_SIZE,)))
# print(tw.shape)
# print("{} tiles, {} labels, centroids {}".format( len( tiles), labels.shape[0], len( centroids)))



print("Finding closest tiles")

# Maps tiles hash to (centroid) tile number
decimated_tiles = dict()

for tk, tile in unique_tiles.items():

    # print(tk)
    # print(tile)

    max_diff = TILE_SIZE ** 2
    for ci in range( len(centroids)):

        centroid = centroids[ci]
        diff = np.bitwise_xor(tile,centroid).sum() # The smaller the better (xor gives 0 if equals, 1 else)
        if diff < max_diff:
            max_diff = diff
            best_c = ci

    # print(best_c)
    # print( centroids[best_c])
    decimated_tiles[tk] = best_c
    #print("-"*80)
    #print(t.reshape( (TILE_SIZE, TILE_SIZE, )))
    #print(best_c.reshape( (TILE_SIZE, TILE_SIZE, )))


tiled_stream = []

print("Retiling images")
for p_ndx in range( len( pictures)):
    p = pictures[p_ndx]
    new_p = np.zeros( (HEIGHT, WIDTH, ), dtype=np.byte)

    p_comp = np.zeros( (HEIGHT*TILE_SIZE, WIDTH*TILE_SIZE, ), dtype=np.bool_)

    for y in range(0, HEIGHT*TILE_SIZE, TILE_SIZE):
        for x in range(0, WIDTH*TILE_SIZE, TILE_SIZE):
            tile = p[y:y+TILE_SIZE,x:x+TILE_SIZE]
            t_hash = hash_npa(tile.flatten())
            cid = decimated_tiles[t_hash]
            p_comp[y:y+TILE_SIZE,x:x+TILE_SIZE] = centroids[cid].reshape( (TILE_SIZE, TILE_SIZE,))

            new_p[y // TILE_SIZE, x // TILE_SIZE] = cid

            # print( t_hash)
            # print(p_comp[y:y+TILE_SIZE,x:x+TILE_SIZE])
            # print(decimated_tiles[t_hash].reshape( (TILE_SIZE, TILE_SIZE,)))

    tiled_stream.extend( new_p.flatten().tolist())

    #print(p_comp.astype(np.uint8))
    #print(p_comp.astype(np.uint8) * 255)

    img = Image.fromarray( p_comp.astype(np.uint8) * 255, mode='L')
    img.save("{}/zulu{:05}.png".format(IMG_PREFIX, p_ndx), "png")
    img.close()

print("Tiled stream length is {}".format( len(tiled_stream)))
ffmpeg(r"-y -framerate 10 -i {}/zulu%05d.png video.avi".format(IMG_PREFIX))

# for t in range( 0, len(tiled_stream), WIDTH*HEIGHT):
#     print( tiled_stream[t:t+WIDTH*HEIGHT])

special_tiles = SpecialTiles( 0, 1, 0x7E) # 0x7F not possible, >= 0x80 neither
delta_frames_stream = make_delta_frames_stream( tiled_stream, special_tiles, WIDTH*HEIGHT)
all_stripes = make_stripes( delta_frames_stream, special_tiles, WIDTH*HEIGHT, 8)
unique_stripes = simplify_stripes( all_stripes)
compute_stripes_frequencies( all_stripes)
stats_unique_stipes( unique_stripes.values())

print("{} stripes were built, {} are unique".format( len( all_stripes), len( unique_stripes)))

simple_huffman( unique_stripes, all_stripes)


def reverseBits(num,bitSize):

     # convert number into binary representation
     # output will be like bin(10) = '0b10101'
     binary = bin(num)

     # skip first two characters of binary
     # representation string and reverse
     # remaining string and then append zeros
     # after it. binary[-1:1:-1]  --> start
     # from last character and reverse it until
     # second last character from left
     reverse = binary[-1:1:-1]
     reverse = reverse + (bitSize - len(reverse))*'0'

     # converts reversed binary string into integer
     return int(reverse,2)

with open("data.a","w") as fo:

    fo.write("tiles:\n")

    for t in centroids:
        #print(t)
        fo.write("\t!byte " + ",".join( ["${:02x}".format( reverseBits( x, 8)) for x in reversed( np.packbits(t))]) + "\n")

    unique_stripes_to_asm( fo, unique_stripes)

#stripes_to_disk( all_stripes)

exit()

COMPARE_PIC_NDX = len(pictures) // 4

stdscr = curses.initscr()
stdscr.nodelay(1)
curses.cbreak()
colors = [ ' '*2, '\u2591'*2,'\u2592'*2,'\u2588'*2, '\u2588'*2 ]

def compress_file( bin_name, bdata):
    bfile = open( bin_name,'bw')
    bfile.write( bdata)
    bfile.close()
    os.system('lz4x.exe -9 -f {} {}'.format(bin_name, bin_name+'lz4'))
    return os.path.getsize( bin_name+'lz4')


to_lz4 = bytearray()
total_on_disk = 0

for COMPARE_PIC_NDX in range(1000):
    #os.system('cls')

    pat = pictures_as_tiles[COMPARE_PIC_NDX]
    p = pictures[COMPARE_PIC_NDX]

    # print(pat.shape)
    for y in range( pat.shape[0]):
        for x in range( pat.shape[1]):
            n = int(pat[y,x])
            pat[y,x] = labels[ n ]

    # print( pat)

    to_lz4.extend(bytearray( pat.astype(np.int16).flatten()))
    if COMPARE_PIC_NDX % 32 == 0:
        #total_on_disk += compress_file( 'test.bin', to_lz4)
        to_lz4 = bytearray()


    decompressed = np.zeros( (height, width), dtype=int )

    for y in range( pat.shape[0]):
        for x in range( pat.shape[1]):
            c = centroids[ int(pat[y,x])].reshape( (4,4) )
            decompressed[ y*4:(y+1)*4, x*4:(x+1)*4 ] = c


    #decompressed = np.round_( decompressed)

    for y in range(40):
        l = ""
        for x in range(width):
            l += colors[decompressed[y,x]]

        l += " | "
        for x in range(width):
            l += colors[int(p[y,x])]

        l += " |"
        print(l)

        stdscr.addstr(y,1,l)

    stdscr.addstr(1,1,"{} on disk => bytes per frame = {}".format(total_on_disk, int( total_on_disk / (1+COMPARE_PIC_NDX))))
    stdscr.refresh()

    # codebook_ram = TILE_SIZE*TILE_SIZE*CODEBOOK_SIZE
    # print("Codebook size = {}".format( codebook_ram))
    # tile_ram = 40*48/(TILE_SIZE*TILE_SIZE)
    # print("Frame size = {}".format( tile_ram))
    # print("Frame in memeory size = {}".format( (34000 - codebook_ram) // tile_ram ))


    time.sleep(0.1)

    c = stdscr.getch()
    if c == ord('q'):
        break


curses.endwin()

# codebook, distortion = vq.kmeans( tw, 100, iter=20, thresh=1e-12,)

# print("Codebook: {}".format(codebook.shape[0]))
# compressed_tiles, distortion  = vq.vq( tiles, codebook)

# print( compressed_tiles)

# print( pictures_as_tiles[0])
