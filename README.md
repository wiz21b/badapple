Bad Apple Demo
--------------

This demo is the classic Bad Apple demo.

How to run
----------

Download the [release](https://github.com/wiz21b/badapple/releases/download/1.0/BadApple.zip).

Open `BAD_DATA.DSK` file in your favourite emulator (I tested Mame and AppleWin).
It will run automatically. When requested to flip the disk, insert `BAD_APPLE_DATA.DSK` in the same drive.
Hit enter, and enjoy !


Background information
----------------------


I decided to do it after watching the demo from Onslaught on C64. It
sounded sad to me that the Apple didn't run the Bad Apple :-)

It was made specifically for the Apple ][+ with floppy disk because
that's the computer I had when I was a kid. I also chose that machine
because of its hardware limitations.

For those interested, my code is here, "as is".

If you're really interested, look at the kmean.py file.  That's the
compressor. This is what it does:

1. drop some frames in the original video (chosen wisely)
2. compute the delta between each frame, let's call that d-frames
3. cut each d-frame in 8x8 tiles
4. reduce the set of all tiles of all d-frames using k-means approach
5. use a form of RLE encoding to compress each run of tiles in d-frame (this gives "stripes")
6. use a Huffman style dictionary to compress each stripe (Huffman is not practical on 
   low performance machines, so I had to make an "approximation" to reach a better
   speed/compression balance.
7. store the data on the disk (regular Prodos sectors, not a file)

Steps 4 and 6 are clearly the more sophisticated ones.

The code is not super optimized and neither is the compressor.  Giving
it more time, I think I could improve some more the screen size of the
video. For exampme, I have the impression that using 16x16 tiles might
be more efficient.

~~If you ever have the opportunity to run it on real Apple, please send
me a video. That would be a very much apprciated gift.~~ It works on real hardware !

For those old enough and in the specific niche, I used to be called
The Wizard of the Imphobia group. By that time it was customary
to send some greetings, so greetings to Scorpio, PL, Darkness,
the Overseas team, my family,...
