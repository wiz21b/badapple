#!/bin/sh

./acme wiz3.a




java -jar AppleCommander-1.3.5.13-ac.jar -d NEW.DSK STARTUP
java -jar AppleCommander-1.3.5.13-ac.jar -d NEW.DSK WIZ4
java -jar AppleCommander-1.3.5.13-ac.jar -d NEW.DSK BADATA

REM java -jar AppleCommander-1.3.5.13-ac.jar -p NEW.DSK STARTUP BIN 0xC00 < STARTUP

python3 cutter.py cut

java -jar AppleCommander-1.3.5.13-ac.jar -p NEW.DSK STARTUP BIN 0xC00 < WIZ4
REM java -jar AppleCommander-1.3.5.13-ac.jar -d NEW.DSK BADATA
REM java -jar AppleCommander-1.3.5.13-ac.jar -p NEW.DSK BADATA BIN 0x4000 < BADATA

python3 cutter.py disk


# c:\port-stc\opt\mame\mame64 apple2p -skip_gameinfo -window -nomax -flop1 NEW.DSK -flop2 cstripes.dsk -rp bios
# \PORT-STC\opt\applewin\Applewin.exe -d1 NEW.DSK -d2 cstripes.dsk

# -speed 40
# -aviwrite z.avi
# ffmpeg -i ~/.mame/snap/z.avi -vf "scale=iw*.5:ih,format=gray" o.avi
mame apple2p -skip_gameinfo -window -nomax -flop1 NEW2.DSK -flop2 cstripes.dsk -rp bios -sound none -aviwrite z.avi
# wine-development ~/AppleWin1.27.13.0/Applewin.exe  -d1  \\home\\stefan\\Dropbox\\bad_apple\\NEW2.DSK -d2  \\home\\stefan\\Dropbox\\bad_apple\\cstripes.dsk -no-printscreen-dlg
