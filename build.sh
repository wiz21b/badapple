#!/bin/sh

./acme wiz3.a
java -jar AppleCommander-1.3.5.13-ac.jar -d NEW.DSK STARTUP
java -jar AppleCommander-1.3.5.13-ac.jar -p NEW.DSK STARTUP BIN 0x4000 < STARTUP
# c:\port-stc\opt\mame\mame64 apple2p -skip_gameinfo -window -nomax -flop1 NEW.DSK -flop2 cstripes.dsk -rp bios
# \PORT-STC\opt\applewin\Applewin.exe -d1 NEW.DSK -d2 cstripes.dsk

# -speed 40
mame apple2p -speed 10  -skip_gameinfo -window -nomax -flop1 NEW.DSK -flop2 cstripes.dsk -rp bios -sound none
# wine-development ~/AppleWin1.27.13.0/Applewin.exe  -d1  \\home\\stefan\\Dropbox\\bad_apple\\NEW.DSK -d2  \\home\\stefan\\Dropbox\\bad_apple\\cstripes.dsk -no-printscreen-dlg
