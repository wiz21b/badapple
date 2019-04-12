#!/bin/sh

./acme wiz3.a

java -jar AppleCommander-1.3.5.13-ac.jar -d NEW.DSK STARTUP
java -jar AppleCommander-1.3.5.13-ac.jar -d NEW.DSK WIZ4
java -jar AppleCommander-1.3.5.13-ac.jar -d NEW.DSK BADATA

# java -jar AppleCommander-1.3.5.13-ac.jar -p NEW.DSK STARTUP BIN 0xC00 < STARTUP

python3 cutter.py


java -jar AppleCommander-1.3.5.13-ac.jar -p NEW.DSK STARTUP BIN 0xC00 < WIZ4
# java -jar AppleCommander-1.3.5.13-ac.jar -p NEW.DSK BADATA BIN 0x4000 < BADATA

python3 cutter.py


# c:\port-stc\opt\mame\mame64 apple2p -skip_gameinfo -window -nomax -flop1 NEW.DSK -flop2 cstripes.dsk -rp bios
# \PORT-STC\opt\applewin\Applewin.exe -d1 NEW.DSK -d2 cstripes.dsk

# -speed 40
# mame apple2p -speed 10  -skip_gameinfo -window -nomax -flop1 NEW2.DSK -flop2 cstripes.dsk -rp bios -sound none
wine-development ~/AppleWin1.27.13.0/Applewin.exe  -d1  \\home\\stefan\\Dropbox\\bad_apple\\NEW2.DSK -d2  \\home\\stefan\\Dropbox\\bad_apple\\cstripes.dsk -no-printscreen-dlg
