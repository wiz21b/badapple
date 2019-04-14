c:\PORT-STC\opt\acme\acme wiz3.a
java -jar AppleCommander-1.3.5.13-ac.jar -d NEW.DSK STARTUP
java -jar AppleCommander-1.3.5.13-ac.jar -d NEW.DSK WIZ4
java -jar AppleCommander-1.3.5.13-ac.jar -d NEW.DSK BADATA

REM java -jar AppleCommander-1.3.5.13-ac.jar -p NEW.DSK STARTUP BIN 0xC00 < STARTUP

python cutter.py cut

java -jar AppleCommander-1.3.5.13-ac.jar -p NEW.DSK STARTUP BIN 0xC00 < WIZ4
REM java -jar AppleCommander-1.3.5.13-ac.jar -d NEW.DSK BADATA
REM java -jar AppleCommander-1.3.5.13-ac.jar -p NEW.DSK BADATA BIN 0x4000 < BADATA

python cutter.py disk

REM Making a release
del BadApple.zip
7z a -tzip BadApple.zip BAD_APPLE.DSK BAD_APPLE_DATA.DSK

REM -speed 40
REM c:\port-stc\opt\mame\mame64 apple2p -skip_gameinfo -window -nomax -flop1 BAD_APPLE.DSK -flop2 BAD_APPLE_DATA.DSK -rp bios
\PORT-STC\opt\applewin\Applewin.exe -d1 BAD_APPLE.DSK -d2 BAD_APPLE_DATA.DSK
