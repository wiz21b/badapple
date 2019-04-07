c:\PORT-STC\opt\acme\acme wiz3.a
java -jar AppleCommander-1.3.5.13-ac.jar -d NEW.DSK STARTUP
java -jar AppleCommander-1.3.5.13-ac.jar -p NEW.DSK STARTUP BIN 0x4000 < STARTUP
REM -speed 40
REM c:\port-stc\opt\mame\mame64 apple2p -skip_gameinfo -window -nomax -flop1 NEW.DSK -flop2 cstripes.dsk -rp bios
\PORT-STC\opt\applewin\Applewin.exe -d1 NEW.DSK -d2 cstripes.dsk
