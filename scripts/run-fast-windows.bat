@echo off
setlocal
call mvn -q -DskipTests package
if errorlevel 1 exit /b 1
java -Xmx3g -jar target\kion-rl-java-1.0.0.jar %*
endlocal
