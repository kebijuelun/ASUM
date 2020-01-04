

import serial
ser=serial.Serial("/dev/ttyUSB0",9600,timeout=0.5) #使用USB连接串行口

# ser.write("8v1000\n".encode())
# ser.write("8v200\n".encode())
ser.write("1v0\n".encode())
#

