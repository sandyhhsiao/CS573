import pyshark 

packet = pyshark.CaptureFile("test.pcap", display-filter="udp") 

for pkt in packet: 

	packet.ip.src 
	packet.se
