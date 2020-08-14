import socket
import time
'''
def server():
  #host = socket.gethostname()   # get local machine name
  host = '192.168.1.21'
  port = 55444  # Make sure it's within the > 1024 $$ <65535 range
  
  s = socket.socket()
  s.bind((host, port))
  
  s.listen(1)
  client_socket, addr = s.accept()
  print("Connection from: " + str(addr))
  while True:
    data = client_socket.recv(1024).decode('utf-8')
    if not data:
      break
    print('From online user: ' + data)
    data = input()
    client_socket.send(data.encode('utf-8'))
  client_socket.close()
'''
class Server(object):
    def __init__(self, port=55444):
        #host = socket.gethostname()   # get local machine name
        self.host = '127.0.0.1'
        self.port = port  # Make sure it's within the > 1024 $$ <65535 range
	
        self.s = socket.socket()
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        #while True:
        #    try:
        self.s.bind((self.host, self.port))
        #        break
        #    except:
        #        self.port += 1
        #        time.sleep(.5)
        
                
    	
        self.s.listen(1)
        self.client_socket, self.addr = self.s.accept()
        print("Connection from {} on port {}".format(str(self.addr), self.port))
    def receive_message(self):
		#while True:
		#print('receiving message...')
        self.data = self.client_socket.recv(1024).decode('utf-8')
		#print('received message:{}'.format(self.data))
			#if not self.data:
				#break
		#print('From online user: ' + self.data)
        return(self.data)
    def send_message(self, message_out):
        #print('sending message: {}'.format(message_out))
        self.message_out = message_out
        self.client_socket.send(self.message_out.encode('utf-8'))
        #print('message sent')
	#self.client_socket.close()



if __name__ == '__main__':
    server = Server()
	#server.receive_message()
    while True:
        server.send_message('hello')
        server.receive_message()