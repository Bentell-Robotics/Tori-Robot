import socket
import time
# This is the file that works, so we'll use this to 
# communicate bidirectionally with host computer
'''
def client():
	#host = socket.gethostname()  # get local machine name
	host = '192.168.1.21'
	port = 55444  # Make sure it's within the > 1024 $$ <65535 range
	  
	s = socket.socket()
	s.connect((host, port))
	  
	message = str(raw_input('-> '))
	while message != 'q':
		s.send(message.encode('utf-8'))
		data = s.recv(1024).decode('utf-8')
		print('Received from server: ' + data)
		message = input('==> ')
	s.close()
'''
class Client(object):
    #host = socket.gethostname()  # get local machine name
    def __init__(self, port=55444):
        self.host = '127.0.0.1'
        self.port = port  # Make sure it's within the > 1024 $$ <65535 range
		  
        self.s = socket.socket()
        #while True:
        #    try:
        self.s.connect((self.host, self.port))
        #        break
        #    except:
        #        pass
        time.sleep(1)
        self.message_out = 'initial message'
    def send_message(self, message_out):
        print('sending message:{}'.format(message_out))
        self.message_out = message_out
        self.s.send(self.message_out.encode('utf-8'))
    def receive_message(self):
	#while self.message_out != 'q':
        print('receiving message...')
        while True:
            try:
                self.message_in = self.s.recv(1024).decode('utf-8')
                print('Received from server: ' + self.message_in)
            except:
                continue
            else:
                break
        #self.message_out = raw_input('==> ')
        return(self.message_in)
    def close(self):
        self.s.close()






if __name__ == '__main__':
  client = Client()
  '''
  time.sleep(3)
  client.send_message('bananas')
  time.sleep(5)
  client.receive_message()
  time.sleep(2)
  client.receive_message()
  '''
  