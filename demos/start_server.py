import os
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 8000
ROOT_DIR = 'htdocs'


def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler):
    print('starting server...')
    web_dir = os.path.join(os.path.dirname(__file__), ROOT_DIR)
    os.chdir(web_dir)
    server_address = ('', 8000)
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()
    print('server started')


if __name__ == '__main__':
    run()
