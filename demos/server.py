"""adapted from: https://gist.github.com/shivakar/82ac5c9cb17c95500db1906600e5e1ea"""
import argparse
import os
import sys
from http.server import SimpleHTTPRequestHandler, HTTPServer
from os.path import realpath, join, dirname, isdir, exists

parser = argparse.ArgumentParser(description='Start simple HTTP server supporting HTTP/1.1 requests (needed to play'
                                             'the aligned audio in HTML5)!')
parser.add_argument('cwd', type=str, nargs='?', default='htdocs',
                    help='(optional) directory to serve from (default: \'htdocs\')')
parser.add_argument('port', type=int, nargs='?', default=8000, help='(optional) port to use (default: 8000)')
args = parser.parse_args()


class RangeHTTPRequestHandler(SimpleHTTPRequestHandler):
    """RangeHTTPRequestHandler is a SimpleHTTPRequestHandler
    with HTTP 'Range' support"""

    def send_head(self):
        """Common code for GET and HEAD commands.
        Return value is either a file object or None
        """

        path = self.translate_path(self.path)
        ctype = self.guess_type(path)

        # Handling file location
        # If directory, let SimpleHTTPRequestHandler handle the request
        if isdir(path):
            return SimpleHTTPRequestHandler.send_head(self)

        # Handle file not found
        if not exists(path):
            return self.send_error(404, self.responses.get(404)[0])

        # Handle file request
        f = open(path, 'rb')
        fs = os.fstat(f.fileno())
        size = fs[6]

        # Parse range header
        # Range headers look like 'bytes=500-1000'
        start, end = 0, size - 1
        print('headers', self.headers.__dict__)
        if 'Range' in self.headers:
            start, end = self.headers.get('Range').strip().strip('bytes=').split('-')
        if start == "":
            # If no start, then the request is for last N bytes
            ## e.g. bytes=-500
            try:
                end = int(end)
            except ValueError as e:
                self.send_error(400, 'invalid range')
            start = size - end
        else:
            try:
                start = int(start)
            except ValueError as e:
                self.send_error(400, 'invalid range')
            if start >= size:
                # If requested start is greater than filesize
                self.send_error(416, self.responses.get(416)[0])
            if end == "":
                # If only start is provided then serve till end
                end = size - 1
            else:
                try:
                    end = int(end)
                except ValueError as e:
                    self.send_error(400, 'invalid range')

        # Correct the values of start and end
        start = max(start, 0)
        end = min(end, size - 1)
        self.range = (start, end)
        # Setup headers and response
        l = end - start + 1
        if 'Range' in self.headers:
            self.send_response(206)
        else:
            self.send_response(200)
        self.send_header('Content-type', ctype)
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Range',
                         'bytes %s-%s/%s' % (start, end, size))
        self.send_header('Content-Length', str(l))
        self.send_header('Last-Modified', self.date_time_string(fs.st_mtime))
        self.end_headers()

        return f

    def copyfile(self, infile, outfile):
        """Copies data between two file objects
        If the current request is a 'Range' request then only the requested
        bytes are copied.
        Otherwise, the entire file is copied using SimpleHTTPServer.copyfile
        """
        if 'Range' not in self.headers:
            SimpleHTTPRequestHandler.copyfile(self, infile, outfile)
            return

        start, end = self.range
        infile.seek(start)
        bufsize = 64 * 1024  # 64KB
        remainder = (end - start) % bufsize
        times = int((end - start) / bufsize)
        steps = [bufsize] * times + [remainder]
        for astep in steps:
            buf = infile.read(bufsize)
            print('sending', infile, len(buf))
            outfile.write(buf)
        return


if __name__ == '__main__':
    print(f'serving from {args.cwd} on port {args.port}')
    os.chdir(join(dirname(realpath(__file__)), args.cwd))
    server_address = ('', args.port)

    HandlerClass = RangeHTTPRequestHandler
    ServerClass = HTTPServer
    HandlerClass.protocol_version = "HTTP/1.1"
    httpd = ServerClass(server_address, HandlerClass)
    sa = httpd.socket.getsockname()
    sys.stderr.write(f'started MJC\n cwd={args.cwd} port={args.port}')
    sys.stderr.flush()
    print(f'Serving HTTP on {sa[0]}:{sa[1]} ... in background')
    httpd.serve_forever()
