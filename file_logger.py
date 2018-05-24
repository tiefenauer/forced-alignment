from abc import ABC


class FileLogger(ABC):
    def __init__(self, file_path):
        self.logfile = open(file_path, 'w')

    def write(self, line):
        self.logfile.writelines(line + '\n')
        self.logfile.flush()

    def write_tabbed(self, elements):
        line = '\t'.join(str(e) for e in elements)
        self.write(line)

    def close(self):
        self.logfile.close()
