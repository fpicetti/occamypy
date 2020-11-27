class Logger:
    """System logger class"""
    
    def __init__(self, file, bufsize=1):
        """Initialize writing to a file logfile"""
        self.file = open(file, "a", bufsize)
        return
    
    def __del__(self):
        """Destructor where the log file is closed"""
        self.file.close()
        return
    
    def addToLog(self, msg):
        """Function to write message to log file"""
        self.file.write(msg + "\n")
        return
