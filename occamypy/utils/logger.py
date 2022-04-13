class Logger:
    """System logger class to store messages to a log file"""
    def __init__(self, file, bufsize: int = 1):
        """
        Logger constructor
        
        Args:
            file: path/to/file to write the log
            bufsize: buffer size
        """
        self.file = open(file, "a", bufsize)
    
    def __del__(self):
        """Close the log file"""
        self.file.close()
        return
    
    def addToLog(self, msg: str):
        """Write a message to log file
        
        Args:
            msg: message to be added to log file
        """
        self.file.write(msg + "\n")
        return
