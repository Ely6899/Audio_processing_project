class FileNotSupportedException(Exception):
    message: str
    file_type: str

    def __init__(self, file_type: str, message: str = "File type is not supported"):
        self.file_type = file_type
        self.message = f"{message}: {file_type}"
        super().__init__(self.message)