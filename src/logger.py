from datetime import datetime


class PyLogger:
    def __init__(self, log_to_file=False, file_path="script"):
        self.log_to_file = log_to_file
        self.file_path = f"logs/{file_path}.log"

        if self.log_to_file:
            # Clear existing log file on creation (optional)
            with open(self.file_path, "w") as f:
                f.write("============ Logging process started ============\n")

    def _log(self, level, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"{timestamp} [{level}]: {message.capitalize()}"

        if self.log_to_file:
            with open(self.file_path, "a") as f:
                f.write(full_message + "\n")
        else:
            print(full_message)

    def info(self, message):
        self._log("INFO", message)

    def warning(self, message):
        self._log("WARN", message)

    def error(self, message):
        self._log("ERROR", message)

    def debug(self, message):
        self._log("DEBUG", message)
