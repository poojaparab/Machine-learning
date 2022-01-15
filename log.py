import logging
import os

def custom_log(path, file):
    log_file = os.path.join(path, file)
    if not os.path.isfile(log_file):
        open(log_file, "w+").close()
    console_logging_format = "%(levelname)s %(message)s"
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
