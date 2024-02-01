import logging

name = 'atomdetection-app'
logger = logging.getLogger(name)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Create a file handler and set its level and formatter
file_handler = logging.FileHandler(f'{name}.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)