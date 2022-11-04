import os
import logging
import logging.config

class BulldozerLogger:

    __instance = None

    @staticmethod
    def getInstance(loggerFilePath: str):
        """
            Bulldozer logger singleton

            We use the classic levels:
                NOTSET
                DEBUG
                INFO
                WARNING ERROR
                CRITICAL
        """
        if BulldozerLogger().__instance is None :

            # Create the Logger
            # Sub folders will inherits from the logger configuration, hence
            # we need to give the root package directory name of Bulldozer
            logger = logging.getLogger("bulldozer")
            logger.setLevel(logging.DEBUG)

            # create file handler which logs even debug messages
            fh = logging.FileHandler(filename=loggerFilePath, mode='w')
            fh.setLevel(logging.DEBUG)

            LOG_FORMAT = '%(name)s - %(asctime)s - %(levelname)s - %(message)s'
            logger_formatter = logging.Formatter(LOG_FORMAT)
            fh.setFormatter(logger_formatter)

            logger.addHandler(fh)

            BulldozerLogger.__instance = logger
        
        return BulldozerLogger.__instance


