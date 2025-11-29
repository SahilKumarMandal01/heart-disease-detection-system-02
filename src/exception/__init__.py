import sys
from src.logging import logging


class HeartDiseaseException(Exception):
    """
    Custom exception for consistent error reporting across the pipeline.
    Handles cases where either:
      - An actual exception traceback exists (wrapped errors)
      - No traceback exists (manual raises)
    """

    def __init__(self, error_message: str, error_details: sys):
        super().__init__(error_message)
        self.error_message = str(error_message)

        # -------------------------------------------
        # Handle traceback safely
        # -------------------------------------------
        try:
            _, _, exc_tb = error_details.exc_info()

            if exc_tb is not None:  # Case: Real underlying exception
                self.file_name = exc_tb.tb_frame.f_code.co_filename
                self.lineno = exc_tb.tb_lineno
            else:  # Case: Manually raised exception (no traceback)
                self.file_name = "N/A"
                self.lineno = "N/A"

        except Exception:
            # Fallback if exc_info() fails for any reason
            self.file_name = "N/A"
            self.lineno = "N/A"

    def __str__(self):
        return (
            f"Error occurred in python script [{self.file_name}] "
            f"at line [{self.lineno}] "
            f"with error message: {self.error_message}"
        )


if __name__ == "__main__":
    try:
        logging.info("Entering try block")
        a = 1 / 0
    except Exception as e:
        raise HeartDiseaseException(e, sys)
