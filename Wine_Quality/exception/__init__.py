import os
import sys

def error_message_detail(error, error_detail: sys):
    try:
        _, _, exc_tb = error_detail.exc_info()
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
        else:
            file_name = "Unknown file"
            line_number = "Unknown line"

        return f"Error occurred python script name [{file_name}] line number [{line_number}] error message [{str(error)}]"
    except Exception as e:
        return f"Error while logging exception: {str(e)} | Original error: {str(error)}"


class custom_Exception(Exception):
    def __init__(self, error_message, error_detail):
        """
        :param error_message: error message in string format
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        return self.error_message
