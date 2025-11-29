import os
import sys
from src.logging import logging
from src.exception import HeartDiseaseException

class S3Sync:
    def sync_folder_to_s3(self, folder, aws_bucket_url):
        try:  
            logging.info(f"☁️ Syncing folder '{folder}' to '{aws_bucket_url}'") 
            
            command = f"aws s3 sync {folder} {aws_bucket_url} "
            
            # Capture the exit code to check for failures
            exit_code = os.system(command)

            if exit_code != 0:
                raise Exception(f"AWS CLI command failed with exit code {exit_code}")

            logging.info("✅ Syncing successfully.")
            
        except Exception as e:
            raise HeartDiseaseException(e, sys) from e
    
    def sync_folder_from_s3(self, folder, aws_bucket_url):
        try:
            logging.info(f"☁️ Syncing folder '{folder}' from '{aws_bucket_url}'")            
            
            command = f"aws s3 sync {aws_bucket_url} {folder} "
            
            # Capture the exit code to check for failures
            exit_code = os.system(command)

            if exit_code != 0:
                raise Exception(f"AWS CLI command failed with exit code {exit_code}")
                
            logging.info("✅ Syncing successfully.")

        except Exception as e:
            raise HeartDiseaseException(e, sys) from e