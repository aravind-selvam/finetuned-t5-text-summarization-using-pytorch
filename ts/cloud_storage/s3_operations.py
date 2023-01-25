import os
class S3Sync:

    def sync_folder_to_s3(self,folder,aws_bucket_url):
        """
        This function takes a folder and an AWS bucket URL as input, and syncs the folder to the AWS
        bucket
        """
        command = f"aws s3 sync {folder} {aws_bucket_url} "
        os.system(command)

    def sync_folder_from_s3(self,folder,aws_bucket_url):
        """
        The function `sync_folder_from_s3` takes in a folder and an aws bucket url and syncs the folder
        with the bucket.
        """
        command = f"aws s3 sync  {aws_bucket_url} {folder} "
        os.system(command)