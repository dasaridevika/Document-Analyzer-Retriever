import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.config import DATA_DIR

logger = logging.getLogger(__name__)

# AWS S3 / S3-compatible Storage Bucket Credentials (Loaded strictly from service environment variables)
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME") or os.getenv("AWS_STORAGE_BUCKET_NAME") or os.getenv("R2_BUCKET_NAME") or ""
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("R2_ACCESS_KEY_ID") or ""
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("R2_SECRET_ACCESS_KEY") or ""
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL") or os.getenv("AWS_ENDPOINT_URL") or os.getenv("R2_ENDPOINT_URL") or None
AWS_REGION = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "us-east-1"

LOCAL_BUCKET_DIR = DATA_DIR / "bucket"
LOCAL_BUCKET_DIR.mkdir(parents=True, exist_ok=True)

class StorageBucketManager:
    """
    AWS S3 & S3-Compatible Storage Bucket Manager.
    All keys and bucket names are injected strictly via environment variables.
    """

    def __init__(self):
        self.bucket_name = S3_BUCKET_NAME
        self.s3_client = None

        if self.bucket_name and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            try:
                import boto3
                client_kwargs = {
                    "aws_access_key_id": AWS_ACCESS_KEY_ID,
                    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
                    "region_name": AWS_REGION
                }
                if S3_ENDPOINT_URL:
                    client_kwargs["endpoint_url"] = S3_ENDPOINT_URL

                self.s3_client = boto3.client("s3", **client_kwargs)
                logger.info(f"Successfully initialized S3 Storage Bucket client for '{self.bucket_name}'.")
            except Exception as e:
                logger.warning(f"Failed to initialize S3 client: {e}. Operating with local volume disk storage.")
        else:
            logger.info("No S3 credentials provided in service environment variables. Operating with local volume disk storage.")

    def save_file(self, filename: str, content: bytes, content_type: str = "application/pdf") -> Dict[str, Any]:
        """
        Saves file to S3 Bucket or local volume disk storage.
        """
        if self.s3_client and self.bucket_name:
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=filename,
                    Body=content,
                    ContentType=content_type
                )
                logger.info(f"Uploaded '{filename}' to S3 Storage Bucket '{self.bucket_name}'.")
                return {
                    "storage_type": "AWS S3 / S3-Compatible Bucket",
                    "bucket_name": self.bucket_name,
                    "filename": filename,
                    "size_bytes": len(content)
                }
            except Exception as e:
                logger.warning(f"S3 Storage Bucket upload error: {e}. Saving to local volume storage.")

        # Local volume disk storage fallback
        file_path = LOCAL_BUCKET_DIR / filename
        with open(file_path, "wb") as f:
            f.write(content)

        return {
            "storage_type": "Railway Persistent Volume Storage",
            "bucket_name": "local-volume",
            "filename": filename,
            "size_bytes": len(content),
            "path": str(file_path)
        }

    def get_file(self, filename: str) -> Optional[bytes]:
        """
        Retrieves raw file bytes from S3 Bucket or local disk.
        """
        if self.s3_client and self.bucket_name:
            try:
                resp = self.s3_client.get_object(Bucket=self.bucket_name, Key=filename)
                return resp["Body"].read()
            except Exception as e:
                logger.warning(f"Error fetching '{filename}' from S3 bucket: {e}")

        file_path = LOCAL_BUCKET_DIR / filename
        if file_path.exists():
            with open(file_path, "rb") as f:
                return f.read()

        return None

    def list_files(self) -> List[Dict[str, Any]]:
        """
        Lists all files in S3 Bucket or local disk storage.
        """
        files = []
        if self.s3_client and self.bucket_name:
            try:
                resp = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
                for obj in resp.get("Contents", []):
                    files.append({
                        "filename": obj["Key"],
                        "size_bytes": obj["Size"],
                        "last_modified": obj["LastModified"].isoformat(),
                        "storage_type": "AWS S3 / S3-Compatible"
                    })
                return files
            except Exception as e:
                logger.warning(f"Error listing S3 bucket files: {e}")

        # Local volume list fallback
        for f in LOCAL_BUCKET_DIR.glob("*.pdf"):
            stat = f.stat()
            files.append({
                "filename": f.name,
                "size_bytes": stat.st_size,
                "last_modified": str(stat.st_mtime),
                "storage_type": "Local Volume Disk"
            })

        return files

    def delete_file(self, filename: str) -> bool:
        """
        Deletes a file from S3 Bucket or local disk.
        """
        deleted = False
        if self.s3_client and self.bucket_name:
            try:
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=filename)
                deleted = True
            except Exception as e:
                logger.warning(f"Failed to delete '{filename}' from S3 bucket: {e}")

        file_path = LOCAL_BUCKET_DIR / filename
        if file_path.exists():
            file_path.unlink()
            deleted = True

        return deleted
