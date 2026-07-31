import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.config import DATA_DIR

logger = logging.getLogger(__name__)

# Cloudflare R2 / S3 Storage Bucket Credentials
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", os.getenv("S3_BUCKET_NAME", "doc-analyser-bucket"))
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID", ""))
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", ""))
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", os.getenv("S3_ENDPOINT_URL", ""))

LOCAL_BUCKET_DIR = DATA_DIR / "bucket"
LOCAL_BUCKET_DIR.mkdir(parents=True, exist_ok=True)

class StorageBucketManager:
    """
    Storage Bucket Manager supporting:
    1. Cloudflare R2 / AWS S3 Storage Buckets (via boto3)
    2. Local persistent disk fallback (Railway volume storage)
    """

    def __init__(self):
        self.bucket_name = R2_BUCKET_NAME
        self.s3_client = None

        if R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY and R2_ENDPOINT_URL:
            try:
                import boto3
                self.s3_client = boto3.client(
                    "s3",
                    endpoint_url=R2_ENDPOINT_URL,
                    aws_access_key_id=R2_ACCESS_KEY_ID,
                    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
                    region_name="auto"
                )
                logger.info(f"Initialized Cloudflare R2 / S3 Storage Bucket client for '{self.bucket_name}'.")
            except Exception as e:
                logger.warning(f"Failed to initialize boto3 S3 client: {e}. Using local volume bucket fallback.")

    def save_file(self, filename: str, content: bytes, content_type: str = "application/pdf") -> Dict[str, Any]:
        """
        Saves a PDF file to the Cloudflare R2 / S3 Bucket or local volume storage.
        """
        if self.s3_client:
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=filename,
                    Body=content,
                    ContentType=content_type
                )
                url = f"{R2_ENDPOINT_URL}/{self.bucket_name}/{filename}"
                logger.info(f"Uploaded '{filename}' to Cloudflare R2 / S3 Storage Bucket.")
                return {
                    "storage_type": "Cloudflare R2 / S3 Bucket",
                    "bucket_name": self.bucket_name,
                    "filename": filename,
                    "size_bytes": len(content),
                    "url": url
                }
            except Exception as e:
                logger.warning(f"Storage Bucket upload failed: {e}. Falling back to local storage.")

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
        Retrieves raw PDF file bytes from the bucket or local disk.
        """
        if self.s3_client:
            try:
                resp = self.s3_client.get_object(Bucket=self.bucket_name, Key=filename)
                return resp["Body"].read()
            except Exception as e:
                logger.warning(f"Error fetching from S3 bucket: {e}")

        file_path = LOCAL_BUCKET_DIR / filename
        if file_path.exists():
            with open(file_path, "rb") as f:
                return f.read()

        return None

    def list_files(self) -> List[Dict[str, Any]]:
        """
        Lists all PDF files stored in the storage bucket.
        """
        files = []
        if self.s3_client:
            try:
                resp = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
                for obj in resp.get("Contents", []):
                    files.append({
                        "filename": obj["Key"],
                        "size_bytes": obj["Size"],
                        "last_modified": obj["LastModified"].isoformat(),
                        "storage_type": "Cloudflare R2 / S3"
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
        Deletes a PDF file from the storage bucket.
        """
        deleted = False
        if self.s3_client:
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
