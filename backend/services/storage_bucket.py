import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.config import DATA_DIR

logger = logging.getLogger(__name__)

# Railway Bucket / S3 / R2 Storage Credentials (Loaded from service variables)
BUCKET_NAME = (
    os.getenv("RAILWAY_BUCKET_NAME") or
    os.getenv("BUCKET_NAME") or
    os.getenv("S3_BUCKET_NAME") or
    os.getenv("R2_BUCKET_NAME") or
    os.getenv("AWS_STORAGE_BUCKET_NAME") or
    ""
)

ACCESS_KEY_ID = (
    os.getenv("RAILWAY_ACCESS_KEY_ID") or
    os.getenv("ACCESS_KEY_ID") or
    os.getenv("S3_ACCESS_KEY_ID") or
    os.getenv("R2_ACCESS_KEY_ID") or
    os.getenv("AWS_ACCESS_KEY_ID") or
    ""
)

SECRET_ACCESS_KEY = (
    os.getenv("RAILWAY_SECRET_ACCESS_KEY") or
    os.getenv("SECRET_ACCESS_KEY") or
    os.getenv("S3_SECRET_ACCESS_KEY") or
    os.getenv("R2_SECRET_ACCESS_KEY") or
    os.getenv("AWS_SECRET_ACCESS_KEY") or
    ""
)

ENDPOINT_URL = (
    os.getenv("RAILWAY_ENDPOINT_URL") or
    os.getenv("ENDPOINT_URL") or
    os.getenv("S3_ENDPOINT_URL") or
    os.getenv("R2_ENDPOINT_URL") or
    os.getenv("AWS_ENDPOINT_URL") or
    None
)

REGION_NAME = os.getenv("S3_REGION") or os.getenv("AWS_DEFAULT_REGION") or "auto"

LOCAL_BUCKET_DIR = DATA_DIR / "bucket"
LOCAL_BUCKET_DIR.mkdir(parents=True, exist_ok=True)

class StorageBucketManager:
    """
    Railway Bucket & S3 Storage Bucket Manager.
    Automatically connects to Railway Storage Buckets or persistent volume disk storage.
    """

    def __init__(self):
        self.bucket_name = BUCKET_NAME
        self.s3_client = None

        if self.bucket_name and ACCESS_KEY_ID and SECRET_ACCESS_KEY:
            try:
                import boto3
                client_kwargs = {
                    "aws_access_key_id": ACCESS_KEY_ID,
                    "aws_secret_access_key": SECRET_ACCESS_KEY,
                    "region_name": REGION_NAME
                }
                if ENDPOINT_URL:
                    client_kwargs["endpoint_url"] = ENDPOINT_URL

                self.s3_client = boto3.client("s3", **client_kwargs)
                logger.info(f"Successfully connected to Railway / S3 Storage Bucket '{self.bucket_name}'.")
            except Exception as e:
                logger.warning(f"Failed to initialize Storage Bucket client: {e}. Operating with Railway volume storage.")
        else:
            logger.info("Operating with Railway volume storage (/app/storage). Set RAILWAY_BUCKET_NAME, ACCESS_KEY_ID, SECRET_ACCESS_KEY, ENDPOINT_URL to link a Railway S3 Bucket.")

    def save_file(self, filename: str, content: bytes, content_type: str = "application/pdf") -> Dict[str, Any]:
        """
        Saves uploaded PDF into Railway Storage Bucket (or persistent volume disk).
        """
        if self.s3_client and self.bucket_name:
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=filename,
                    Body=content,
                    ContentType=content_type
                )
                logger.info(f"Uploaded '{filename}' into Railway Storage Bucket '{self.bucket_name}'.")
                return {
                    "storage_type": f"Railway Storage Bucket ({self.bucket_name})",
                    "bucket_name": self.bucket_name,
                    "filename": filename,
                    "size_bytes": len(content)
                }
            except Exception as e:
                logger.warning(f"Railway Storage Bucket upload error: {e}. Saving to local volume storage.")

        # Railway Volume Storage Fallback (/app/storage/bucket/)
        file_path = LOCAL_BUCKET_DIR / filename
        with open(file_path, "wb") as f:
            f.write(content)

        return {
            "storage_type": "Railway Persistent Volume Storage",
            "bucket_name": "railway-volume",
            "filename": filename,
            "size_bytes": len(content),
            "path": str(file_path)
        }

    def get_file(self, filename: str) -> Optional[bytes]:
        """
        Retrieves PDF file bytes from Railway Storage Bucket or volume disk.
        """
        if self.s3_client and self.bucket_name:
            try:
                resp = self.s3_client.get_object(Bucket=self.bucket_name, Key=filename)
                return resp["Body"].read()
            except Exception as e:
                logger.warning(f"Error fetching '{filename}' from Railway Bucket: {e}")

        file_path = LOCAL_BUCKET_DIR / filename
        if file_path.exists():
            with open(file_path, "rb") as f:
                return f.read()

        return None

    def list_files(self) -> List[Dict[str, Any]]:
        """
        Lists all PDF files stored in Railway Storage Bucket or volume disk.
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
                        "storage_type": f"Railway Storage Bucket ({self.bucket_name})"
                    })
                return files
            except Exception as e:
                logger.warning(f"Error listing Railway Bucket files: {e}")

        for f in LOCAL_BUCKET_DIR.glob("*.pdf"):
            stat = f.stat()
            files.append({
                "filename": f.name,
                "size_bytes": stat.st_size,
                "last_modified": str(stat.st_mtime),
                "storage_type": "Railway Persistent Volume"
            })

        return files

    def delete_file(self, filename: str) -> bool:
        """
        Deletes a PDF file from Railway Storage Bucket or volume disk.
        """
        deleted = False
        if self.s3_client and self.bucket_name:
            try:
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=filename)
                deleted = True
            except Exception as e:
                logger.warning(f"Failed to delete '{filename}' from Railway Bucket: {e}")

        file_path = LOCAL_BUCKET_DIR / filename
        if file_path.exists():
            file_path.unlink()
            deleted = True

        return deleted
