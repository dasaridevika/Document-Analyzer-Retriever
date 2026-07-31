import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.config import DATA_DIR

logger = logging.getLogger(__name__)

# Comprehensive Railway Bucket & S3 Environment Variable Resolver
BUCKET_NAME = (
    os.getenv("AWS_STORAGE_BUCKET_NAME") or
    os.getenv("RAILWAY_BUCKET_NAME") or
    os.getenv("BUCKET_NAME") or
    os.getenv("S3_BUCKET_NAME") or
    os.getenv("R2_BUCKET_NAME") or
    "recorded-case-mw0hrll2-cc"
)

ACCESS_KEY_ID = (
    os.getenv("AWS_ACCESS_KEY_ID") or
    os.getenv("RAILWAY_ACCESS_KEY_ID") or
    os.getenv("ACCESS_KEY_ID") or
    os.getenv("S3_ACCESS_KEY_ID") or
    os.getenv("R2_ACCESS_KEY_ID") or
    ""
)

SECRET_ACCESS_KEY = (
    os.getenv("AWS_SECRET_ACCESS_KEY") or
    os.getenv("RAILWAY_SECRET_ACCESS_KEY") or
    os.getenv("SECRET_ACCESS_KEY") or
    os.getenv("S3_SECRET_ACCESS_KEY") or
    os.getenv("R2_SECRET_ACCESS_KEY") or
    ""
)

ENDPOINT_URL = (
    os.getenv("AWS_ENDPOINT_URL_S3") or
    os.getenv("AWS_ENDPOINT_URL") or
    os.getenv("RAILWAY_ENDPOINT_URL") or
    os.getenv("ENDPOINT_URL") or
    os.getenv("S3_ENDPOINT_URL") or
    os.getenv("R2_ENDPOINT_URL") or
    ""
)

REGION_NAME = os.getenv("S3_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"

# Ensure endpoint URL has https:// prefix if set
if ENDPOINT_URL and not ENDPOINT_URL.startswith("http://") and not ENDPOINT_URL.startswith("https://"):
    ENDPOINT_URL = f"https://{ENDPOINT_URL}"

LOCAL_BUCKET_DIR = DATA_DIR / "bucket"
LOCAL_BUCKET_DIR.mkdir(parents=True, exist_ok=True)

class StorageBucketManager:
    """
    Railway Bucket S3 Storage Bucket Manager.
    Directly uploads PDFs to Railway's S3 Storage Bucket ('recorded-case').
    """

    def __init__(self):
        self.bucket_name = BUCKET_NAME
        self.s3_client = None

        if ACCESS_KEY_ID and SECRET_ACCESS_KEY:
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
                logger.info(f"Successfully initialized Railway S3 Bucket client for '{self.bucket_name}'. Endpoint: {ENDPOINT_URL or 'AWS Default'}")
            except Exception as e:
                logger.error(f"Failed to initialize S3 Bucket client: {e}")
        else:
            logger.warning(f"S3 Credentials missing (ACCESS_KEY_ID / SECRET_ACCESS_KEY). Bucket uploads will fall back to local disk storage.")

    def save_file(self, filename: str, content: bytes, content_type: str = "application/pdf") -> Dict[str, Any]:
        """
        Uploads PDF into Railway S3 Bucket ('recorded-case') and local disk storage.
        """
        s3_uploaded = False

        if self.s3_client and self.bucket_name:
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=filename,
                    Body=content,
                    ContentType=content_type
                )
                logger.info(f"Successfully uploaded '{filename}' to Railway S3 Bucket '{self.bucket_name}'.")
                s3_uploaded = True
            except Exception as e:
                logger.error(f"Failed to upload '{filename}' to Railway S3 Bucket '{self.bucket_name}': {e}")

        # Always save copy to local volume storage as backup
        file_path = LOCAL_BUCKET_DIR / filename
        with open(file_path, "wb") as f:
            f.write(content)

        return {
            "storage_type": f"Railway S3 Bucket ({self.bucket_name})" if s3_uploaded else "Railway Volume Disk",
            "bucket_name": self.bucket_name,
            "filename": filename,
            "size_bytes": len(content),
            "s3_uploaded": s3_uploaded
        }

    def get_file(self, filename: str) -> Optional[bytes]:
        if self.s3_client and self.bucket_name:
            try:
                resp = self.s3_client.get_object(Bucket=self.bucket_name, Key=filename)
                return resp["Body"].read()
            except Exception as e:
                logger.warning(f"Error reading '{filename}' from S3 Bucket: {e}")

        file_path = LOCAL_BUCKET_DIR / filename
        if file_path.exists():
            with open(file_path, "rb") as f:
                return f.read()

        return None

    def list_files(self) -> List[Dict[str, Any]]:
        files = []
        if self.s3_client and self.bucket_name:
            try:
                resp = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
                for obj in resp.get("Contents", []):
                    files.append({
                        "filename": obj["Key"],
                        "size_bytes": obj["Size"],
                        "last_modified": str(obj["LastModified"]),
                        "storage_type": f"Railway S3 Bucket ({self.bucket_name})"
                    })
                if files:
                    return files
            except Exception as e:
                logger.warning(f"Error listing S3 Bucket files: {e}")

        for f in LOCAL_BUCKET_DIR.glob("*.pdf"):
            stat = f.stat()
            files.append({
                "filename": f.name,
                "size_bytes": stat.st_size,
                "last_modified": str(stat.st_mtime),
                "storage_type": "Railway Volume Disk"
            })

        return files

    def delete_file(self, filename: str) -> bool:
        deleted = False
        if self.s3_client and self.bucket_name:
            try:
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=filename)
                deleted = True
            except Exception as e:
                logger.warning(f"Failed to delete '{filename}' from S3 Bucket: {e}")

        file_path = LOCAL_BUCKET_DIR / filename
        if file_path.exists():
            file_path.unlink()
            deleted = True

        return deleted
