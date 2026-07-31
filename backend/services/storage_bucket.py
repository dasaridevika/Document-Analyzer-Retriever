import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.config import DATA_DIR, BACKUP_DATA_DIR, get_clean_env

logger = logging.getLogger(__name__)

BUCKET_NAME = (
    get_clean_env("AWS_STORAGE_BUCKET_NAME") or
    get_clean_env("RAILWAY_BUCKET_NAME") or
    get_clean_env("BUCKET_NAME") or
    get_clean_env("S3_BUCKET_NAME") or
    get_clean_env("R2_BUCKET_NAME") or
    ""
)

ACCESS_KEY_ID = (
    get_clean_env("AWS_ACCESS_KEY_ID") or
    get_clean_env("RAILWAY_ACCESS_KEY_ID") or
    get_clean_env("ACCESS_KEY_ID") or
    get_clean_env("S3_ACCESS_KEY_ID") or
    get_clean_env("R2_ACCESS_KEY_ID") or
    ""
)

SECRET_ACCESS_KEY = (
    get_clean_env("AWS_SECRET_ACCESS_KEY") or
    get_clean_env("RAILWAY_SECRET_ACCESS_KEY") or
    get_clean_env("SECRET_ACCESS_KEY") or
    get_clean_env("S3_SECRET_ACCESS_KEY") or
    get_clean_env("R2_SECRET_ACCESS_KEY") or
    ""
)

ENDPOINT_URL = (
    get_clean_env("AWS_ENDPOINT_URL_S3") or
    get_clean_env("AWS_ENDPOINT_URL") or
    get_clean_env("RAILWAY_ENDPOINT_URL") or
    get_clean_env("ENDPOINT_URL") or
    get_clean_env("S3_ENDPOINT_URL") or
    get_clean_env("R2_ENDPOINT_URL") or
    ""
)

REGION_NAME = os.getenv("S3_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"

if ENDPOINT_URL and not ENDPOINT_URL.startswith("http://") and not ENDPOINT_URL.startswith("https://"):
    ENDPOINT_URL = f"https://{ENDPOINT_URL}"

# Directories
PRIMARY_BUCKET_DIR = DATA_DIR / "bucket"
PRIMARY_BUCKET_DIR.mkdir(parents=True, exist_ok=True)

BACKUP_BUCKET_DIR = BACKUP_DATA_DIR / "bucket"
BACKUP_BUCKET_DIR.mkdir(parents=True, exist_ok=True)

class StorageBucketManager:
    """
    Railway Bucket S3 Storage & Persistent Volume Manager.
    Uploads PDFs to Railway's S3 Bucket 'recorded-case' and local volume storage.
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
                
                try:
                    buckets_resp = self.s3_client.list_buckets()
                    b_list = [b["Name"] for b in buckets_resp.get("Buckets", [])]
                    if b_list:
                        self.bucket_name = b_list[0]
                        logger.info(f"Auto-detected Railway S3 Bucket name: '{self.bucket_name}'")
                except Exception as b_err:
                    logger.warning(f"Could not list S3 buckets: {b_err}")

                logger.info(f"Initialized Railway S3 Bucket client for '{self.bucket_name or 'Default'}'.")
            except Exception as e:
                logger.error(f"Failed to initialize S3 Bucket client: {e}")
                self.s3_client = None
        else:
            logger.info("Operating with Railway volume storage. Copy S3 keys from the 'Credentials' tab to upload into 'recorded-case' S3 Bucket.")

    def save_file(self, filename: str, content: bytes, content_type: str = "application/pdf") -> Dict[str, Any]:
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
                logger.error(f"S3 Upload failed for '{filename}': {e}. Please copy your Access Key & Secret Key from the 'Credentials' tab into Railway App Variables.")

        # Save copy to primary volume storage
        p_path = PRIMARY_BUCKET_DIR / filename
        with open(p_path, "wb") as f:
            f.write(content)

        # Save copy to backup volume storage
        b_path = BACKUP_BUCKET_DIR / filename
        with open(b_path, "wb") as f:
            f.write(content)

        return {
            "storage_type": f"Railway S3 Bucket ({self.bucket_name})" if s3_uploaded else f"Railway Persistent Volume ({PRIMARY_BUCKET_DIR})",
            "bucket_name": self.bucket_name or "railway-volume",
            "filename": filename,
            "size_bytes": len(content),
            "s3_uploaded": s3_uploaded,
            "saved_path": str(p_path)
        }

    def get_file(self, filename: str) -> Optional[bytes]:
        if self.s3_client and self.bucket_name:
            try:
                resp = self.s3_client.get_object(Bucket=self.bucket_name, Key=filename)
                return resp["Body"].read()
            except Exception as e:
                logger.warning(f"Error reading '{filename}' from S3 Bucket: {e}")

        for dir_path in [PRIMARY_BUCKET_DIR, BACKUP_BUCKET_DIR]:
            f_path = dir_path / filename
            if f_path.exists():
                with open(f_path, "rb") as f:
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
            except Exception as e:
                logger.warning(f"Error listing S3 Bucket files: {e}")

        # Check local volume directories
        for dir_path in [PRIMARY_BUCKET_DIR, BACKUP_BUCKET_DIR]:
            for f in dir_path.glob("*.pdf"):
                if not any(x["filename"] == f.name for x in files):
                    stat = f.stat()
                    files.append({
                        "filename": f.name,
                        "size_bytes": stat.st_size,
                        "last_modified": str(stat.st_mtime),
                        "storage_type": f"Railway Volume ({dir_path})"
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

        for dir_path in [PRIMARY_BUCKET_DIR, BACKUP_BUCKET_DIR]:
            f_path = dir_path / filename
            if f_path.exists():
                f_path.unlink()
                deleted = True

        return deleted
