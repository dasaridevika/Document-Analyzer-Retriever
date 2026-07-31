import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Railway / S3 Storage Bucket Resolver
BUCKET_NAME = (
    os.getenv("RAILWAY_BUCKET_NAME", "").strip() or
    os.getenv("BUCKET_NAME", "").strip() or
    os.getenv("S3_BUCKET_NAME", "").strip() or
    os.getenv("AWS_STORAGE_BUCKET_NAME", "").strip() or
    "recorded-case-mw0hrll2-cc"
)

ACCESS_KEY_ID = (
    os.getenv("RAILWAY_ACCESS_KEY_ID", "").strip() or
    os.getenv("ACCESS_KEY_ID", "").strip() or
    os.getenv("S3_ACCESS_KEY_ID", "").strip() or
    os.getenv("AWS_ACCESS_KEY_ID", "").strip()
)

SECRET_ACCESS_KEY = (
    os.getenv("RAILWAY_SECRET_ACCESS_KEY", "").strip() or
    os.getenv("SECRET_ACCESS_KEY", "").strip() or
    os.getenv("S3_SECRET_ACCESS_KEY", "").strip() or
    os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
)

ENDPOINT_URL = (
    os.getenv("RAILWAY_ENDPOINT_URL", "").strip() or
    os.getenv("ENDPOINT_URL", "").strip() or
    os.getenv("S3_ENDPOINT_URL", "").strip() or
    os.getenv("AWS_ENDPOINT_URL_S3", "").strip() or
    os.getenv("AWS_ENDPOINT_URL", "").strip()
)

REGION_NAME = os.getenv("RAILWAY_REGION") or os.getenv("S3_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"

if ENDPOINT_URL and not ENDPOINT_URL.startswith("http://") and not ENDPOINT_URL.startswith("https://"):
    ENDPOINT_URL = f"https://{ENDPOINT_URL}"

class StorageBucketManager:
    """
    Exclusive S3 Storage Bucket Manager.
    Saves and manages uploaded documents strictly inside the S3 Storage Bucket.
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
                        logger.info(f"Auto-detected S3 Bucket name: '{self.bucket_name}'")
                except Exception as b_err:
                    logger.warning(f"Could not list S3 buckets: {b_err}")

                logger.info(f"Initialized Exclusive S3 Storage Bucket client for '{self.bucket_name}'.")
            except Exception as e:
                logger.error(f"Failed to initialize S3 Bucket client: {e}")
                self.s3_client = None
        else:
            logger.info(f"S3 Credentials pending. Set RAILWAY_ACCESS_KEY_ID, RAILWAY_SECRET_ACCESS_KEY, RAILWAY_ENDPOINT_URL in Railway Variables to write directly to S3 Bucket '{self.bucket_name}'.")

    def save_file(self, filename: str, content: bytes, content_type: str = "application/pdf") -> Dict[str, Any]:
        """
        Saves PDF file directly into the S3 Storage Bucket.
        """
        if self.s3_client and self.bucket_name:
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=filename,
                    Body=content,
                    ContentType=content_type
                )
                logger.info(f"Successfully saved '{filename}' exclusively to S3 Storage Bucket '{self.bucket_name}'.")
                return {
                    "storage_type": f"S3 Storage Bucket ({self.bucket_name})",
                    "bucket_name": self.bucket_name,
                    "filename": filename,
                    "size_bytes": len(content),
                    "s3_uploaded": True
                }
            except Exception as e:
                logger.error(f"S3 Bucket Upload error for '{filename}': {e}")
                raise RuntimeError(f"Failed to upload '{filename}' to S3 Storage Bucket: {str(e)}")

        raise RuntimeError("S3 Storage Bucket client is not configured. Please add RAILWAY_ACCESS_KEY_ID, RAILWAY_SECRET_ACCESS_KEY, and RAILWAY_ENDPOINT_URL in Railway Variables.")

    def get_file(self, filename: str) -> Optional[bytes]:
        """
        Retrieves PDF file bytes directly from the S3 Storage Bucket.
        """
        if self.s3_client and self.bucket_name:
            try:
                resp = self.s3_client.get_object(Bucket=self.bucket_name, Key=filename)
                return resp["Body"].read()
            except Exception as e:
                logger.error(f"Error reading '{filename}' from S3 Bucket: {e}")
                return None

        return None

    def list_files(self) -> List[Dict[str, Any]]:
        """
        Lists all PDF files stored in the S3 Storage Bucket.
        """
        files = []
        if self.s3_client and self.bucket_name:
            try:
                resp = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
                for obj in resp.get("Contents", []):
                    files.append({
                        "filename": obj["Key"],
                        "size_bytes": obj["Size"],
                        "last_modified": str(obj["LastModified"]),
                        "storage_type": f"S3 Storage Bucket ({self.bucket_name})"
                    })
                return files
            except Exception as e:
                logger.error(f"Error listing S3 Bucket files: {e}")

        return []

    def delete_file(self, filename: str) -> bool:
        """
        Deletes a PDF file directly from the S3 Storage Bucket.
        """
        if self.s3_client and self.bucket_name:
            try:
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=filename)
                logger.info(f"Deleted '{filename}' from S3 Storage Bucket '{self.bucket_name}'.")
                return True
            except Exception as e:
                logger.error(f"Failed to delete '{filename}' from S3 Bucket: {e}")
                return False

        return False
