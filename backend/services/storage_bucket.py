import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Helper to sanitize placeholder strings from env variables
def get_clean_env(key: str, default: str = "") -> str:
    val = os.getenv(key, default).strip()
    if not val or "your_" in val.lower() or "placeholder" in val.lower() or "xxx" in val.lower():
        return ""
    return val

# Comprehensive Railway Bucket Variable Resolver (Supporting Railway's exact names: BUCKET, ENDPOINT, ACCESS_KEY_ID, SECRET_ACCESS_KEY)
BUCKET_NAME = (
    get_clean_env("BUCKET") or
    get_clean_env("BUCKET_NAME") or
    get_clean_env("RAILWAY_BUCKET_NAME") or
    get_clean_env("S3_BUCKET_NAME") or
    get_clean_env("AWS_STORAGE_BUCKET_NAME") or
    "recorded-case-mw0hrll2-cc"
)

ACCESS_KEY_ID = (
    get_clean_env("ACCESS_KEY_ID") or
    get_clean_env("RAILWAY_ACCESS_KEY_ID") or
    get_clean_env("S3_ACCESS_KEY_ID") or
    get_clean_env("AWS_ACCESS_KEY_ID") or
    ""
)

SECRET_ACCESS_KEY = (
    get_clean_env("SECRET_ACCESS_KEY") or
    get_clean_env("RAILWAY_SECRET_ACCESS_KEY") or
    get_clean_env("S3_SECRET_ACCESS_KEY") or
    get_clean_env("AWS_SECRET_ACCESS_KEY") or
    ""
)

ENDPOINT_URL = (
    get_clean_env("ENDPOINT") or
    get_clean_env("ENDPOINT_URL") or
    get_clean_env("RAILWAY_ENDPOINT_URL") or
    get_clean_env("S3_ENDPOINT_URL") or
    get_clean_env("AWS_ENDPOINT_URL_S3") or
    get_clean_env("AWS_ENDPOINT_URL") or
    ""
)

REGION_NAME = get_clean_env("RAILWAY_REGION") or get_clean_env("S3_REGION") or "us-east-1"

if ENDPOINT_URL and not ENDPOINT_URL.startswith("http://") and not ENDPOINT_URL.startswith("https://"):
    ENDPOINT_URL = f"https://{ENDPOINT_URL}"

class StorageBucketManager:
    """
    Exclusive S3 Storage Bucket Manager.
    Directly connects using Railway's exact environment variables (BUCKET, ENDPOINT, ACCESS_KEY_ID, SECRET_ACCESS_KEY).
    """

    def __init__(self):
        self.bucket_name = BUCKET_NAME
        self.s3_client = None

        logger.info(f"Connecting to Railway Storage Bucket '{self.bucket_name}' via endpoint '{ENDPOINT_URL or 'Default'}'...")

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
                        logger.info(f"Auto-detected Railway Storage Bucket name: '{self.bucket_name}'")
                except Exception as b_err:
                    logger.warning(f"Could not list S3 buckets: {b_err}")

                logger.info(f"Successfully initialized S3 Storage Bucket client for '{self.bucket_name}'.")
            except Exception as e:
                logger.error(f"Failed to initialize S3 Bucket client: {e}")
                self.s3_client = None
        else:
            logger.warning("S3 Credentials pending. ACCESS_KEY_ID or SECRET_ACCESS_KEY is empty.")

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
                logger.info(f"Successfully saved '{filename}' directly to S3 Storage Bucket '{self.bucket_name}'.")
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

        raise RuntimeError("S3 Storage Bucket client is not configured. Check ACCESS_KEY_ID and SECRET_ACCESS_KEY.")

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
