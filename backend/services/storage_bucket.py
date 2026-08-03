import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.config import DATA_DIR, BACKUP_DATA_DIR, get_clean_env

logger = logging.getLogger(__name__)

# Railway Bucket Environment Variable Resolver
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

PRIMARY_BUCKET_DIR = DATA_DIR / "bucket"
PRIMARY_BUCKET_DIR.mkdir(parents=True, exist_ok=True)

BACKUP_BUCKET_DIR = BACKUP_DATA_DIR / "bucket"
BACKUP_BUCKET_DIR.mkdir(parents=True, exist_ok=True)

USER_MAP_FILE = DATA_DIR / "user_files_map.json"

class StorageBucketManager:
    """
    Railway S3 Storage Bucket Manager with Strict Mandatory User Isolation.
    """

    def __init__(self):
        self.bucket_name = BUCKET_NAME
        self.s3_client = None
        self.user_file_map = self._load_user_file_map()

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
                except Exception as b_err:
                    logger.warning(f"Could not list S3 buckets: {b_err}")

                logger.info(f"Initialized User-Isolated S3 Storage Bucket client for '{self.bucket_name}'.")
            except Exception as e:
                logger.error(f"Failed to initialize S3 Bucket client: {e}")
                self.s3_client = None

    def _load_user_file_map(self) -> Dict[str, str]:
        if USER_MAP_FILE.exists():
            try:
                with open(USER_MAP_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_user_file_map(self):
        try:
            with open(USER_MAP_FILE, "w") as f:
                json.dump(self.user_file_map, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save user file map: {e}")

    def save_file(self, filename: str, content: bytes, user_id: str = "anonymous_user", content_type: str = "application/pdf") -> Dict[str, Any]:
        s3_uploaded = False
        safe_user_id = user_id if user_id and user_id.strip() else "anonymous_user"

        if self.s3_client and self.bucket_name:
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=filename,
                    Body=content,
                    ContentType=content_type,
                    Metadata={"user_id": safe_user_id}
                )
                logger.info(f"Successfully saved '{filename}' for user '{safe_user_id}' directly to S3 Storage Bucket '{self.bucket_name}'.")
                s3_uploaded = True
            except Exception as e:
                logger.error(f"S3 Upload error for '{filename}': {e}. Preserving copy on Railway Volume.")

        # Save local volume copies
        p_path = PRIMARY_BUCKET_DIR / filename
        with open(p_path, "wb") as f:
            f.write(content)

        b_path = BACKUP_BUCKET_DIR / filename
        with open(b_path, "wb") as f:
            f.write(content)

        # Record user ownership
        self.user_file_map[filename] = safe_user_id
        self._save_user_file_map()

        return {
            "storage_type": f"S3 Storage Bucket ({self.bucket_name})" if s3_uploaded else "Railway Volume Disk",
            "bucket_name": self.bucket_name,
            "filename": filename,
            "user_id": safe_user_id,
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

    def list_files(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lists files STRICTLY for the requested user_id. Returns empty if user_id is missing.
        """
        if not user_id or not user_id.strip():
            return []

        clean_uid = user_id.strip()
        files = []

        if self.s3_client and self.bucket_name:
            try:
                resp = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
                for obj in resp.get("Contents", []):
                    fn = obj["Key"]
                    owner = self.user_file_map.get(fn, "")

                    if owner == clean_uid:
                        files.append({
                            "filename": fn,
                            "user_id": owner,
                            "size_bytes": obj["Size"],
                            "last_modified": str(obj["LastModified"]),
                            "storage_type": f"S3 Storage Bucket ({self.bucket_name})"
                        })
            except Exception as e:
                logger.warning(f"Error listing S3 Bucket files: {e}")

        # Check local volume directories
        for dir_path in [PRIMARY_BUCKET_DIR, BACKUP_BUCKET_DIR]:
            for f in dir_path.glob("*.pdf"):
                fn = f.name
                owner = self.user_file_map.get(fn, "")
                if owner == clean_uid and not any(x["filename"] == fn for x in files):
                    stat = f.stat()
                    files.append({
                        "filename": fn,
                        "user_id": owner,
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

        if filename in self.user_file_map:
            del self.user_file_map[filename]
            self._save_user_file_map()

        return deleted
