"""
MinIO Object Storage Manager
Handles file uploads/downloads for project images and audiobooks.
"""
import os
import io
import logging
from datetime import timedelta
from minio import Minio
from minio.error import S3Error

logger = logging.getLogger("AI-Router.MinIO")

# Buckets
BUCKET_IMAGES = "project-images"
BUCKET_AUDIO = "audiobooks"

class StorageManager:
    def __init__(self):
        # Credentials from environment (set in docker-compose.yml)
        self.client = Minio(
            endpoint=os.getenv("MINIO_ENDPOINT", "ai-minio:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "admin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin123"),
            secure=False  # HTTP, not HTTPS (internal Docker network)
        )
        self._ensure_buckets()

    def _ensure_buckets(self):
        """Create buckets if they don't exist (idempotent)."""
        for bucket in [BUCKET_IMAGES, BUCKET_AUDIO]:
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    logger.info(f"✅ Created bucket: {bucket}")
                else:
                    logger.info(f"🪣 Bucket exists: {bucket}")
            except Exception as e:
                logger.error(f"❌ Bucket init error: {repr(e)}")

    def upload_image(self, project_slug: str, filename: str, 
                     data: bytes, content_type: str = "image/png") -> dict:
        """Upload an image to project-images bucket."""
        # Object key: project-slug/filename.png
        object_name = f"{project_slug}/{filename}"
        try:
            self.client.put_object(
                bucket_name=BUCKET_IMAGES,
                object_name=object_name,
                data=io.BytesIO(data),
                length=len(data),
                content_type=content_type
            )
            logger.info(f"✅ Uploaded image: {object_name} ({len(data)} bytes)")
            return {
                "bucket": BUCKET_IMAGES,
                "object": object_name,
                "size": len(data),
                "url": self.get_presigned_url(BUCKET_IMAGES, object_name)
            }
        except S3Error as e:
            logger.error(f"❌ Upload error: {repr(e)}")
            return {"error": str(e)}

    def get_presigned_url(self, bucket: str, object_name: str, 
                          expires: int = 7) -> str:
        """Generate a temporary download URL (default: 7 days)."""
        try:
            url = self.client.presigned_get_object(
                bucket_name=bucket,
                object_name=object_name,
                expires=timedelta(days=expires)
            )
            return url
        except S3Error as e:
            logger.error(f"❌ Presigned URL error: {repr(e)}")
            return ""

    def list_objects(self, bucket: str, prefix: str = "") -> list:
        """List all objects in a bucket (optionally filtered by prefix/project)."""
        try:
            objects = self.client.list_objects(bucket, prefix=prefix, recursive=True)
            result = []
            for obj in objects:
                result.append({
                    "name": obj.object_name,
                    "size": obj.size,
                    "modified": obj.last_modified.isoformat() if obj.last_modified else None
                })
            return result
        except S3Error as e:
            logger.error(f"❌ List error: {repr(e)}")
            return []

    def delete_object(self, bucket: str, object_name: str) -> bool:
        """Delete a single object from a bucket."""
        try:
            self.client.remove_object(bucket, object_name)
            logger.info(f"🗑️ Deleted: {bucket}/{object_name}")
            return True
        except S3Error as e:
            logger.error(f"❌ Delete error: {repr(e)}")
            return False

# Singleton instance
storage = StorageManager()
