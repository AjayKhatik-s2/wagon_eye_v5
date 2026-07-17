"""S3 URL helpers."""
from __future__ import annotations

from urllib.parse import quote


def split_bucket_prefix(bucket_string: str) -> tuple[str, str]:
    """Split 'bucket/optional/prefix' into (bucket, prefix)."""
    if "/" in bucket_string:
        bucket, prefix = bucket_string.split("/", 1)
        return bucket, prefix.rstrip("/")
    return bucket_string, ""


def s3_console_url(bucket: str, key: str, region: str) -> str:
    encoded_key = quote(key, safe="")
    return (
        f"https://s3.console.aws.amazon.com/s3/object/{bucket}"
        f"?region={region}&prefix={encoded_key}"
    )


def s3_object_url(bucket_string: str, key: str, region: str) -> str:
    """Direct https URL (path-style) for an S3 object."""
    bucket, _ = split_bucket_prefix(bucket_string)
    encoded_key = quote(key, safe="/")
    return f"https://{bucket}.s3.{region}.amazonaws.com/{encoded_key}"
