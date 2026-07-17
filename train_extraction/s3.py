"""S3 client wrapper used across the pipeline."""
from __future__ import annotations

import logging
import os
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from .url_utils import split_bucket_prefix


class S3Client:
    """Thin wrapper over boto3.client('s3') with bucket/prefix conventions."""

    def __init__(
        self,
        region: str,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.region = region
        self.logger = logger or logging.getLogger(__name__)

        if aws_access_key and aws_secret_key:
            self.client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region,
            )
            self.logger.info("S3 client initialised with provided credentials.")
        else:
            self.client = boto3.client("s3", region_name=region)
            self.logger.info("S3 client initialised with IAM role credentials.")

    # --- access checks ---------------------------------------------------

    def test_access(self, bucket_string: str) -> bool:
        bucket, _ = split_bucket_prefix(bucket_string)
        try:
            self.client.list_objects_v2(Bucket=bucket, MaxKeys=1)
            self.logger.info("S3 access OK: s3://%s", bucket)
            return True
        except ClientError as e:
            self.logger.error("S3 access failed for s3://%s: %s", bucket, e)
            raise

    # --- listing ---------------------------------------------------------

    def list_objects(self, bucket_string: str, prefix: str = "") -> list[dict]:
        bucket, _ = split_bucket_prefix(bucket_string)
        results: list[dict] = []
        token = None
        while True:
            params = {"Bucket": bucket, "Prefix": prefix}
            if token:
                params["ContinuationToken"] = token
            resp = self.client.list_objects_v2(**params)
            if "Contents" in resp:
                results.extend(resp["Contents"])
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
            else:
                break
        return results

    def list_common_prefixes(self, bucket: str, prefix: str) -> list[str]:
        prefixes: list[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                prefixes.append(cp["Prefix"])
        return prefixes

    # --- download / upload ---------------------------------------------

    def download_file(self, bucket_string: str, key: str, local_path: str) -> str:
        bucket, _ = split_bucket_prefix(bucket_string)
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        self.client.download_file(bucket, key, local_path)
        return local_path

    def upload_file(self, local_path: str, bucket_string: str, key: str) -> str:
        """Upload to bucket. If bucket_string contains a prefix, it is prepended to key.

        Returns the full key (with prefix) for downstream URL generation.
        """
        bucket, prefix = split_bucket_prefix(bucket_string)
        full_key = f"{prefix}/{key}" if prefix else key
        self.client.upload_file(local_path, bucket, full_key)
        return full_key

    def head_object(self, bucket_string: str, key: str) -> dict:
        bucket, _ = split_bucket_prefix(bucket_string)
        return self.client.head_object(Bucket=bucket, Key=key)

    def object_exists(self, bucket_string: str, key: str) -> bool:
        try:
            self.head_object(bucket_string, key)
            return True
        except ClientError:
            return False
