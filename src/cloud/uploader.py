"""
Azure Blob Storage uploader for Sieger teaching data.

Uploads captured images for a module (stain/uv/tail) to Azure Blob Storage
for cloud-based training pipeline.

Blob path structure:
    {container}/{customer_id}/{module}/{session_id}/
        metadata.json        ← site info, capture date, counts, config snapshot
        images/{filename}    ← captured PNG images

Usage:
    uploader = BlobUploader(cloud_config)
    result = uploader.upload_session(
        module="stain",
        session_id="abc-123",
        image_paths=[Path("captures/stain/.../img.png"), ...],
        metadata={"site": "ghcl", "n_images": 200, ...},
        progress_cb=lambda n, total: print(f"{n}/{total}"),
    )
"""

import json
import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class BlobUploader:
    """Uploads teaching session data to Azure Blob Storage.

    Args:
        config: The 'cloud' section of config.json.
            Required keys:
                account_name (str): Azure storage account name
                container (str): Blob container name
                customer_id (str): Customer identifier (used as blob prefix)
                sas_token (str): SAS token with write permission
    """

    def __init__(self, config: dict):
        self.account_name = config["account_name"]
        self.container = config["container"]
        self.customer_id = config["customer_id"]
        sas_token = config.get("sas_token", "")

        if not sas_token:
            raise ValueError(
                "cloud.sas_token is empty — generate a SAS token in Azure Portal "
                "with Blob write + create permissions and add it to config.json"
            )

        account_url = f"https://{self.account_name}.blob.core.windows.net"

        try:
            from azure.storage.blob import BlobServiceClient
            self._client = BlobServiceClient(
                account_url=account_url,
                credential=sas_token,
            )
            self._container_client = self._client.get_container_client(self.container)
        except ImportError:
            raise ImportError(
                "azure-storage-blob not installed. Run: uv add azure-storage-blob"
            )

        logger.info(
            "BlobUploader initialized: account=%s container=%s customer=%s",
            self.account_name, self.container, self.customer_id,
        )

    def _blob_prefix(self, module: str, session_id: str) -> str:
        """Return the blob prefix for this session."""
        return f"{self.customer_id}/{module}/{session_id}"

    def upload_session(
        self,
        module: str,
        session_id: str,
        image_paths: list,
        metadata: dict,
        data_root: Path,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> dict:
        """Upload all images + metadata for a teaching session.

        Args:
            module: Teaching module — 'stain', 'uv', 'tail'
            session_id: Capture session UUID
            image_paths: List of relative paths (relative to data_root)
            metadata: Dict written as metadata.json in the blob
            data_root: Local data root path
            progress_cb: Optional callback(n_uploaded, total) for progress reporting

        Returns:
            Dict with upload stats: n_uploaded, n_failed, blob_prefix, container
        """
        prefix = self._blob_prefix(module, session_id)
        n_uploaded = 0
        n_failed = 0
        total = len(image_paths)

        logger.info(
            "Starting blob upload: module=%s session=%s n_images=%d prefix=%s",
            module, session_id, total, prefix,
        )

        # Upload metadata.json first
        metadata_blob = f"{prefix}/metadata.json"
        try:
            blob_client = self._container_client.get_blob_client(metadata_blob)
            blob_client.upload_blob(
                json.dumps(metadata, indent=2).encode(),
                overwrite=True,
            )
            logger.info("Uploaded metadata.json → %s", metadata_blob)
        except Exception as e:
            logger.error("Failed to upload metadata.json: %s", e)
            raise RuntimeError(f"Metadata upload failed: {e}") from e

        # Upload images
        for i, rel_path in enumerate(image_paths):
            local_path = data_root / rel_path
            if not local_path.exists():
                logger.warning("Image not found, skipping: %s", local_path)
                n_failed += 1
                continue

            blob_name = f"{prefix}/images/{local_path.name}"
            try:
                blob_client = self._container_client.get_blob_client(blob_name)
                with open(local_path, "rb") as f:
                    blob_client.upload_blob(f, overwrite=True)
                n_uploaded += 1
                if progress_cb:
                    progress_cb(n_uploaded, total)
            except Exception as e:
                logger.warning("Failed to upload %s: %s", local_path.name, e)
                n_failed += 1

        logger.info(
            "Upload complete: module=%s session=%s uploaded=%d failed=%d",
            module, session_id, n_uploaded, n_failed,
        )

        return {
            "n_uploaded": n_uploaded,
            "n_failed": n_failed,
            "total": total,
            "blob_prefix": prefix,
            "container": self.container,
            "account": self.account_name,
            "metadata_blob": metadata_blob,
        }
