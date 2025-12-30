# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
SharePoint Integration for MJ Data Scraper Suite

Provides SharePoint connectivity for:
- Document storage and retrieval
- List data synchronization
- Scrape result archival
- Report generation and upload
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional, BinaryIO
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import json

logger = logging.getLogger(__name__)

# Optional imports - graceful degradation if not installed
try:
    from office365.runtime.auth.client_credential import ClientCredential
    from office365.sharepoint.client_context import ClientContext
    from office365.sharepoint.files.file import File
    from office365.sharepoint.listitems.listitem import ListItem
    SHAREPOINT_AVAILABLE = True
except ImportError:
    SHAREPOINT_AVAILABLE = False
    logger.warning("SharePoint SDK not installed. Install with: pip install Office365-REST-Python-Client")


@dataclass
class SharePointConfig:
    """SharePoint connection configuration."""
    site_url: str
    client_id: str
    client_secret: str
    tenant_id: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "SharePointConfig":
        """Create config from environment variables."""
        return cls(
            site_url=os.getenv("SHAREPOINT_SITE_URL", ""),
            client_id=os.getenv("SHAREPOINT_CLIENT_ID", ""),
            client_secret=os.getenv("SHAREPOINT_CLIENT_SECRET", ""),
            tenant_id=os.getenv("SHAREPOINT_TENANT_ID"),
        )


@dataclass
class SharePointFile:
    """Represents a SharePoint file."""
    name: str
    server_relative_url: str
    size: int
    created: datetime
    modified: datetime
    content_type: str


@dataclass
class SharePointListItem:
    """Represents a SharePoint list item."""
    id: int
    title: str
    fields: Dict[str, Any]
    created: datetime
    modified: datetime


class SharePointClient:
    """
    Client for SharePoint operations.
    
    Features:
    - File upload/download
    - List operations
    - Folder management
    - Search functionality
    """

    def __init__(self, config: Optional[SharePointConfig] = None):
        """
        Initialize SharePoint client.

        Args:
            config: SharePoint configuration (uses env vars if not provided)
        """
        if not SHAREPOINT_AVAILABLE:
            raise ImportError("SharePoint SDK not installed")
        
        self.config = config or SharePointConfig.from_env()
        self._ctx: Optional[ClientContext] = None
        self._connected = False

    def connect(self) -> bool:
        """
        Establish connection to SharePoint.

        Returns:
            True if connected successfully
        """
        try:
            credentials = ClientCredential(
                self.config.client_id,
                self.config.client_secret
            )
            self._ctx = ClientContext(self.config.site_url).with_credentials(credentials)
            
            # Test connection
            web = self._ctx.web
            self._ctx.load(web)
            self._ctx.execute_query()
            
            self._connected = True
            logger.info(f"Connected to SharePoint: {web.properties['Title']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to SharePoint: {e}")
            self._connected = False
            return False

    def _ensure_connected(self) -> None:
        """Ensure client is connected."""
        if not self._connected:
            if not self.connect():
                raise ConnectionError("Not connected to SharePoint")

    # File Operations

    def upload_file(
        self,
        local_path: str,
        remote_folder: str,
        remote_name: Optional[str] = None
    ) -> Optional[SharePointFile]:
        """
        Upload a file to SharePoint.

        Args:
            local_path: Path to local file
            remote_folder: SharePoint folder path (e.g., "Shared Documents/Scraper")
            remote_name: Optional name for remote file

        Returns:
            SharePointFile if successful
        """
        self._ensure_connected()
        
        try:
            with open(local_path, "rb") as f:
                content = f.read()
            
            file_name = remote_name or os.path.basename(local_path)
            target_folder = self._ctx.web.get_folder_by_server_relative_url(remote_folder)
            
            uploaded_file = target_folder.upload_file(file_name, content).execute_query()
            
            logger.info(f"Uploaded file: {file_name} to {remote_folder}")
            
            return SharePointFile(
                name=file_name,
                server_relative_url=uploaded_file.serverRelativeUrl,
                size=len(content),
                created=datetime.utcnow(),
                modified=datetime.utcnow(),
                content_type="application/octet-stream"
            )
            
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            return None

    def upload_content(
        self,
        content: bytes,
        remote_folder: str,
        file_name: str,
        content_type: str = "application/octet-stream"
    ) -> Optional[SharePointFile]:
        """
        Upload content directly to SharePoint.

        Args:
            content: File content as bytes
            remote_folder: SharePoint folder path
            file_name: Name for the file
            content_type: MIME type

        Returns:
            SharePointFile if successful
        """
        self._ensure_connected()
        
        try:
            target_folder = self._ctx.web.get_folder_by_server_relative_url(remote_folder)
            uploaded_file = target_folder.upload_file(file_name, content).execute_query()
            
            logger.info(f"Uploaded content as: {file_name}")
            
            return SharePointFile(
                name=file_name,
                server_relative_url=uploaded_file.serverRelativeUrl,
                size=len(content),
                created=datetime.utcnow(),
                modified=datetime.utcnow(),
                content_type=content_type
            )
            
        except Exception as e:
            logger.error(f"Failed to upload content: {e}")
            return None

    def download_file(self, server_relative_url: str) -> Optional[bytes]:
        """
        Download a file from SharePoint.

        Args:
            server_relative_url: Server-relative URL of the file

        Returns:
            File content as bytes
        """
        self._ensure_connected()
        
        try:
            file = self._ctx.web.get_file_by_server_relative_url(server_relative_url)
            content = file.read().execute_query()
            
            logger.info(f"Downloaded file: {server_relative_url}")
            return content.value
            
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            return None

    def list_files(self, folder_path: str) -> List[SharePointFile]:
        """
        List files in a SharePoint folder.

        Args:
            folder_path: Server-relative folder path

        Returns:
            List of SharePointFile objects
        """
        self._ensure_connected()
        
        try:
            folder = self._ctx.web.get_folder_by_server_relative_url(folder_path)
            files = folder.files
            self._ctx.load(files)
            self._ctx.execute_query()
            
            result = []
            for f in files:
                result.append(SharePointFile(
                    name=f.properties.get("Name", ""),
                    server_relative_url=f.properties.get("ServerRelativeUrl", ""),
                    size=f.properties.get("Length", 0),
                    created=datetime.fromisoformat(f.properties.get("TimeCreated", "").replace("Z", "+00:00")) if f.properties.get("TimeCreated") else datetime.utcnow(),
                    modified=datetime.fromisoformat(f.properties.get("TimeLastModified", "").replace("Z", "+00:00")) if f.properties.get("TimeLastModified") else datetime.utcnow(),
                    content_type=f.properties.get("ContentType", {}).get("Name", "unknown")
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []

    def create_folder(self, folder_path: str) -> bool:
        """
        Create a folder in SharePoint.

        Args:
            folder_path: Path for the new folder

        Returns:
            True if created successfully
        """
        self._ensure_connected()
        
        try:
            self._ctx.web.folders.add(folder_path).execute_query()
            logger.info(f"Created folder: {folder_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create folder: {e}")
            return False

    def delete_file(self, server_relative_url: str) -> bool:
        """
        Delete a file from SharePoint.

        Args:
            server_relative_url: Server-relative URL of the file

        Returns:
            True if deleted successfully
        """
        self._ensure_connected()
        
        try:
            file = self._ctx.web.get_file_by_server_relative_url(server_relative_url)
            file.delete_object().execute_query()
            logger.info(f"Deleted file: {server_relative_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            return False

    # List Operations

    def get_list_items(
        self,
        list_name: str,
        query: Optional[str] = None,
        limit: int = 100
    ) -> List[SharePointListItem]:
        """
        Get items from a SharePoint list.

        Args:
            list_name: Name of the list
            query: Optional CAML query filter
            limit: Maximum items to return

        Returns:
            List of SharePointListItem objects
        """
        self._ensure_connected()
        
        try:
            sp_list = self._ctx.web.lists.get_by_title(list_name)
            items = sp_list.items.get().top(limit).execute_query()
            
            result = []
            for item in items:
                result.append(SharePointListItem(
                    id=item.properties.get("Id", 0),
                    title=item.properties.get("Title", ""),
                    fields=dict(item.properties),
                    created=datetime.fromisoformat(item.properties.get("Created", "").replace("Z", "+00:00")) if item.properties.get("Created") else datetime.utcnow(),
                    modified=datetime.fromisoformat(item.properties.get("Modified", "").replace("Z", "+00:00")) if item.properties.get("Modified") else datetime.utcnow()
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get list items: {e}")
            return []

    def add_list_item(
        self,
        list_name: str,
        fields: Dict[str, Any]
    ) -> Optional[SharePointListItem]:
        """
        Add an item to a SharePoint list.

        Args:
            list_name: Name of the list
            fields: Field values for the new item

        Returns:
            Created SharePointListItem
        """
        self._ensure_connected()
        
        try:
            sp_list = self._ctx.web.lists.get_by_title(list_name)
            item = sp_list.add_item(fields).execute_query()
            
            logger.info(f"Added item to list: {list_name}")
            
            return SharePointListItem(
                id=item.properties.get("Id", 0),
                title=fields.get("Title", ""),
                fields=fields,
                created=datetime.utcnow(),
                modified=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to add list item: {e}")
            return None

    def update_list_item(
        self,
        list_name: str,
        item_id: int,
        fields: Dict[str, Any]
    ) -> bool:
        """
        Update a SharePoint list item.

        Args:
            list_name: Name of the list
            item_id: ID of the item to update
            fields: Field values to update

        Returns:
            True if updated successfully
        """
        self._ensure_connected()
        
        try:
            sp_list = self._ctx.web.lists.get_by_title(list_name)
            item = sp_list.get_item_by_id(item_id)
            item.set_property("Title", fields.get("Title", item.properties.get("Title")))
            for key, value in fields.items():
                item.set_property(key, value)
            item.update().execute_query()
            
            logger.info(f"Updated item {item_id} in list: {list_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update list item: {e}")
            return False

    def delete_list_item(self, list_name: str, item_id: int) -> bool:
        """
        Delete a SharePoint list item.

        Args:
            list_name: Name of the list
            item_id: ID of the item to delete

        Returns:
            True if deleted successfully
        """
        self._ensure_connected()
        
        try:
            sp_list = self._ctx.web.lists.get_by_title(list_name)
            item = sp_list.get_item_by_id(item_id)
            item.delete_object().execute_query()
            
            logger.info(f"Deleted item {item_id} from list: {list_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete list item: {e}")
            return False

    # Scraper-specific operations

    def archive_scrape_result(
        self,
        job_id: str,
        scraper_type: str,
        data: Dict[str, Any],
        folder: str = "Shared Documents/Scraper/Results"
    ) -> Optional[SharePointFile]:
        """
        Archive a scrape result to SharePoint.

        Args:
            job_id: Scrape job ID
            scraper_type: Type of scraper used
            data: Scraped data
            folder: Target folder

        Returns:
            SharePointFile if successful
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_name = f"{scraper_type}_{job_id}_{timestamp}.json"
        
        content = json.dumps({
            "job_id": job_id,
            "scraper_type": scraper_type,
            "scraped_at": datetime.utcnow().isoformat(),
            "data": data
        }, indent=2).encode("utf-8")
        
        return self.upload_content(
            content=content,
            remote_folder=folder,
            file_name=file_name,
            content_type="application/json"
        )

    def log_scrape_job(
        self,
        list_name: str,
        job_id: str,
        scraper_type: str,
        url: str,
        status: str,
        records_count: int,
        error: Optional[str] = None
    ) -> Optional[SharePointListItem]:
        """
        Log a scrape job to a SharePoint list.

        Args:
            list_name: Name of the tracking list
            job_id: Job ID
            scraper_type: Scraper type
            url: Target URL
            status: Job status
            records_count: Number of records scraped
            error: Error message if failed

        Returns:
            Created list item
        """
        return self.add_list_item(
            list_name=list_name,
            fields={
                "Title": job_id,
                "ScraperType": scraper_type,
                "TargetURL": url,
                "Status": status,
                "RecordsCount": records_count,
                "ErrorMessage": error or "",
                "CompletedAt": datetime.utcnow().isoformat()
            }
        )


class SharePointIntegration:
    """
    High-level SharePoint integration for the scraper suite.
    
    Provides simplified interface for common operations.
    """

    def __init__(self, config: Optional[SharePointConfig] = None):
        """Initialize integration."""
        self.config = config
        self._client: Optional[SharePointClient] = None
        self._enabled = SHAREPOINT_AVAILABLE and bool(
            (config and config.site_url) or os.getenv("SHAREPOINT_SITE_URL")
        )

    @property
    def enabled(self) -> bool:
        """Check if SharePoint integration is enabled."""
        return self._enabled

    def _get_client(self) -> SharePointClient:
        """Get or create SharePoint client."""
        if not self._enabled:
            raise RuntimeError("SharePoint integration is not enabled")
        
        if self._client is None:
            self._client = SharePointClient(self.config)
            self._client.connect()
        
        return self._client

    async def archive_results(
        self,
        job_id: str,
        scraper_type: str,
        results: List[Dict[str, Any]]
    ) -> bool:
        """
        Archive scrape results to SharePoint.

        Args:
            job_id: Job ID
            scraper_type: Scraper type
            results: List of scraped records

        Returns:
            True if archived successfully
        """
        if not self._enabled:
            logger.debug("SharePoint not enabled, skipping archive")
            return False
        
        try:
            client = self._get_client()
            
            # Archive as JSON file
            file = client.archive_scrape_result(
                job_id=job_id,
                scraper_type=scraper_type,
                data={"records": results, "count": len(results)}
            )
            
            return file is not None
            
        except Exception as e:
            logger.error(f"Failed to archive results: {e}")
            return False

    async def log_job(
        self,
        job_id: str,
        scraper_type: str,
        url: str,
        success: bool,
        records_count: int = 0,
        error: Optional[str] = None
    ) -> bool:
        """
        Log job to SharePoint tracking list.

        Args:
            job_id: Job ID
            scraper_type: Scraper type
            url: Target URL
            success: Whether job succeeded
            records_count: Number of records
            error: Error message if failed

        Returns:
            True if logged successfully
        """
        if not self._enabled:
            return False
        
        try:
            client = self._get_client()
            
            item = client.log_scrape_job(
                list_name="Scraper Jobs",
                job_id=job_id,
                scraper_type=scraper_type,
                url=url,
                status="Completed" if success else "Failed",
                records_count=records_count,
                error=error
            )
            
            return item is not None
            
        except Exception as e:
            logger.error(f"Failed to log job: {e}")
            return False


# Global instance
sharepoint = SharePointIntegration()
