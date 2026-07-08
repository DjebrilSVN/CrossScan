"""
Clinical Orthanc PACS and DICOMweb REST Client.

Provides standard DICOMweb protocol support:
- STOW-RS (RESTful DICOM Store): Uploads DICOM-SEG and volumes to Orthanc PACS.
- WADO-RS (RESTful DICOM Retrieve): Fetches studies and metadata for visualization.
- QIDO-RS (RESTful DICOM Query): Searches PACS archives.
- Web Viewer Hooks: Generates direct links for OHIF Viewer and 3D Slicer.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import urllib.request
import urllib.error


class OrthancDICOMwebClient:
    """
    Standard DICOMweb client interfacing with an Orthanc PACS server.
    """
    def __init__(
        self,
        base_url: str = "http://localhost:8042",
        dicomweb_root: str = "/dicom-web",
        auth: Optional[Tuple[str, str]] = None,
        timeout_seconds: float = 5.0
    ):
        self.base_url = base_url.rstrip("/")
        self.dicomweb_endpoint = f"{self.base_url}{dicomweb_root}"
        self.auth = auth
        self.timeout = timeout_seconds

    def check_health(self) -> Dict[str, Union[bool, str]]:
        """Checks if Orthanc PACS server is reachable."""
        url = f"{self.base_url}/system"
        try:
            req = urllib.request.Request(url)
            if self.auth:
                import base64
                creds = base64.b64encode(f"{self.auth[0]}:{self.auth[1]}".encode()).decode()
                req.add_header("Authorization", f"Basic {creds}")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
                return {"is_online": True, "version": data.get("Version", "Unknown")}
        except Exception as e:
            return {"is_online": False, "version": f"Offline ({e})"}
