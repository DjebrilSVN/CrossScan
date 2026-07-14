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
        auth: Optional[Tuple[str, str]] = ("orthanc", "orthanc"),
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

    def create_stow_multipart_payload(self, dicom_bytes: bytes) -> Tuple[bytes, str]:
        """
        Creates standard multipart/related payload for STOW-RS transmission.
        Type: application/dicom
        """
        boundary = "----CrossScanDicomWebBoundary7MA4YWxkTrZu0gW"
        body = (
            f"--{boundary}\r\n"
            f"Content-Type: application/dicom\r\n\r\n"
        ).encode("utf-8") + dicom_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
        
        content_type = f'multipart/related; type="application/dicom"; boundary={boundary}'
        return body, content_type

    def stow_store_instance(
        self,
        dicom_file_path: Union[str, Path]
    ) -> Dict[str, Union[bool, str, int]]:
        """
        Transmits a DICOM or DICOM-SEG object to Orthanc via STOW-RS.
        """
        dicom_file_path = Path(dicom_file_path)
        if not dicom_file_path.exists():
            raise FileNotFoundError(f"DICOM file not found: {dicom_file_path}")
            
        dicom_bytes = dicom_file_path.read_bytes()
        body, content_type = self.create_stow_multipart_payload(dicom_bytes)
        url = f"{self.dicomweb_endpoint}/studies"
        
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", content_type)
        if self.auth:
            import base64
            creds = base64.b64encode(f"{self.auth[0]}:{self.auth[1]}".encode()).decode()
            req.add_header("Authorization", f"Basic {creds}")
            
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status_code = resp.status
                return {"success": True, "status_code": status_code, "url": url}
        except urllib.error.URLError as e:
            # Standalone/offline clinical simulation
            return {
                "success": False,
                "status_code": getattr(e, "code", 503),
                "error": f"Orthanc server unreachable at {self.base_url} (simulated offline PACS): {e}"
            }

    def qido_query_studies(self, patient_id: Optional[str] = None) -> List[Dict]:
        """Queries PACS archives for matching patient studies via QIDO-RS."""
        query_url = f"{self.dicomweb_endpoint}/studies"
        if patient_id:
            query_url += f"?PatientID={patient_id}"
            
        req = urllib.request.Request(query_url)
        req.add_header("Accept", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return []

    def get_ohif_viewer_url(self, study_instance_uid: str) -> str:
        """Returns direct web link to inspect segmentation in OHIF Viewer."""
        return f"{self.base_url}/ohif/viewer?StudyInstanceUIDs={study_instance_uid}"

    def get_slicer_command(self, dicom_dir: Union[str, Path], seg_file: Union[str, Path]) -> str:
        """Generates 3D Slicer CLI command to launch and overlay DICOM-SEG on background scan."""
        import os
        local_app = Path(os.environ.get("LOCALAPPDATA", "")) / "slicer.org"
        slicer_bin = "Slicer.exe"
        if local_app.exists():
            matches = sorted(list(local_app.glob("**/Slicer.exe")))
            if matches:
                slicer_bin = f'"{matches[-1]}"'
        return f'{slicer_bin} --load-dicom "{dicom_dir}" --load "{seg_file}"'

