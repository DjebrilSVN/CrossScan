"""
Interactive Clinical Gallery & Radiologist Portfolio Generator Script.

Produces standalone HTML5 clinical dashboard (output/clinical_demo_gallery.html)
with 3D slice navigation, dual-modality overlays, entropy ribbons, and PACS links.
"""

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console

from crossscan.visualization.gallery import export_clinical_gallery

app = typer.Typer(help="CrossScan Clinical Gallery Generator")
console = Console()


@app.command()
def generate(
    output_path: Path = typer.Option(Path("./output/clinical_demo_gallery.html"), "--out", "-o", help="Output HTML file path"),
    open_browser: bool = typer.Option(False, "--open", help="Open gallery in web browser upon creation")
):
    """Generates the interactive clinical demo gallery HTML report."""
    console.rule("[bold cyan]Generating Interactive Clinical Demo Gallery[/bold cyan]")
    
    out_p = export_clinical_gallery(output_html_path=output_path)
    file_size_kb = out_p.stat().st_size / 1024
    
    console.print(f"[bold green]Clinical portfolio gallery successfully generated![/bold green]")
    console.print(f"File Path: [bold cyan]{out_p.resolve()}[/bold cyan] ({file_size_kb:.1f} KB)")
    console.print("\n[bold]Features Included:[/bold]")
    console.print("• Dual-Modality 3D slice navigation (CT & MRI)")
    console.print("• Multi-organ SNOMED CT / RadLex contour overlays")
    console.print("• Voxel Shannon entropy uncertainty ribbons")
    console.print("• Conformal Risk Control triage status (AUTO_ACCEPT / FLAG_FOR_REVIEW / REJECT)")
    console.print("• Direct launch hooks for OHIF Web Viewer & 3D Slicer CLI")
    
    if open_browser:
        import webbrowser
        console.print(f"Opening gallery in default browser: {out_p.as_uri()}")
        webbrowser.open(out_p.as_uri())


if __name__ == "__main__":
    app()

