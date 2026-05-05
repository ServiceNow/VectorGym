"""SVG processing utilities."""

from PIL import Image
from bs4 import BeautifulSoup
from typing import Tuple, Optional
import re
from svgpathtools import svgstr2paths
import numpy as np
import cairosvg
from io import BytesIO
import xml.etree.ElementTree as ET
import cv2
import signal


# Constants
VOID_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"></svg>"""


def get_error_img(h: int = 512, w: int = 512) -> np.ndarray:
    """
    Returns a white image of shape (h, w, 3) with a red 'X' in the center.
    
    Args:
        h: Height
        w: Width
        
    Returns:
        Error image as numpy array
    """
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    thickness = max(1, h // 50)
    color = (255, 0, 0)  # Red in RGB
    center_x, center_y = w // 2, h // 2
    offset = min(h, w) // 5
    cv2.line(img, (center_x - offset, center_y - offset), (center_x + offset, center_y + offset), color, thickness)
    cv2.line(img, (center_x - offset, center_y + offset), (center_x + offset, center_y - offset), color, thickness)
    return img


def robust_svg_to_pil(
    svg_string: str,
    output_width: int = 512,
    output_height: int = 512,
    extract: bool = False,
    repair: bool = True,
    failed_fallback: str = "error"
) -> Tuple[Image.Image, str]:
    """
    Robustly convert SVG string to PIL Image with error handling.
    
    Args:
        svg_string: SVG string to convert
        output_width: Output image width
        output_height: Output image height
        extract: Whether to extract SVG from text
        repair: Whether to attempt SVG repair
        failed_fallback: Fallback type ("error", "white", "black")
        
    Returns:
        Tuple of (PIL Image, status string)
    """
    if extract and repair:
        raise ValueError("extract and repair cannot be True at the same time")
    
    # Construct fallback image
    if failed_fallback == "error":
        fallback_img = Image.fromarray(get_error_img(output_height, output_width))
    elif failed_fallback == "white":
        fallback_img = Image.new('RGB', (output_width, output_height), (255, 255, 255))
    elif failed_fallback == "black":
        fallback_img = Image.new('RGB', (output_width, output_height), (0, 0, 0))
    else:
        fallback_img = Image.fromarray(get_error_img(output_height, output_width))
    
    # Extract SVG from text if needed
    if extract:
        svg_string = extract_svg(svg_string)
    
    # Check SVG validity
    try:
        svgstr2paths(svg_string)
        svg_status = "valid"
    except Exception:
        svg_status = "invalid"
    
    # Repair SVG if invalid
    if svg_status == "invalid" and repair:
        svg_string = clean_svg(svg_string)
        try:
            svgstr2paths(svg_string)
            svg_status = "repaired"
        except Exception:
            svg_status = "invalid"
    
    if svg_status == "invalid":
        return fallback_img, svg_status
    
    # Rasterize SVG
    try:
        svg_raster_bytes = cairosvg.svg2png(
            bytestring=svg_string,
            background_color='white',
            output_width=output_width,
            output_height=output_height,
            dpi=128,
            scale=2
        )
        return Image.open(BytesIO(svg_raster_bytes)), svg_status
    except Exception:
        return fallback_img, "invalid"


def extract_svg(text: str) -> str:
    """
    Extract the last complete SVG block from text.
    
    Args:
        text: Text that may contain SVG snippets
        
    Returns:
        Last SVG string found, or empty string
    """
    pattern = r'<svg\b[^>]*>.*?</svg>'
    matches = list(re.finditer(pattern, text, flags=re.DOTALL | re.IGNORECASE))
    return matches[-1].group(0) if matches else ""


def clean_svg(svg_text: str, output_width: Optional[int] = None, output_height: Optional[int] = None) -> str:
    """
    Clean and repair SVG text using cairosvg.
    
    Args:
        svg_text: SVG string to clean
        output_width: Optional output width
        output_height: Optional output height
        
    Returns:
        Cleaned SVG string
    """
    soup = BeautifulSoup(svg_text, 'xml')
    svg_bs4 = soup.prettify()
    
    # Store original signal handler
    original_handler = signal.getsignal(signal.SIGALRM)
    
    try:
        # Set timeout to prevent hanging
        def timeout_handler(signum, frame):
            raise TimeoutError("SVG processing timed out")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)
        
        # Convert with cairosvg
        svg_cairo = cairosvg.svg2svg(
            svg_bs4,
            output_width=output_width,
            output_height=output_height
        ).decode()
        
    except TimeoutError:
        print("SVG conversion timed out, using fallback method")
        svg_cairo = use_placeholder()
    finally:
        # Always cancel alarm and restore handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)
    
    # Remove XML header
    svg_clean = "\n".join([
        line for line in svg_cairo.split("\n")
        if not line.strip().startswith("<?xml")
    ])
    
    return svg_clean


def use_placeholder() -> str:
    """Return a placeholder SVG."""
    return VOID_SVG


def rasterize_svg(
    svg_string: str,
    h: int = 224,
    w: int = 224,
    dpi: int = 128,
    scale: int = 2
) -> Image.Image:
    """
    Rasterize SVG string to PIL Image.
    
    Args:
        svg_string: SVG string to rasterize
        h: Output height
        w: Output width
        dpi: DPI for rendering
        scale: Scale factor
        
    Returns:
        PIL Image
    """
    try:
        # Try original
        svg_raster_bytes = cairosvg.svg2png(
            bytestring=svg_string,
            background_color='white',
            output_width=w,
            output_height=h,
            dpi=dpi,
            scale=scale
        )
        return Image.open(BytesIO(svg_raster_bytes))
    except Exception:
        try:
            # Try cleaned
            svg = clean_svg(svg_string)
            svg_raster_bytes = cairosvg.svg2png(
                bytestring=svg,
                background_color='white',
                output_width=w,
                output_height=h,
                dpi=dpi,
                scale=scale
            )
            return Image.open(BytesIO(svg_raster_bytes))
        except Exception:
            # Use placeholder
            svg = use_placeholder()
            svg_raster_bytes = cairosvg.svg2png(
                bytestring=svg,
                background_color='white',
                output_width=w,
                output_height=h,
                dpi=dpi,
                scale=scale
            )
            return Image.open(BytesIO(svg_raster_bytes))


def get_svg_original_size(svg_string: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse SVG string and return its original width and height.
    
    Args:
        svg_string: SVG string to parse
        
    Returns:
        Tuple of (width, height) or (None, None) if parsing fails
    """
    try:
        root = ET.fromstring(svg_string)
        
        # Try to get width and height attributes
        width = _parse_length(root.get("width"))
        height = _parse_length(root.get("height"))
        
        # If not defined, try viewBox
        if width is None or height is None:
            viewbox = root.get("viewBox")
            if viewbox:
                parts = viewbox.split()
                if len(parts) == 4:
                    width = float(parts[2])
                    height = float(parts[3])
        
        return width, height
    except Exception:
        return None, None


def _parse_length(length_str: Optional[str]) -> Optional[float]:
    """Parse an SVG length string and return a float."""
    if length_str is None:
        return None
    match = re.match(r"([0-9.]+)", length_str)
    if match:
        return float(match.group(1))
    return None


def is_valid_svg(svg_text: str) -> bool:
    """
    Check if SVG text is valid.
    
    Args:
        svg_text: SVG string to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        svgstr2paths(svg_text)
        return True
    except Exception:
        return False

