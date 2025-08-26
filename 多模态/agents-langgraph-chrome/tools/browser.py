from playwright.sync_api import sync_playwright
from PIL import Image
import io


def screenshot_page(url: str, width: int = 1280, height: int = 800) -> bytes:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(url, wait_until="networkidle")
        buf = page.screenshot(type="png", full_page=True)
        browser.close()
    return buf


def load_image(png_bytes: bytes):
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


