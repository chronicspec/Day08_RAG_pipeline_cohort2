"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


import asyncio
import json
from datetime import datetime
from pathlib import Path
import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Danh sách URL bài báo cần crawl
ARTICLE_URLS = [
    "https://vtcnews.vn/ca-si-chi-dan-su-nghiep-truot-doc-2-lan-bi-nghi-lien-quan-ma-tuy-ar906502.html",
    "https://vtcnews.vn/ca-si-chi-dan-va-anh-trai-bi-de-nghi-truy-to-lien-quan-to-chuc-su-dung-ma-tuy-ar960946.html",
    "https://vtcnews.vn/truoc-khi-bi-bat-giam-vi-ma-tuy-chi-dan-thuong-khoe-cuoc-song-giau-sang-ar907306.html",
    "https://vtcnews.vn/ntk-cong-tri-tu-lao-dai-hang-dau-lang-thoi-trang-viet-toi-toi-pham-ma-tuy-ar955900.html",
    "https://vtcnews.vn/cong-tri-va-loat-sao-viet-tung-bi-bat-vi-ma-tuy-ar955936.html"
]



async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # Run request inside a thread pool to avoid blocking async loop
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(None, lambda: requests.get(url, headers=headers, timeout=15))
        response.raise_for_status()
        html_content = response.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        # Fallback empty data if fetch fails
        return {
            "url": url,
            "title": "Lỗi tải bài viết",
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": f"Không thể tải nội dung từ {url} do lỗi: {e}"
        }

    title = "Unknown"
    content_markdown = ""

    # Try beautifulsoup first
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")

        # Parse title
        title_tag = soup.find("h1") or soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else "Unknown"

        # Parse description / summary if present
        sapo_tag = soup.find("p", class_=lambda value: value and "description" in value)
        sapo = sapo_tag.get_text(strip=True) if sapo_tag else ""

        def extract_text_from_container(container):
            paragraphs = [p.get_text(strip=True) for p in container.find_all("p") if p.get_text(strip=True)]
            return "\n\n".join(paragraphs).strip()

        # Common containers for news article bodies
        selectors = [
            "div.content-wrapper",
            "div.edittor-content",
            "div.fck_detail",
            "div.detail-content",
            "div.article-content",
            "div.content",
            "article",
            "div.content-wrapper",
            "div.maincontent",
            "div.main-content",
            "div.box-cont",
        ]

        body_text = ""
        for selector in selectors:
            container = soup.select_one(selector)
            if container:
                body_text = extract_text_from_container(container)
                if len(body_text) > 50:
                    break

        if not body_text:
            # Fallback: search for candidate divs with article-like class names
            candidates = []
            for div in soup.find_all("div"):
                classes = div.get("class") or []
                classes = [c.lower() for c in classes if isinstance(c, str)]
                if any(keyword in cls for cls in classes for keyword in ["content", "detail", "article", "main", "body", "news"]):
                    candidates.append(div)

            for div in candidates:
                body_text = extract_text_from_container(div)
                if len(body_text) > 100:
                    break

        if not body_text:
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 30]
            body_text = "\n\n".join(paragraphs)

        content_markdown = f"{sapo}\n\n{body_text}".strip() if sapo else body_text
    except ImportError:
        # Fallback built-in HTMLParser
        from html.parser import HTMLParser
        class SimpleHTMLParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.in_title = False
                self.title = "Unknown"
                self.in_p = False
                self.paragraphs = []
            
            def handle_starttag(self, tag, attrs):
                if tag == "title":
                    self.in_title = True
                elif tag == "p":
                    self.in_p = True
            
            def handle_endtag(self, tag):
                if tag == "title":
                    self.in_title = False
                elif tag == "p":
                    self.in_p = False
            
            def handle_data(self, data):
                if self.in_title:
                    self.title = data.strip()
                elif self.in_p:
                    cleaned = data.strip()
                    if cleaned:
                        self.paragraphs.append(cleaned)

        parser = SimpleHTMLParser()
        parser.feed(html_content)
        title = parser.title
        content_markdown = "\n\n".join(parser.paragraphs)

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content_markdown,
    }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
