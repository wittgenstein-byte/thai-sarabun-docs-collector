"""Thin adapter around the installed `autocrawler` package for Google Images collection."""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from train_ocr.common.manifest import write_jsonl
from train_ocr.common.settings import GOOGLE_BLOCKED_DOMAINS, GOOGLE_CRAWL_DIR

logger = logging.getLogger(__name__)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


class CaptchaBlockedError(RuntimeError):
    """Raised when Google serves a CAPTCHA challenge and fail_on_captcha=True."""

    def __init__(self, message: str, partial_links: list[str] | None = None) -> None:
        super().__init__(message)
        self.partial_links = list(partial_links or [])


def is_blocked_domain(url: str, blocked_domains: tuple[str, ...] = GOOGLE_BLOCKED_DOMAINS) -> bool:
    """Return True if the URL host matches or is a subdomain of any blocked domain."""
    if not url or not blocked_domains:
        return False
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    return any(host == domain.lower() or host.endswith(f".{domain.lower()}") for domain in blocked_domains)


def _click_first_result(
    driver: Any,
    xpath: str,
    timeout: float = 15.0,
    max_attempts: int = 2,
    fail_on_captcha: bool = False,
) -> None:
    """Click the first search result thumbnail with bounded retries and CAPTCHA checks."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    from autocrawler.collectors.base import highlight, is_captcha_page

    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        if fail_on_captcha and is_captcha_page(driver):
            raise CaptchaBlockedError(f"CAPTCHA detected before clicking first result at {driver.current_url}")
        try:
            element = WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((By.XPATH, xpath)))
            element.click()
            highlight(driver, element)
            return
        except Exception as exc:
            last_exc = exc
            if fail_on_captcha and is_captcha_page(driver):
                raise CaptchaBlockedError(f"CAPTCHA detected at {driver.current_url}") from exc
            if attempt + 1 < max_attempts:
                logger.warning("Click timed out (%s) - refreshing once...", xpath)
                driver.refresh()
                time.sleep(2)
    raise RuntimeError(f"Could not click first result ({xpath}) after {max_attempts} attempts") from last_exc


def collect_google_full_resilient(
    keyword: str,
    config: Any,
    blocked_domains: tuple[str, ...] = GOOGLE_BLOCKED_DOMAINS,
    fail_on_captcha: bool = False,
    offset: int = 0,
) -> list[str]:
    """Full-resolution Google Images collector using direct Next-button click with Keys.RIGHT fallback."""
    from selenium.common.exceptions import StaleElementReferenceException
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys

    from autocrawler.collectors.base import (
        build_driver,
        build_query_url,
        get_scroll_position,
        highlight,
        is_captcha_page,
        pick_proxy,
        remove_duplicates,
        wait_for_manual_captcha,
    )
    from autocrawler.sites import Site, face_query_param

    google_search_url = "https://www.google.com/search"
    driver = build_driver(
        no_gui=config.no_gui,
        proxy=pick_proxy(config.proxy_list),
        user_data_dir=Path(config.chrome_profile_dir).resolve(),
    )
    links: list[str] = []
    try:
        logger.info("[Full Resolution Mode] Keyword: %s", keyword)
        face_param = face_query_param(Site.GOOGLE) if config.face else ""
        url = build_query_url(google_search_url, "q", keyword, tbm="isch") + face_param
        driver.get(url)
        time.sleep(1.5)
        if is_captcha_page(driver):
            if fail_on_captcha:
                raise CaptchaBlockedError(f"CAPTCHA detected at {driver.current_url}", partial_links=[])
            wait_for_manual_captcha(driver, config.no_gui)

        _click_first_result(
            driver,
            '//div[@jsname="dTDiAc"]',
            timeout=15.0,
            max_attempts=2,
            fail_on_captcha=fail_on_captcha,
        )
        time.sleep(1)

        if offset > 0:
            logger.info("Fast-skipping first %d results...", offset)
            next_button_xpath = '//button[@jsname="OCpkoe"]'
            for _ in range(offset):
                if is_captcha_page(driver):
                    if fail_on_captcha:
                        raise CaptchaBlockedError(
                            f"CAPTCHA detected while skipping at {driver.current_url}",
                            partial_links=[],
                        )
                    wait_for_manual_captcha(driver, config.no_gui)

                next_buttons = []
                try:
                    next_buttons = driver.find_elements(By.XPATH, next_button_xpath)
                    disabled = bool(next_buttons) and (
                        next_buttons[0].get_attribute("disabled") is not None
                        or next_buttons[0].get_attribute("aria-disabled") == "true"
                    )
                except StaleElementReferenceException:
                    disabled = False

                if disabled:
                    logger.info("Reached end of results while skipping at %d", offset)
                    return []

                advanced = False
                try:
                    if next_buttons:
                        next_buttons[0].click()
                        advanced = True
                except Exception:
                    advanced = False
                if not advanced:
                    ActionChains(driver).send_keys(Keys.RIGHT).perform()
                time.sleep(0.2)

        body = driver.find_element(By.TAG_NAME, "body")
        if config.limit > 0:
            limit = max(0, config.limit - offset)
            if limit == 0:
                return []
        else:
            limit = 10000
        last_scroll = 0
        patience = 0
        max_patience = 30
        last_src: str | None = None
        disabled_since: float | None = None
        disabled_confirm_seconds = 15
        xpath = '//div[@jsname="figiqf"]//img[not(contains(@src,"gstatic.com"))]'
        next_button_xpath = '//button[@jsname="OCpkoe"]'

        while len(links) < limit:
            try:
                start = time.time()
                wait_timeout = 0.5 if disabled_since is not None else 2.5
                imgs: list = []
                src = None
                while time.time() - start < wait_timeout:
                    imgs = body.find_elements(By.XPATH, xpath)
                    if imgs:
                        candidate = imgs[-1].get_attribute("src")
                        if candidate and candidate != last_src:
                            src = candidate
                            break
                    time.sleep(0.1)

                if src:
                    last_src = src
                    highlight(driver, imgs[-1])
                    if not is_blocked_domain(src, blocked_domains) and src not in links:
                        links.append(src)
                        logger.info("%d: %s", len(links), src)
            except KeyboardInterrupt:
                break
            except StaleElementReferenceException:
                pass

            if is_captcha_page(driver):
                if fail_on_captcha:
                    raise CaptchaBlockedError(
                        f"CAPTCHA appeared mid-crawl at {driver.current_url}",
                        partial_links=remove_duplicates(links),
                    )
                wait_for_manual_captcha(driver, config.no_gui)

            next_buttons = []
            try:
                next_buttons = driver.find_elements(By.XPATH, next_button_xpath)
                disabled = bool(next_buttons) and (
                    next_buttons[0].get_attribute("disabled") is not None
                    or next_buttons[0].get_attribute("aria-disabled") == "true"
                )
            except StaleElementReferenceException:
                disabled = False

            if disabled:
                if disabled_since is None:
                    disabled_since = time.time()
                elif time.time() - disabled_since >= disabled_confirm_seconds:
                    break
            else:
                disabled_since = None

            scroll = get_scroll_position(driver)
            if scroll == last_scroll:
                patience += 1
            else:
                patience = 0
                last_scroll = scroll
            if patience >= max_patience:
                break

            advanced = False
            try:
                if next_buttons and not disabled:
                    next_buttons[0].click()
                    advanced = True
            except Exception:
                advanced = False
            if not advanced:
                ActionChains(driver).send_keys(Keys.RIGHT).perform()

        links = remove_duplicates(links)
        logger.info("Collect links done. Site: google_full, Keyword: %s, Total: %d", keyword, len(links))
        return links
    finally:
        driver.quit()


def _download_links_with_offset(
    keyword: str,
    links: list[str],
    download_path: Path,
    start_index: int = 0,
    max_count: int = 0,
) -> int:
    """Download image links starting filenames at `google_{start_index:04d}` without overwriting."""
    import requests
    from autocrawler.downloader import base64_to_object, get_extension_from_link, validate_image

    keyword_dir = download_path / keyword.replace('"', "")
    keyword_dir.mkdir(parents=True, exist_ok=True)

    target = max_count or len(links)
    success_count = 0

    for offset_idx, link in enumerate(links):
        if success_count >= target:
            break

        file_idx = start_index + offset_idx
        no_ext_path = keyword_dir / f"google_{file_idx:04d}"
        path: Path | None = None

        try:
            logger.info(
                "Downloading %s from google: %d / %d (idx=%04d)",
                keyword,
                success_count + 1,
                target,
                file_idx,
            )
            if link.startswith("data:image/jpeg;base64"):
                path = no_ext_path.with_suffix(".jpg")
                path.write_bytes(base64_to_object(link))
                ext = "jpg"
            elif link.startswith("data:image/png;base64"):
                path = no_ext_path.with_suffix(".png")
                path.write_bytes(base64_to_object(link))
                ext = "png"
            else:
                ext = get_extension_from_link(link)
                path = no_ext_path.with_suffix(f".{ext}")
                response = requests.get(link, stream=True, timeout=10)
                with path.open("wb") as file:
                    shutil.copyfileobj(response.raw, file)
                del response

            real_ext = validate_image(path)
            if real_ext is None:
                logger.warning("Unreadable file - %s", link)
                path.unlink(missing_ok=True)
                continue
            if real_ext != ext:
                renamed = path.with_suffix(f".{real_ext}")
                path.rename(renamed)
            success_count += 1
        except KeyboardInterrupt:
            break
        except Exception as exc:
            if path is not None:
                path.unlink(missing_ok=True)
            logger.warning("Download failed - %s", exc)
            continue

    return success_count


def _build_task_profile(base_profile: Path, temp_root: Path) -> Path:
    task_dir = temp_root / "chrome-profile"
    if base_profile.exists():
        ignore = shutil.ignore_patterns("Singleton*", "RunningChromeVersion")
        shutil.copytree(base_profile, task_dir, dirs_exist_ok=True, ignore=ignore)
    else:
        task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


def crawl_google_images(
    keywords_file: Path,
    download_dir: Path = GOOGLE_CRAWL_DIR / "raw",
    profile_dir: Path = GOOGLE_CRAWL_DIR / "chrome-profile",
    full_resolution: bool = True,
    no_gui: bool = False,
    limit: int = 20,
    offset: int = 0,
    proxy_list: list[str] | None = None,
    blocked_domains: tuple[str, ...] = GOOGLE_BLOCKED_DOMAINS,
    skip_warmup: bool = False,
    skip_existing: bool = True,
    fail_on_captcha: bool = False,
    keyword_delay: float = 3.5,
) -> Path:
    """Collect Google Images via installed `autocrawler` and emit a standardized manifest.jsonl."""
    from dataclasses import replace

    from selenium.common.exceptions import TimeoutException, WebDriverException

    from autocrawler.collectors.google import collect_google
    from autocrawler.config import CrawlConfig
    from autocrawler.crawler import warm_up_chrome_profile
    from autocrawler.downloader import download_images
    from autocrawler.files import get_keywords
    from autocrawler.sites import Site

    resolved_profile = Path(profile_dir).resolve()
    resolved_download = Path(download_dir).resolve()
    resolved_keywords = Path(keywords_file).resolve()
    resolved_download.mkdir(parents=True, exist_ok=True)

    base_config = CrawlConfig(
        keywords_file=resolved_keywords,
        download_path=resolved_download,
        chrome_profile_dir=resolved_profile,
        skip_already_exist=skip_existing,
        n_processes=1,
        do_google=True,
        do_naver=False,
        full_resolution=full_resolution,
        face=False,
        no_gui=no_gui,
        limit=limit,
        proxy_list=list(proxy_list or []),
    )

    if not no_gui and not skip_warmup:
        warm_up_chrome_profile(base_config)

    keywords = get_keywords(resolved_keywords)
    done_marker_name = "google_done" if offset == 0 else f"google_done_{offset}_{limit}"
    max_download_count = (max(0, limit - offset) if limit > 0 else 0) if offset > 0 else limit
    try:
        for idx, keyword in enumerate(keywords):
            keyword_dir = resolved_download / keyword.replace('"', "")
            done_marker = keyword_dir / done_marker_name
            if skip_existing and done_marker.exists():
                logger.info("Skipping completed keyword (%s): %s", done_marker_name, keyword)
                continue

            if idx > 0 and keyword_delay > 0:
                delay = random.uniform(keyword_delay * 0.7, keyword_delay * 1.3)
                logger.info("Sleeping %.1fs before keyword %d/%d: %s", delay, idx + 1, len(keywords), keyword)
                time.sleep(delay)

            try:
                with tempfile.TemporaryDirectory(prefix="train-ocr-chrome-") as tmp:
                    task_profile = _build_task_profile(resolved_profile, Path(tmp))
                    task_config = replace(base_config, chrome_profile_dir=task_profile)
                    if full_resolution:
                        links = collect_google_full_resilient(
                            keyword,
                            task_config,
                            blocked_domains=blocked_domains,
                            fail_on_captcha=fail_on_captcha,
                            offset=offset,
                        )
                    else:
                        raw_links = [
                            link
                            for link in collect_google(keyword, task_config)
                            if not is_blocked_domain(link, blocked_domains)
                        ]
                        links = raw_links[offset:limit] if (offset > 0 and limit > 0) else raw_links[offset:]

                if offset > 0:
                    _download_links_with_offset(
                        keyword,
                        links,
                        resolved_download,
                        start_index=offset,
                        max_count=max_download_count,
                    )
                else:
                    download_images(keyword, links, Site.GOOGLE, resolved_download, max_count=limit)
                keyword_dir.mkdir(parents=True, exist_ok=True)
                done_marker.touch()
            except CaptchaBlockedError as exc:
                if exc.partial_links:
                    logger.info(
                        "Downloading %d partial links for '%s' before stopping...",
                        len(exc.partial_links),
                        keyword,
                    )
                    if offset > 0:
                        _download_links_with_offset(
                            keyword,
                            exc.partial_links,
                            resolved_download,
                            start_index=offset,
                            max_count=max_download_count,
                        )
                    else:
                        download_images(
                            keyword,
                            exc.partial_links,
                            Site.GOOGLE,
                            resolved_download,
                            max_count=limit,
                        )
                logger.error(
                    "CAPTCHA_BLOCKED: stopping remaining keywords cleanly (collected partial progress): %s",
                    exc,
                )
                break
            except (WebDriverException, TimeoutException, RuntimeError) as exc:
                logger.warning("Skipping keyword '%s' due to browser/network error: %s", keyword, exc)
                continue
    finally:
        records: list[dict] = []
        for img_path in sorted(resolved_download.rglob("*")):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            raw = img_path.read_bytes()
            records.append(
                {
                    "keyword": img_path.parent.name,
                    "filename": img_path.name,
                    "local_path": str(img_path.as_posix()),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size_bytes": len(raw),
                    "source_portal": "google_images",
                }
            )

        manifest_path = resolved_download.parent / "manifest.jsonl"
        write_jsonl(manifest_path, records)
        logger.info("Wrote Google Images manifest (%d files): %s", len(records), manifest_path)

    return manifest_path


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Collect Google Images using AutoCrawler adapter.")
    parser.add_argument("--keywords-file", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, default=GOOGLE_CRAWL_DIR / "raw")
    parser.add_argument("--profile-dir", type=Path, default=GOOGLE_CRAWL_DIR / "chrome-profile")
    parser.add_argument("--full", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-gui", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start index to skip already-collected results (e.g. --offset 40 --limit 100 collects 40..99).",
    )
    parser.add_argument("--proxy-list", type=str, default="")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-on-captcha", action="store_true")
    parser.add_argument("--keyword-delay", type=float, default=3.5)
    args = parser.parse_args(argv)

    proxies = [p.strip() for p in args.proxy_list.split(",") if p.strip()]
    crawl_google_images(
        keywords_file=args.keywords_file,
        download_dir=args.download_dir,
        profile_dir=args.profile_dir,
        full_resolution=args.full,
        no_gui=args.no_gui,
        limit=args.limit,
        offset=args.offset,
        proxy_list=proxies,
        skip_warmup=args.skip_warmup,
        skip_existing=args.skip_existing,
        fail_on_captcha=args.fail_on_captcha,
        keyword_delay=args.keyword_delay,
    )
