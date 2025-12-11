# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
News Scraper Plugin for MJ Data Scraper Suite

Scrapes news articles from RSS feeds and websites using intelligent content extraction.
Supports multiple sources, sentiment analysis, and article deduplication.
"""

import asyncio
import logging
import re
import hashlib
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree as ET

from core.base_scraper import BaseScraper, ScraperConfig

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__description__ = "News scraper with RSS feeds and HTML crawling capabilities"
__author__ = "MJ Intelligence"
__dependencies__ = ["requests", "beautifulsoup4", "lxml", "feedparser", "newspaper3k"]


class NewsScraperConfig(ScraperConfig):
    """Configuration specific to news scraping."""
    rss_feeds: List[str] = None  # List of RSS feed URLs
    news_sources: List[str] = None  # List of news website URLs
    max_articles_per_source: int = 50
    lookback_days: int = 7
    extract_full_content: bool = True
    extract_images: bool = True
    extract_authors: bool = True
    extract_publish_date: bool = True
    perform_sentiment_analysis: bool = False
    deduplicate_articles: bool = True
    min_article_length: int = 100
    max_article_length: int = 10000
    keywords_filter: List[str] = None  # Only articles containing these keywords
    categories: List[str] = ['business', 'technology', 'finance', 'politics']


class NewsScraper(BaseScraper):
    """
    Comprehensive news scraper supporting RSS feeds and web crawling.
    Includes article deduplication, content extraction, and metadata analysis.
    """

    def __init__(self, config: NewsScraperConfig):
        super().__init__(config)
        self.news_config = config

        # News scraping state
        self.article_cache: Dict[str, Dict[str, Any]] = {}
        self.processed_urls: Set[str] = set()
        self.rss_feed_stats: Dict[str, Dict[str, Any]] = {}
        self.source_stats: Dict[str, Dict[str, Any]] = {}

        # Content processing
        self.url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`[\]]+')
        self.duplicate_threshold = 0.85  # Similarity threshold for deduplication

        self.articles_scraped = 0

    async def _execute_scrape(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute news scraping from RSS feeds and websites.

        Args:
            target: Contains 'rss_feeds', 'sources', 'keywords', or general news parameters

        Returns:
            Comprehensive news articles data
        """
        rss_feeds = target.get('rss_feeds', self.news_config.rss_feeds or [])
        sources = target.get('sources', self.news_config.news_sources or [])
        keywords = target.get('keywords', self.news_config.keywords_filter or [])

        if not rss_feeds and not sources:
            raise ValueError("News scraper requires 'rss_feeds', 'sources', or default configuration")

        logger.info(f"Scraping news from {len(rss_feeds)} RSS feeds and {len(sources)} websites")

        # Scrape RSS feeds
        rss_articles = []
        if rss_feeds:
            rss_articles = await self._scrape_rss_feeds(rss_feeds, keywords)

        # Scrape websites
        web_articles = []
        if sources:
            web_articles = await self._scrape_news_websites(sources, keywords)

        # Combine and process articles
        all_articles = rss_articles + web_articles

        # Deduplicate if enabled
        if self.news_config.deduplicate_articles:
            all_articles = await self._deduplicate_articles(all_articles)

        # Filter by keywords if specified
        if keywords:
            all_articles = self._filter_by_keywords(all_articles, keywords)

        # Sort by publish date
        all_articles.sort(key=lambda x: x.get('publish_date', datetime.min), reverse=True)

        # Limit results
        max_total = self.news_config.max_articles_per_source * max(len(rss_feeds), len(sources), 1)
        all_articles = all_articles[:max_total]

        result = {
            "articles": all_articles,
            "total_articles": len(all_articles),
            "rss_feeds_processed": len(rss_feeds),
            "websites_processed": len(sources),
            "keywords_used": keywords,
            "scraped_at": datetime.utcnow().isoformat(),
            "stats": {
                "rss_articles": len(rss_articles),
                "web_articles": len(web_articles),
                "duplicates_removed": len(rss_articles) + len(web_articles) - len(all_articles),
                "sources": self._get_source_stats()
            }
        }

        self.articles_scraped += len(all_articles)
        return result

    async def _scrape_rss_feeds(self, feeds: List[str], keywords: List[str]) -> List[Dict[str, Any]]:
        """Scrape articles from RSS feeds."""
        all_articles = []

        for feed_url in feeds:
            try:
                logger.debug(f"Processing RSS feed: {feed_url}")
                feed_articles = await self._process_rss_feed(feed_url, keywords)
                all_articles.extend(feed_articles)

                # Update feed stats
                self.rss_feed_stats[feed_url] = {
                    'articles_found': len(feed_articles),
                    'last_updated': datetime.utcnow(),
                    'success': True
                }

            except Exception as e:
                logger.error(f"Failed to process RSS feed {feed_url}: {e}")
                self.rss_feed_stats[feed_url] = {
                    'error': str(e),
                    'last_updated': datetime.utcnow(),
                    'success': False
                }

        return all_articles

    async def _process_rss_feed(self, feed_url: str, keywords: List[str]) -> List[Dict[str, Any]]:
        """Process a single RSS feed."""
        try:
            import feedparser
            import aiohttp

            # Fetch RSS feed
            async with aiohttp.ClientSession() as session:
                async with session.get(feed_url, timeout=30) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}")

                    feed_content = await response.text()

            # Parse RSS feed
            feed = feedparser.parse(feed_content)

            articles = []
            cutoff_date = datetime.utcnow() - timedelta(days=self.news_config.lookback_days)

            for entry in feed.entries:
                try:
                    # Extract article data
                    article_data = self._extract_rss_article_data(entry)

                    # Check date
                    if article_data.get('publish_date') and article_data['publish_date'] < cutoff_date:
                        continue

                    # Check keywords
                    if keywords and not self._matches_keywords(article_data, keywords):
                        continue

                    # Get full content if configured
                    if self.news_config.extract_full_content and article_data.get('url'):
                        try:
                            full_content = await self._extract_full_article_content(article_data['url'])
                            article_data.update(full_content)
                        except Exception as e:
                            logger.debug(f"Could not extract full content for {article_data['url']}: {e}")

                    articles.append(article_data)

                    # Limit articles per feed
                    if len(articles) >= self.news_config.max_articles_per_source:
                        break

                except Exception as e:
                    logger.debug(f"Error processing RSS entry: {e}")
                    continue

            return articles

        except Exception as e:
            logger.error(f"Failed to process RSS feed {feed_url}: {e}")
            return []

    def _extract_rss_article_data(self, entry) -> Dict[str, Any]:
        """Extract article data from RSS entry."""
        # Handle different RSS formats
        title = getattr(entry, 'title', '')
        description = getattr(entry, 'description', '') or getattr(entry, 'summary', '')
        link = getattr(entry, 'link', '') or getattr(entry, 'url', '')

        # Extract publish date
        publish_date = None
        for date_field in ['published_parsed', 'updated_parsed', 'created_parsed']:
            if hasattr(entry, date_field) and getattr(entry, date_field):
                try:
                    import time
                    timestamp = time.mktime(getattr(entry, date_field))
                    publish_date = datetime.fromtimestamp(timestamp)
                    break
                except:
                    continue

        # Extract author
        author = getattr(entry, 'author', '') if self.news_config.extract_authors else None

        # Extract categories/tags
        categories = []
        if hasattr(entry, 'tags'):
            categories = [tag.term for tag in entry.tags if hasattr(tag, 'term')]

        return {
            "id": hashlib.md5(link.encode()).hexdigest()[:16] if link else None,
            "title": title,
            "description": self._clean_html(description),
            "url": link,
            "publish_date": publish_date,
            "author": author,
            "categories": categories,
            "source": "rss_feed",
            "content_type": "summary",
            "word_count": len((title + " " + description).split()) if title and description else 0
        }

    async def _scrape_news_websites(self, sources: List[str], keywords: List[str]) -> List[Dict[str, Any]]:
        """Scrape articles directly from news websites."""
        all_articles = []

        for source_url in sources:
            try:
                logger.debug(f"Scraping news website: {source_url}")
                source_articles = await self._scrape_news_source(source_url, keywords)
                all_articles.extend(source_articles)

                # Update source stats
                self.source_stats[source_url] = {
                    'articles_found': len(source_articles),
                    'last_updated': datetime.utcnow(),
                    'success': True
                }

            except Exception as e:
                logger.error(f"Failed to scrape news source {source_url}: {e}")
                self.source_stats[source_url] = {
                    'error': str(e),
                    'last_updated': datetime.utcnow(),
                    'success': False
                }

        return all_articles

    async def _scrape_news_source(self, source_url: str, keywords: List[str]) -> List[Dict[str, Any]]:
        """Scrape articles from a single news website."""
        try:
            from newspaper import Article, Config

            # Configure newspaper
            config = Config()
            config.browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            config.request_timeout = 10

            # Find article URLs on the source page
            article_urls = await self._find_article_urls(source_url)

            articles = []
            for article_url in article_urls[:self.news_config.max_articles_per_source]:
                try:
                    # Skip if already processed
                    if article_url in self.processed_urls:
                        continue

                    self.processed_urls.add(article_url)

                    # Download and parse article
                    article_data = await self._process_news_article(article_url, config)

                    # Check keywords
                    if keywords and not self._matches_keywords(article_data, keywords):
                        continue

                    # Check content length
                    content_length = len(article_data.get('content', ''))
                    if content_length < self.news_config.min_article_length:
                        continue
                    if content_length > self.news_config.max_article_length:
                        article_data['content'] = article_data['content'][:self.news_config.max_article_length] + "..."

                    articles.append(article_data)

                except Exception as e:
                    logger.debug(f"Error processing article {article_url}: {e}")
                    continue

            return articles

        except Exception as e:
            logger.error(f"Failed to scrape news source {source_url}: {e}")
            return []

    async def _find_article_urls(self, source_url: str) -> List[str]:
        """Find article URLs on a news source page."""
        try:
            import aiohttp
            from bs4 import BeautifulSoup

            async with aiohttp.ClientSession() as session:
                async with session.get(source_url, timeout=30) as response:
                    if response.status != 200:
                        return []

                    html = await response.text()

            soup = BeautifulSoup(html, 'lxml')

            # Common article URL patterns
            article_urls = []

            # Look for article links
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if not href:
                    continue

                # Convert relative URLs to absolute
                full_url = urljoin(source_url, href)

                # Check if it looks like an article URL
                if self._is_article_url(full_url, source_url):
                    article_urls.append(full_url)

            # Remove duplicates while preserving order
            seen = set()
            unique_urls = []
            for url in article_urls:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)

            return unique_urls[:50]  # Limit to prevent excessive crawling

        except Exception as e:
            logger.error(f"Error finding article URLs on {source_url}: {e}")
            return []

    def _is_article_url(self, url: str, source_domain: str) -> bool:
        """Determine if a URL likely points to a news article."""
        try:
            parsed = urlparse(url)

            # Must be HTTP/HTTPS
            if parsed.scheme not in ['http', 'https']:
                return False

            # Must be from same domain (or subdomain)
            url_domain = parsed.netloc.lower()
            source_domain_parsed = urlparse(source_domain).netloc.lower()

            if not (url_domain == source_domain_parsed or url_domain.endswith('.' + source_domain_parsed)):
                return False

            # Avoid common non-article URLs
            path = parsed.path.lower()
            if any(skip in path for skip in ['/tag/', '/category/', '/author/', '/search', '/page/', '/feed']):
                return False

            # Look for article-like patterns
            path_parts = [p for p in path.split('/') if p]
            if not path_parts:
                return False

            # Articles typically have date-like patterns or descriptive titles
            has_date_pattern = bool(re.search(r'\d{4}/\d{2}/\d{2}', path))
            has_title_like = len(path_parts[-1].split('-')) > 2  # Title with hyphens

            return has_date_pattern or has_title_like or len(path_parts[-1]) > 20

        except:
            return False

    async def _process_news_article(self, article_url: str, config) -> Dict[str, Any]:
        """Process a single news article."""
        try:
            from newspaper import Article

            # Create article object
            article = Article(article_url, config=config)

            # Download and parse
            article.download()
            article.parse()

            # Extract images if configured
            images = []
            if self.news_config.extract_images:
                article.nlp()
                images = article.images[:5] if article.images else []

            # Build article data
            article_data = {
                "id": hashlib.md5(article_url.encode()).hexdigest()[:16],
                "title": article.title,
                "content": article.text,
                "summary": article.summary if hasattr(article, 'summary') else None,
                "url": article_url,
                "publish_date": article.publish_date,
                "authors": article.authors if self.news_config.extract_authors else [],
                "keywords": article.keywords if hasattr(article, 'keywords') else [],
                "images": images,
                "source": urlparse(article_url).netloc,
                "content_type": "full_article",
                "word_count": len(article.text.split()) if article.text else 0,
                "language": getattr(article, 'meta_lang', 'en')
            }

            return article_data

        except Exception as e:
            logger.error(f"Failed to process article {article_url}: {e}")
            # Return basic data
            return {
                "id": hashlib.md5(article_url.encode()).hexdigest()[:16],
                "title": "Unknown",
                "url": article_url,
                "content": None,
                "source": urlparse(article_url).netloc,
                "error": str(e)
            }

    async def _extract_full_article_content(self, article_url: str) -> Dict[str, Any]:
        """Extract full content for RSS articles."""
        try:
            from newspaper import Article, Config

            config = Config()
            config.browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'

            article = Article(article_url, config=config)
            article.download()
            article.parse()

            return {
                "full_content": article.text,
                "authors": article.authors,
                "publish_date": article.publish_date,
                "images": list(article.images)[:3] if article.images else [],
                "word_count": len(article.text.split())
            }

        except Exception as e:
            logger.debug(f"Could not extract full content: {e}")
            return {}

    async def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on content similarity."""
        if not articles:
            return articles

        # Simple deduplication based on title and content similarity
        unique_articles = []

        for article in articles:
            is_duplicate = False

            for existing in unique_articles:
                if self._articles_similar(article, existing):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_articles.append(article)

        return unique_articles

    def _articles_similar(self, article1: Dict[str, Any], article2: Dict[str, Any]) -> bool:
        """Check if two articles are similar."""
        # Compare titles
        title1 = article1.get('title', '').lower().strip()
        title2 = article2.get('title', '').lower().strip()

        if title1 and title2:
            # Simple similarity check
            words1 = set(title1.split())
            words2 = set(title2.split())

            if words1 and words2:
                similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                if similarity > self.duplicate_threshold:
                    return True

        # Compare URLs (same article from different sources)
        url1 = article1.get('url', '')
        url2 = article2.get('url', '')
        if url1 and url2 and self._urls_point_to_same_article(url1, url2):
            return True

        return False

    def _urls_point_to_same_article(self, url1: str, url2: str) -> bool:
        """Check if two URLs point to the same article."""
        # Simple check based on URL similarity
        # In production, this could use more sophisticated methods
        return False

    def _filter_by_keywords(self, articles: List[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
        """Filter articles by keyword presence."""
        if not keywords:
            return articles

        filtered = []
        keywords_lower = [kw.lower() for kw in keywords]

        for article in articles:
            text_to_check = ""
            text_to_check += article.get('title', '') + " "
            text_to_check += article.get('description', '') + " "
            text_to_check += article.get('content', '') + " "
            text_to_check = text_to_check.lower()

            if any(keyword in text_to_check for keyword in keywords_lower):
                filtered.append(article)

        return filtered

    def _matches_keywords(self, article: Dict[str, Any], keywords: List[str]) -> bool:
        """Check if article matches any of the keywords."""
        return bool(self._filter_by_keywords([article], keywords))

    def _clean_html(self, html_text: str) -> str:
        """Clean HTML tags from text."""
        if not html_text:
            return ""

        # Remove HTML tags
        clean_text = re.sub(r'<[^>]+>', '', html_text)

        # Decode HTML entities
        import html
        clean_text = html.unescape(clean_text)

        # Clean up whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        return clean_text

    def _get_source_stats(self) -> Dict[str, Any]:
        """Get statistics about scraped sources."""
        return {
            'rss_feeds': len(self.rss_feed_stats),
            'websites': len(self.source_stats),
            'total_sources': len(self.rss_feed_stats) + len(self.source_stats)
        }

    async def _validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate news scraping result."""
        # Check basic structure
        if 'articles' not in result or 'total_articles' not in result:
            return False

        # Check if we got some articles or have valid error states
        articles = result.get('articles', [])
        stats = result.get('stats', {})

        has_articles = len(articles) > 0
        has_source_stats = stats.get('total_sources', 0) > 0

        return has_articles or has_source_stats

    def get_news_metrics(self) -> Dict[str, Any]:
        """Get news scraping specific metrics."""
        return {
            **self.get_metrics(),
            'articles_scraped': self.articles_scraped,
            'rss_feeds_tracked': len(self.rss_feed_stats),
            'websites_tracked': len(self.source_stats),
            'cache_size': len(self.article_cache),
            'features_enabled': {
                'full_content': self.news_config.extract_full_content,
                'images': self.news_config.extract_images,
                'authors': self.news_config.extract_authors,
                'sentiment': self.news_config.perform_sentiment_analysis,
                'deduplication': self.news_config.deduplicate_articles
            },
            'limits': {
                'max_articles_per_source': self.news_config.max_articles_per_source,
                'lookback_days': self.news_config.lookback_days,
                'min_article_length': self.news_config.min_article_length,
                'max_article_length': self.news_config.max_article_length
            }
        }

    async def cleanup(self) -> None:
        """Cleanup news scraper resources."""
        await super().cleanup()

        # Clear caches and state
        self.article_cache.clear()
        self.processed_urls.clear()
        self.rss_feed_stats.clear()
        self.source_stats.clear()
        self.articles_scraped = 0

        logger.info("News scraper cleaned up")