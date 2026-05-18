import logging
import re
import httpx
from bs4 import BeautifulSoup
import asyncio
import xml.etree.ElementTree as ET
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from yandex_ai_studio_sdk import AIStudio

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

BLOCKED_DOMAINS = {"vk", "t.me", "ok.ru", "dzen.ru"}
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/132.0.0.0 YaBrowser/25.2.0.0 Safari/537.36"
)

_QUERY_SYSTEM = (
    "Ты генерируешь поисковые запросы для верификации новости. "
    "Сформируй ровно {n} запросов — каждый с новой строки, пронумерованных. "
    "Запросы должны быть конкретными, без местоимений-ссылок на новость."
)


class YandexSearchService:
    def __init__(self):
        self._sdk = AIStudio(
            folder_id=settings.yandex_folder_id,
            auth=settings.yandex_auth,
        )

    def _search_sync(self, query: str,
               n: int = 5,
               seen_urls: set | None = None) -> list[dict]:
        if seen_urls is None:
            seen_urls = set()
        try:
            search = self._sdk.search_api.web(search_type="ru", user_agent=USER_AGENT)
            operation = search.run_deferred(query, format="xml", page=0)
            xml = operation.wait().decode("utf-8")
            return self._parse(xml, n, seen_urls)
        except Exception as e:
            logger.error(f"Yandex Search failed for query {query}: {e}")
            return []

    async def search(self, query: str, n: int = 5,
                     seen_urls: set | None = None) -> list[dict]:
        if seen_urls is None:
            seen_urls = set()
        results = await asyncio.to_thread(
            self._search_sync, query, n, seen_urls
        )
        # Обогащаем результаты полным текстом
        await self._enrich_content(results)
        return results

    async def _enrich_content(self, results: list[dict],
                              max_chars: int = 2000) -> None:
        """Параллельно загружает полные тексты для всех результатов."""
        if not results:
            return

        tasks = [
            self._fetch_full_text(r["url"], max_chars=max_chars)
            for r in results
        ]
        full_texts = await asyncio.gather(*tasks)

        for result, full_text in zip(results, full_texts):
            if full_text and len(full_text) > len(result.get("content", "")):
                result["content_snippet"] = result["content"]
                result["content"] = full_text

    async def multi_search(self, queries: list[str],
                           n_per_query: int = 5) -> list[dict]:
        seen = set()
        evidences = []
        for q in queries:
            batch = await self.search(q, n=n_per_query, seen_urls=seen)
            evidences.extend(batch)
        return evidences

    def _parse(self, xml_content: str, limit: int, seen: set) -> list[dict]:
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.warning("XML parse error: %s", e)
            return []

        results = []
        for doc in root.findall(".//doc"):
            url = self._t(doc.find("url"))
            domain = self._t(doc.find("domain"))
            if any(s in domain for s in BLOCKED_DOMAINS) or url in seen:
                continue
            seen.add(url)
            results.append({
                "title": self._t(doc.find("title")),
                "content": self._t(doc.find("passages")),
                "domain": domain,
                "url": url,
                "date_str": self._t(doc.find("modtime"))
                            or self._t(doc.find("pubdate"))
                            or "",
            })
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _t(el) -> str:
        return "".join(el.itertext()).strip() if el is not None else ""

    @staticmethod
    def _extract_main_text(html: str, max_chars: int = 2000) -> str:
        """Извлекает основной текст статьи из HTML."""
        soup = BeautifulSoup(html, "html.parser")

        # Удаляем навигацию, скрипты, стили, футеры
        for tag in soup.find_all(
                ["script", "style", "nav", "footer", "header",
                 "aside", "form", "iframe", "noscript"]
        ):
            tag.decompose()

        # Пытаемся найти основной контент по типичным тегам/классам
        main = (
                soup.find("article")
                or soup.find("main")
                or soup.find("div", class_=re.compile(
            r"article|content|body|text|post", re.I
        ))
        )
        container = main if main else soup.body if soup.body else soup

        # Собираем текст из абзацев
        paragraphs = []
        for p in container.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 30:
                paragraphs.append(text)

        full_text = " ".join(paragraphs)
        return full_text[:max_chars] if full_text else ""

    async def _fetch_full_text(self, url: str,
                               timeout: float = 5.0,
                               max_chars: int = 2000) -> str:
        """Загружает страницу и извлекает основной текст."""
        try:
            async with httpx.AsyncClient(
                    timeout=timeout,
                    follow_redirects=True,
                    verify=False,
                    headers={"User-Agent": USER_AGENT},
            ) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    return ""
                content_type = resp.headers.get("content-type", "")
                if "text/html" not in content_type:
                    return ""
                return self._extract_main_text(resp.text, max_chars)
        except Exception as e:
            logger.debug("Failed to fetch %s: %s", url, e)
            return ""


class GigaChatService:

    def __init__(self):
        self._giga = GigaChat(
            model=settings.gigachat_model,
            credentials=settings.gigachat_credentials,
            scope=settings.gigachat_scope,
            verify_ssl_certs=False
        )

    def _complete_sync(self, system: str, user: str) -> str:
        try:
            payload = Chat(messages=[
                Messages(role=MessagesRole.SYSTEM, content=system),
                Messages(role=MessagesRole.USER, content=user),
            ])
            return self._giga.chat(
                payload
            ).choices[0].message.content.strip()
        except Exception as e:
            logger.error("GigaChat completion failed: %s", e)
            return ""

    async def complete(self, system: str, user: str) -> str:
        return await asyncio.to_thread(
            self._complete_sync, system, user
        )

    def _complete_messages_sync(self, messages: list[dict]) -> str:
        try:
            role_map = {
                "system": MessagesRole.SYSTEM,
                "user": MessagesRole.USER,
                "assistant": MessagesRole.ASSISTANT,
            }
            giga_msgs = [
                Messages(role=role_map[m["role"]], content=m["content"])
                for m in messages
            ]
            return self._giga.chat(
                Chat(messages=giga_msgs)
            ).choices[0].message.content.strip()
        except Exception as e:
            logger.error("GigaChat multi-message failed: %s", e)
            return ""

    async def complete_messages(self, messages: list[dict]) -> str:
        return await asyncio.to_thread(
            self._complete_messages_sync, messages
        )

    async def generate_queries(self, news_text: str,
                               n: int = 5) -> list[str]:
        raw = await self.complete(
            _QUERY_SYSTEM.format(n=n),
            f"Новость: {news_text}\nКоличество запросов: {n}",
        )
        if not raw:
            logger.warning("Query generation returned empty, using fallback")
            return [news_text[:150]]

        queries = []
        for line in raw.split("\n"):
            line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
            if len(line) > 5:
                queries.append(line)
        return queries[:n] if queries else [news_text[:150]]
