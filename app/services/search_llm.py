import logging
import re
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

    def search(self, query: str,
               n: int = 5,
               seen_urls: set | None = None) -> list[dict]:
        if seen_urls is None:
            seen_urls = set()

        search = self._sdk.search_api.web(search_type="ru", user_agent=USER_AGENT)
        operation = search.run_deferred(query, format="xml", page=0)
        xml = operation.wait().decode("utf-8")
        return self._parse(xml, n, seen_urls)

    def multi_search(self, queries: list[str],
                     n_per_query: int = 5) -> list[dict]:
        """Execute multiple queries with global deduplication."""
        seen = set()
        evidences = []
        for q in queries:
            evidences.extend(self.search(q, n=n_per_query, seen_urls=seen))
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


class GigaChatService:

    def __init__(self):
        self._giga = GigaChat(
            model=settings.gigachat_model,
            credentials=settings.gigachat_credentials,
            scope=settings.gigachat_scope,
            verify_ssl_certs=False
        )

    def complete(self, system: str, user: str) -> str:
        payload = Chat(messages=[
            Messages(role=MessagesRole.SYSTEM, content=system),
            Messages(role=MessagesRole.USER, content=user),
        ])
        return self._giga.chat(payload).choices[0].message.content.strip()

    def complete_messages(self, messages: list[dict]) -> str:
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

    def generate_queries(self, news_text: str, n: int = 5) -> list[str]:
        raw = self.complete(
            _QUERY_SYSTEM.format(n=n),
            f"Новость: {news_text}\nКоличество запросов: {n}",
        )
        queries = []
        for line in raw.split("\n"):
            line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
            if len(line) > 5:
                queries.append(line)
        return queries[:n]
