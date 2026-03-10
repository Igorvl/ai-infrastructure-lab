import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

logger = logging.getLogger("AI-Router.Qdrant")

class VectorDBManager:
    def __init__(self):
        # Подключаемся к Qdrant
        self.client = QdrantClient(host="ai-qdrant", port=6333)
        self.collection_name = "project_prompts"
        
        # Модель для получения векторов. "BAAI/bge-small-en-v1.5" - отличный быстрый стандарт
        self.embedding_model = "BAAI/bge-small-en-v1.5"
        
        # Qdrant автоматически скачает эту модель при первом запуске!
        self.client.set_model(self.embedding_model)
        
        # Инициализируем коллекцию при старте
        self._init_collection()

    def _init_collection(self):
        try:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=self.client.get_fastembed_vector_params()
                )
                logger.info(f"✅ Создана коллекция векторов '{self.collection_name}'")
        except Exception as e:
            logger.info(f"🔄 Коллекция '{self.collection_name}' уже создана другим процессом: {e}")
    def add_prompt(self, project_slug: str, prompt_text: str, generation_id: str):
        """Векторизует и сохраняет промпт в базу."""
        try:
            # Метод add автоматически делает текст -> вектор -> сохранение!
            self.client.add(
                collection_name=self.collection_name,
                documents=[prompt_text],
                metadata=[{
                    "project_slug": project_slug,
                    "generation_id": generation_id,
                    "text": prompt_text
                }]
            )
            logger.info(f"✅ Вектор добавлен для генерации: {generation_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка добавления вектора: {e}")
            return False

    def search_similar_prompts(self, project_slug: str, query_text: str, limit: int = 3):
        """Ищет похожие промпты внутри конкретного проекта."""
        try:
            # Query автоматически делает текст -> вектор -> поиск!
            search_result = self.client.query(
                collection_name=self.collection_name,
                query_text=query_text,
                limit=limit,
                # Ищем только по текущему проекту!
                # query_filter=models.Filter(...) — добавим чуть позже при интеграции
            )
            
            results = []
            for point in search_result:
                results.append({
                    "score": point.score,
                    "text": point.metadata.get("text"),
                    "generation_id": point.metadata.get("generation_id")
                })
            return results
        except Exception as e:
            logger.error(f"❌ Ошибка поиска векторов: {e}")
            return []

# Singleton (по аналогии с PostgreSQL)
vector_db = VectorDBManager()
