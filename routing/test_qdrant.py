import random
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

print("🔄 Подключаемся к Qdrant...")
# Подключаемся к Qdrant по имени сервиса в Docker-сети
client = QdrantClient(host="ai-qdrant", port=6333)

collection_name = "test_collection"
vector_size = 4 # Для теста возьмем маленькие векторы из 4 чисел

# 1. Создаем коллекцию (если она уже есть - пересоздадим)
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)
    
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
)
print(f"✅ Коллекция '{collection_name}' создана!")

# 2. Добавляем данные (Вектор + полезная нагрузка "payload")
points = [
    PointStruct(id=1, vector=[0.1, 0.2, 0.3, 0.4], payload={"text": "красный спорткар"}),
    PointStruct(id=2, vector=[0.9, 0.8, 0.7, 0.6], payload={"text": "синий трактор"}),
    PointStruct(id=3, vector=[0.1, 0.2, 0.3, 0.5], payload={"text": "красная феррари"}), # Очень похож на 1
]

client.upsert(collection_name=collection_name, points=points)
print(f"✅ 3 вектора успешно добавлены!")

# 3. Делаем поиск
# Допустим, новый запрос дал такой вектор:
query_vector = [0.15, 0.25, 0.35, 0.45] 

print("\n🔍 Ищем векторы, похожие на", query_vector)
search_result = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=2 # Хотим 2 самых близких результата
)

for result in search_result.points:
    print(f"Совпадение: {result.score:.4f} | Текст: {result.payload['text']}")
