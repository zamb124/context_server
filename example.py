import faiss
import numpy as np

dimension = 768
index = faiss.IndexFlatIP(dimension)

# Добавьте один вектор
vector = np.random.rand(1, dimension).astype('float32')
faiss.normalize_L2(vector)
index.add(vector)

faiss_index_id = 0  # Индекс добавленного вектора (начинается с 0)

# Замените вектор случайным вектором
random_embedding = np.random.rand(1, dimension).astype('float32')
faiss.normalize_L2(random_embedding)

print(f"Shape of random_embedding: {random_embedding.shape}")  # Проверка формы

try:
    index.assign(np.array([faiss_index_id], dtype=np.int64), random_embedding)  # Исправлено! Двумерный массив для индексов
    print("Успешно заменили вектор!")
except Exception as e:
    print(f"Ошибка при замене вектора: {e}")
    import traceback
    traceback.print_exc()