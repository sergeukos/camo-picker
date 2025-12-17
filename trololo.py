# Зависимости: pip install requests Pillow mercantile scikit-learn scikit-image numpy
import requests
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from io import BytesIO
import math
import mercantile
import logging
import re
from typing import Tuple, Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import os
from sklearn.cluster import KMeans
from collections import Counter
from skimage import color
from requests.adapters import HTTPAdapter
import heapq
import sys
import time
import requests.exceptions
import json
import webbrowser

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- КОНСТАНТЫ АНАЛИЗА ---
NAKARTE_INITIAL_URL = "https://nakarte.me/#m=2/0.17578/0.00000&l=E"
N_DOMINANT_COLORS = 6
N_TOP_MATCHES = 5
TILE_SIZE = 512
MAX_DELTA_E_CUTOFF = 16.0
W_COLOR = 1.0
EARTH_RADIUS = 6371
ZOOM_LEVEL = 15
CAMO_VIZ_SIZE = 50
MAX_COMPOSITE_DIM = 3
MAX_DB_LOAD_WORKERS = 15

# =====================================================================
# --- КЛАССЫ ЭФФЕКТИВНОСТИ DELTA E (Обновлены) ---
# =====================================================================
CAMO_EFFECTIVENESS_CLASSES = {
    # Для лесов, умеренных и тропических зон, высокой контрастности
    "FOREST_TEMPERATE_TROPIC_HIGH_VA": {
        "name": "Леса/Тропики/Высокий Контраст",
        "thresholds": [
            # 🌟 Изменен текст: "Отлично (Практически незаметен)" -> "Отлично"
            (3.0, "Отлично"),
            # 🌟 Изменен текст: "Очень хорошо (Сложно заметить)" -> "Очень хорошо"
            (7.0, "Очень хорошо"),
            # 🌟 Изменен текст: "Хорошо (Требует внимания)" -> "Хорошо"
            (13.0, "Хорошо"),
            # 🌟 Изменен текст: "Приемлемо (Срабатывает на дистанции)" -> "Приемлемо"
            (20.0, "Приемлемо"),
            # 🌟 Изменен текст: "Плохо (Слишком заметен)" -> "Плохо"
            (30.0, "Плохо"),
            # 🌟 Изменен текст: "Неэффективен (Полностью виден)" -> "Неэффективен"
            (float('inf'), "Неэффективен")
        ]
    },
    # Для пустынь, городов, снега, низкого контраста
    "ARID_DESERT_URBAN_SNOW_LOW_VA": {
        "name": "Пустыни/Снег/Низкий Контраст",
        "thresholds": [
            # 🌟 Изменен текст: "Отлично (Практически незаметен)" -> "Отлично"
            (4.0, "Отлично"),
            # 🌟 Изменен текст: "Очень хорошо (Сложно заметить)" -> "Очень хорошо"
            (9.0, "Очень хорошо"),
            # 🌟 Изменен текст: "Хорошо (Приемлемый уровень)" -> "Хорошо"
            (16.0, "Хорошо"),
            # 🌟 Изменен текст: "Приемлемо (Нуждается в доработке)" -> "Приемлемо"
            (25.0, "Приемлемо"),
            # 🌟 Изменен текст: "Плохо (Выделяется)" -> "Плохо"
            (35.0, "Плохо"),
            # 🌟 Изменен текст: "Неэффективен (Полностью виден)" -> "Неэффективен"
            (float('inf'), "Неэффективен")
        ]
    }
}


def get_environment_profile_key(lat: float, camo_type: str = "N/A") -> str:
    """Определяет, какой профиль эффективности Delta E использовать."""

    # 1. По широте (приоритет для таблиц)
    abs_lat = abs(lat)
    if abs_lat >= 66.5:
        # Полярные зоны (часто снег, тундра, низкий контраст)
        return "ARID_DESERT_URBAN_SNOW_LOW_VA"
    elif abs_lat >= 23.5:
        # Умеренные зоны (леса, высокая контрастность)
        return "FOREST_TEMPERATE_TROPIC_HIGH_VA"
    else:
        # Тропические зоны (джунгли, высокая контрастность)
        return "FOREST_TEMPERATE_TROPIC_HIGH_VA"


def get_effectiveness_class(score: float, lat: float) -> str:
    """Возвращает текстовый класс эффективности, основанный на широте местности."""
    key = get_environment_profile_key(lat)
    thresholds = CAMO_EFFECTIVENESS_CLASSES[key]["thresholds"]

    for limit, description in thresholds:
        if score < limit:
            return description
    return thresholds[-1][1]


def get_score_color(score: float, lat: float) -> Tuple[int, int, int]:
    """Возвращает RGB цвет на основе класса эффективности, соответствующего широте."""
    key = get_environment_profile_key(lat)
    thresholds = CAMO_EFFECTIVENESS_CLASSES[key]["thresholds"]

    # Логика цвета
    if score < thresholds[0][0]:
        return (0, 150, 0)
    elif score < thresholds[1][0]:
        return (0, 100, 0)
    elif score < thresholds[2][0]:
        return (255, 165, 0)
    elif score < thresholds[3][0]:
        return (200, 100, 0)
    elif score < thresholds[4][0]:
        return (200, 0, 0)
    else:
        return (0, 0, 0)

    # =====================================================================


# --- ССЫЛКИ НА УДАЛЕННУЮ БАЗУ ДАННЫХ (GitHub) ---
# =====================================================================
GITHUB_BASE_URL = "https://raw.githubusercontent.com/sergeukos/camo-picker/main/CamoDatabase/"
GITHUB_METADATA_BASE_URL = GITHUB_BASE_URL + "camo_metadata.json"


# =====================================================================
# --- ФУНКЦИИ ЗАГРУЗКИ ИЗОБРАЖЕНИЙ И ЦВЕТОВОЙ АНАЛИЗ ---
# =====================================================================

def get_dominant_colors_from_url(image_url: str, n_colors: int = N_DOMINANT_COLORS) -> List[Tuple[np.ndarray, float]]:
    """Загружает изображение из Интернета по URL и анализирует доминирующие цвета с повторными попытками."""
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    TIMEOUT = 15

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(image_url, timeout=TIMEOUT)
            response.raise_for_status()

            img = Image.open(BytesIO(response.content)).convert('RGB')
            img_array = np.array(img)
            all_pixels = img_array.reshape(-1, 3)

            valid_pixels_mask = (np.sum(all_pixels, axis=1) > 10) & (np.sum(all_pixels, axis=1) < 755)
            sample_pixels = all_pixels[valid_pixels_mask]

            if len(sample_pixels) < 100:
                logger.warning(f"URL {image_url}: Недостаточно валидных пикселей для анализа.")
                return []

            sample_size = 50000
            if len(sample_pixels) > sample_size:
                indices = np.random.choice(len(sample_pixels), sample_size, replace=False)
                sample_pixels = sample_pixels[indices]

            unique_samples = np.unique(sample_pixels, axis=0)
            n_clusters = min(n_colors, len(unique_samples))

            if n_clusters < 1: return []

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(sample_pixels)
            label_counts = Counter(labels)
            total_pixels = len(sample_pixels)

            dominant_colors = []
            for label, count in label_counts.most_common():
                color_rgb = kmeans.cluster_centers_[label].astype(int)
                percentage = (count / total_pixels) * 100
                dominant_colors.append((color_rgb, percentage))

            return dominant_colors

        except requests.exceptions.RequestException as e:
            logger.warning(f"Ошибка сети/загрузки URL {image_url} (Попытка {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"Не удалось загрузить URL {image_url} после {MAX_RETRIES} попыток.")
                return []

        except Exception as e:
            logger.error(f"Непредвиденная ошибка обработки изображения URL {image_url}: {e}")
            return []

    return []


def _get_camo_image(url: str, size: int) -> Image.Image:
    """Скачивает, конвертирует и обрезает изображение камуфляжа для визуализации."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        return img
    except Exception:
        logger.warning(f"Ошибка загрузки/обработки изображения камуфляжа: {url}. Используется заглушка.")
        return Image.new('RGB', (size, size), (150, 150, 150))


# =====================================================================
# --- КЛАСС ESRI IMAGERY PROCESSOR ---
# =====================================================================
class ESRIImageryProcessor:
    def __init__(self):
        self.base_url = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile"
        self.tile_cache = {}
        self.session = requests.Session()
        self.TILE_SIZE = TILE_SIZE
        self.DISPLAY_SCALE = 0.5
        self.EFFECTIVE_TILE_SIZE = int(self.TILE_SIZE * self.DISPLAY_SCALE)
        self.MAX_DELTA_E_CUTOFF = MAX_DELTA_E_CUTOFF

    def latlon_to_tile(self, lat: float, lon: float, zoom: int) -> mercantile.Tile:
        return mercantile.tile(lon, lat, zoom)

    def get_esri_tile(self, z: int, x: int, y: int, max_retries: int = 3) -> Optional[Image.Image]:
        cache_key = f"{z}/{x}/{y}"
        if cache_key in self.tile_cache:
            return self.tile_cache[cache_key]

        url = f"{self.base_url}/{z}/{y}/{x}"
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                tile_image = Image.open(BytesIO(response.content))
                self.tile_cache[cache_key] = tile_image
                return tile_image
            except Exception as e:
                logger.warning(f"Ошибка загрузки тайла {z}/{x}/{y} (Попытка {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        return None

    def _fetch_single_tile_wrapper(self, z, x, y, max_t, tile_size):
        is_in_bounds = (0 <= x <= max_t and 0 <= y <= max_t)
        center_lat, center_lon = self.get_tile_center_coordinates(x, y, z)

        if not is_in_bounds:
            return (z, x, y, None, center_lat, center_lon, False)

        tile_img = self.get_esri_tile(z, x, y)
        is_available = (tile_img is not None)

        return (z, x, y, tile_img, center_lat, center_lon, is_available)

    def get_tile_center_coordinates(self, tile_x: int, tile_y: int, zoom: int) -> Tuple[float, float]:
        bounds = mercantile.bounds(tile_x, tile_y, zoom)
        center_lat = (bounds.north + bounds.south) / 2
        center_lon = (bounds.east + bounds.west) / 2
        return center_lat, center_lon

    def get_composite_image(self, lat: float, lon: float, zoom: int, tile_size: int = TILE_SIZE) -> Tuple[
        Image.Image, List[Tuple[int, int, int, Optional[Image.Image], float, float, bool]]]:

        center_tile = self.latlon_to_tile(lat, lon, zoom)

        min_x = center_tile.x - 1
        max_x = center_tile.x + 1
        min_y = center_tile.y - 1
        max_y = center_tile.y + 1

        mosaic_cols = max_x - min_x + 1
        mosaic_rows = max_y - min_y + 1

        effective_tile_size = self.EFFECTIVE_TILE_SIZE
        composite_w = mosaic_cols * effective_tile_size
        composite_h = mosaic_rows * effective_tile_size

        composite = Image.new('RGB', (composite_w, composite_h), (128, 128, 128))

        all_tile_requests = []
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                all_tile_requests.append((zoom, center_tile.x + dx, center_tile.y + dy, 2 ** zoom - 1, tile_size))

        fetched_tiles_info = []
        with ThreadPoolExecutor(max_workers=9) as executor:
            futures = [executor.submit(self._fetch_single_tile_wrapper, *req) for req in all_tile_requests]
            for future in futures:
                try:
                    fetched_tiles_info.append(future.result())
                except Exception as exc:
                    logger.error(f'При генерации тайла возникло исключение: {exc}')
                    fetched_tiles_info.append((0, 0, 0, None, 0, 0, False))

        tile_map = {
            (info[1], info[2]): info[3]
            for info in fetched_tiles_info
            if info[6] and info[3] is not None
        }

        if not tile_map:
            logger.warning("Не удалось загрузить ни одного тайла. Возвращается серый плейсхолдер.")
            return composite, fetched_tiles_info

        for (t_x, t_y), tile_img_to_paste in tile_map.items():
            col_index = t_x - min_x
            row_index = t_y - min_y

            tile_img_resized = tile_img_to_paste.resize((effective_tile_size, effective_tile_size),
                                                        Image.Resampling.LANCZOS)

            if isinstance(tile_img_resized, Image.Image):
                composite.paste(tile_img_resized, (col_index * effective_tile_size, row_index * effective_tile_size))

        return composite, fetched_tiles_info

    def _get_valid_pixels(self, tiles_info: List[Tuple[int, int, int, Optional[Image.Image], float, float, bool]]) -> \
    Tuple[
        np.ndarray, int]:
        all_pixels = []
        valid_tiles_count = 0
        for tile_z, tile_x, tile_y, tile_img, center_lat, center_lon, is_available in tiles_info:
            if not is_available or tile_img is None or tile_img.mode != 'RGB':
                continue
            tile_pixels = np.array(tile_img)
            is_esri_sentinel_color = (np.all(tile_pixels == [100, 100, 100]) or
                                      np.all(tile_pixels == [50, 50, 50]) or
                                      np.all(tile_pixels == [200, 100, 100]))
            if is_esri_sentinel_color:
                continue
            avg_color = np.mean(tile_pixels, axis=(0, 1))
            if np.all(avg_color > 240) or np.all(avg_color < 15):
                continue
            all_pixels.append(tile_pixels.reshape(-1, 3))
            valid_tiles_count += 1
        if not all_pixels:
            return np.array([]), 0
        return np.vstack(all_pixels), valid_tiles_count

    def get_average_color_from_tiles(self,
                                     tiles_info: List[Tuple[
                                         int, int, int, Optional[Image.Image], float, float, bool]]) -> np.ndarray:
        all_pixels, valid_tiles_count = self._get_valid_pixels(tiles_info)
        if valid_tiles_count == 0:
            return np.array([128, 128, 128])
        average_color = np.mean(all_pixels, axis=0)
        return average_color.astype(int)

    def get_dominant_colors(self, tiles_info: List[Tuple[int, int, int, Optional[Image.Image], float, float, bool]],
                            n_colors: int = N_DOMINANT_COLORS) -> List[Tuple[np.ndarray, float]]:
        all_pixels, valid_tiles_count = self._get_valid_pixels(tiles_info)
        if valid_tiles_count == 0:
            return []

        sample_size = 50000
        if len(all_pixels) > sample_size:
            indices = np.random.choice(len(all_pixels), sample_size, replace=False)
            sample_pixels = all_pixels[indices]
        else:
            sample_pixels = all_pixels

        unique_samples = np.unique(sample_pixels, axis=0)
        n_clusters = min(n_colors, len(unique_samples))

        if n_clusters < 1: return []

        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(sample_pixels)
            label_counts = Counter(labels)
            total_pixels = len(sample_pixels)
            dominant_colors = []
            for label, count in label_counts.most_common():
                color = kmeans.cluster_centers_[label].astype(int)
                percentage = (count / total_pixels) * 100
                dominant_colors.append((color, percentage))
            return dominant_colors

        except Exception:
            return []

    def _rgb_to_lab_skimage(self, rgb_color: np.ndarray) -> np.ndarray:
        rgb_normalized = rgb_color / 255.0
        lab_color = color.rgb2lab(rgb_normalized[None, None, :])
        return lab_color[0, 0, :]

    def _calculate_color_match_score(self, target_dominants: List[Tuple[np.ndarray, float]],
                                     camo_dominants: List[Tuple[np.ndarray, float]]) -> float:
        """Расчет Delta E 2000 (только цветовой анализ) с логикой штрафов."""
        if not target_dominants or not camo_dominants:
            return float('inf')

        total_weighted_distance = 0.0
        total_penalty = 0.0
        has_major_mismatch = False

        camo_colors_lab = np.array([self._rgb_to_lab_skimage(color) for color, _ in camo_dominants])

        if camo_colors_lab.size == 0: return float('inf')

        for target_color_rgb, target_percentage in target_dominants:
            target_color_lab_single = self._rgb_to_lab_skimage(target_color_rgb)
            L_target, A_target, B_target = target_color_lab_single
            Chroma_target = math.sqrt(A_target ** 2 + B_target ** 2)

            target_lab_array = np.tile(target_color_lab_single, (len(camo_colors_lab), 1))
            distances = color.deltaE_ciede2000(target_lab_array, camo_colors_lab)

            if distances.size == 0: continue

            min_distance = np.min(distances)
            best_match_lab = camo_colors_lab[np.argmin(distances)]
            L_camo, A_camo, B_camo = best_match_lab
            Chroma_camo = math.sqrt(A_camo ** 2 + B_camo ** 2)

            current_penalty = 0.0
            weight_factor = target_percentage / 100.0
            effective_distance = min_distance

            L_diff = abs(L_target - L_camo)
            L_CONTRAST_THRESHOLD = 25.0
            L_CONTRAST_PENALTY_FACTOR = 0.10

            if L_diff > L_CONTRAST_THRESHOLD:
                current_penalty += (L_diff - L_CONTRAST_THRESHOLD) * L_CONTRAST_PENALTY_FACTOR * weight_factor

            if min_distance > 15.0:
                has_major_mismatch = True

            L_diff_old = abs(L_target - L_camo)
            if L_diff_old > 25:
                current_penalty += (L_diff_old - 25) * 0.05 * weight_factor

            if Chroma_target < 10 and Chroma_camo > 20:
                current_penalty += Chroma_camo * 0.1 * weight_factor

            if Chroma_target > 10 and L_target < 30 and Chroma_camo < 10:
                current_penalty += 3.0 * weight_factor

            total_penalty += current_penalty
            total_weighted_distance += effective_distance * weight_factor

            if total_weighted_distance > self.MAX_DELTA_E_CUTOFF * 1.5:
                return float('inf')

        final_score = total_weighted_distance + total_penalty

        if has_major_mismatch:
            final_score *= 1.2

        return final_score

    def create_safe_filename(self, lat: float, lon: float) -> str:
        lat_abs = abs(lat)
        lon_abs = abs(lon)
        lat_hemisphere = "S" if lat < 0 else "N"
        lon_hemisphere = "W" if lon < 0 else "E"
        lat_str = f"{lat_abs:.5f}".replace('.', '_')
        lon_str = f"{lon_abs:.5f}".replace('.', '_')
        filename = f"color_analysis_{lat_str}{lat_hemisphere}_{lon_str}{lon_hemisphere}.png"
        return re.sub(r'[^\w\.-]', '_', filename)

    def format_coordinate(self, coord: float, is_lat: bool = True) -> str:
        hemisphere = "S" if coord < 0 and is_lat else "N" if is_lat else "W" if coord < 0 else "E"
        abs_coord = abs(coord)
        return f"{abs_coord:.6f}°{hemisphere}"

    def create_nakarte_link(self, lat: float, lon: float, zoom: int = ZOOM_LEVEL) -> str:
        return f"https://nakarte.me/#m={zoom}/{lat}/{lon}&l=E"

    def get_best_zoom_level(self, area_width_meters: float) -> int:
        if area_width_meters <= 500:
            return 16
        elif area_width_meters <= 1000:
            return 15
        elif area_width_meters <= 2000:
            return 14
        elif area_width_meters <= 4000:
            return 13
        elif area_width_meters <= 8000:
            return 12
        else:
            return 10

    def find_available_zoom(self, lat: float, lon: float, start_zoom: int, min_zoom: int = 10) -> int:
        for zoom in range(start_zoom, max(min_zoom, 13) - 1, -1):
            try:
                center_tile = self.latlon_to_tile(lat, lon, zoom)
                url = f"{self.base_url}/{zoom}/{center_tile.y}/{center_tile.x}"
                response = self.session.head(url, timeout=3)
                if response.status_code == 200:
                    return zoom
            except Exception:
                continue
        return max(min_zoom, 13)


# =====================================================================
# --- ФУНКЦИИ ЗАГРУЗКИ БАЗЫ ДАННЫХ И ЦВЕТОВОГО АНАЛИЗА ---
# =====================================================================

def load_virtual_camo_data() -> Dict[str, Dict[str, Any]]:
    """
    Загружает метаданные из удаленного JSON, затем выполняет анализ цвета,
    используя увеличенное количество потоков.
    """
    logger.info("Начало загрузки базы данных камуфляжей...")

    # 1. Загрузка JSON-файла с метаданными (с обходом кэша)
    cache_buster_url = GITHUB_METADATA_BASE_URL + f"?t={int(time.time())}"

    try:
        print(f"Загрузка метаданных из {cache_buster_url}...")
        response = requests.get(cache_buster_url, timeout=15)
        response.raise_for_status()
        remote_camo_db = response.json()
        print("Метаданные успешно загружены.")
    except Exception as e:
        logger.error(f"Не удалось загрузить или разобрать файл метаданных: {e}")
        return {}

    loaded_camo_properties = {}
    camo_items = list(remote_camo_db.items())

    def analyze_single_camo(camo_name, properties):
        url = GITHUB_BASE_URL + camo_name
        dominant_colors = get_dominant_colors_from_url(url)

        if dominant_colors:
            new_props = properties.copy()
            new_props['PATH'] = url
            new_props['DOMINANT_COLORS'] = dominant_colors
            return (camo_name, new_props)
        else:
            logger.error(f"Не удалось извлечь цвета из {url}. Камуфляж '{camo_name}' пропущен.")
            return None

    # Увеличенное количество потоков для ускорения
    with ThreadPoolExecutor(max_workers=MAX_DB_LOAD_WORKERS) as executor:
        futures = [executor.submit(analyze_single_camo, name, props) for name, props in camo_items]

        for future in futures:
            result = future.result()
            if result:
                camo_name, new_props = result
                loaded_camo_properties[camo_name] = new_props
                logger.info(f"Успех: {camo_name}")

    logger.info(
        f"Загрузка завершена. Успешно загружено {len(loaded_camo_properties)}/{len(remote_camo_db)} камуфляжей.")
    return loaded_camo_properties


def get_latitude_zone(lat: float) -> str:
    """Определяет широтую зону для фильтрации камуфляжа."""
    abs_lat = abs(lat)
    if abs_lat >= 66.5:
        return "POLAR"
    elif abs_lat >= 23.5:
        return "TEMPERATE"
    else:
        return "TROPICAL"

    # =====================================================================


# --- ФУНКЦИИ ПАРСИНГА КООРДИНАТ ---
# =====================================================================

def parse_coordinate_string(input_str: str) -> Optional[Tuple[float, float]]:
    """
    Универсальный парсер координат для DD, URL-якоря nakarte.me и DMS.
    """
    input_str = input_str.strip()

    # 1. Парсинг URL-якоря nakarte.me
    m_match = re.search(r'#m=\d{1,2}/(-?\d+\.?\d*)/(-?\d+\.?\d*)', input_str)
    if m_match:
        try:
            lat = float(m_match.group(1))
            lon = float(m_match.group(2))
            return lat, lon
        except ValueError:
            pass

            # 2. Парсинг Десятичных Градусов (DD)
    dd_pattern = re.compile(r'(-?\d+\.?\d*)\s*[,\s]\s*(-?\d+\.?\d*)')
    dd_match = dd_pattern.search(input_str)
    if dd_match:
        try:
            lat = float(dd_match.group(1))
            lon = float(dd_match.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except ValueError:
            pass

            # 3. Парсинг Градусы/Минуты/Секунды (DMS)
    dms_pattern = re.compile(
        r"(\d+)[°\s](\d+)['\s](\d+\.?\d*)[\"']?([NnSs])"
        r".*"
        r"(\d+)[°\s](\d+)['\s](\d+\.?\d*)[\"']?([EeWw])"
    )
    dms_match = dms_pattern.search(input_str.replace(" ", ""))
    if dms_match:
        try:
            lat_d, lat_m, lat_s, lat_h = dms_match.groups()[0:4]
            lon_d, lon_m, lon_s, lon_h = dms_match.groups()[4:8]

            lat_val = float(lat_d) + float(lat_m) / 60 + float(lat_s) / 3600
            lon_val = float(lon_d) + float(lon_m) / 60 + float(lon_s) / 3600

            if lat_h in 'Ss': lat_val *= -1
            if lon_h in 'Ww': lon_val *= -1

            return lat_val, lon_val
        except ValueError:
            pass

    return None


# Глобальная переменная для отслеживания первого запуска
FIRST_RUN = True


def get_user_coordinates():
    """
    Функция ввода координат с открытием браузера только при первом запуске.
    """
    global FIRST_RUN

    print("\n" + "=" * 60)
    print("ШАГ 2: ВВЕДИТЕ КООРДИНАТЫ")
    print("=" * 60)
    print("1. **Интерактивный ввод:** Откройте nakarte.me, скопируйте координаты (URL-якорь или ГГ°ММ'СС).")
    print("2. **Ручной ввод:** Введите координаты в виде DD (например: 55.7558 37.6173).")

    if FIRST_RUN:
        print("\n**Автоматическое открытие карты nakarte.me...**")
        try:
            nakarte_url = NAKARTE_INITIAL_URL
            webbrowser.open(nakarte_url)
            print(f"Браузер открыт на {nakarte_url}. Скопируйте данные и вставьте ниже.")
        except Exception:
            print("Не удалось автоматически открыть браузер. Пожалуйста, откройте nakarte.me вручную.")
        FIRST_RUN = False
    else:
        print("\nВведите 'open' для повторного открытия карты.")

    print("-" * 60)

    while True:
        try:
            coords_input = input("Вставьте скопированные данные или введите DD координаты: ").strip()

            if coords_input.lower() == 'open':
                try:
                    nakarte_url = NAKARTE_INITIAL_URL
                    webbrowser.open(nakarte_url)
                    print(f"Браузер открыт на {nakarte_url}. Скопируйте данные и вставьте ниже.")
                    continue
                except Exception:
                    print("Не удалось открыть браузер.")
                    continue

            if not coords_input:
                print("Пустой ввод. Попробуйте снова.")
                continue

            parsed_coords = parse_coordinate_string(coords_input)

            if parsed_coords:
                lat, lon = parsed_coords
            else:
                parts = re.split(r'[,;\s]+', coords_input)
                if len(parts) >= 2:
                    lat = float(parts[0])
                    lon = float(parts[1])
                else:
                    raise ValueError("Не удалось распознать формат координат.")

            if not (-90 <= lat <= 90):
                print(f"Широта {lat} должна быть в диапазоне от -90 до 90 градусов. Попробуйте снова.")
                continue
            if not (-180 <= lon <= 180):
                print(f"Долгота {lon} должна быть в диапазоне от -180 до 180 градусов. Попробуйте снова.")
                continue

            lat_hemisphere = "С" if lat >= 0 else "Ю"
            lon_hemisphere = "В" if lon >= 0 else "З"
            lat_abs = abs(lat)
            lon_abs = abs(lon)

            print(f"Координаты приняты: {lat_abs:.6f}°{lat_hemisphere}, {lon_abs:.6f}°{lon_hemisphere}")
            return lat, lon

        except ValueError as e:
            print(
                f"Ошибка ввода/формата: {e}. Пожалуйста, убедитесь, что вводите правильный формат (например: 55.7558 37.6173 или URL-якорь).")
        except KeyboardInterrupt:
            raise


def get_area_width():
    """Функция выбора ширины области анализа."""
    print("\nШАГ 3: ВЫБЕРИТЕ ШИРИНУ ОБЛАСТИ АНАЛИЗА")
    print("1. Маленькая (около 500 метров)")
    print("2. Средняя (около 1 километра)")
    print("3. Большая (около 2 километров)")
    print("4. Очень большая (около 4 километров)")

    while True:
        try:
            choice = input("Выберите опцию (1-4): ").strip()
            if choice == '1':
                return 500
            elif choice == '2':
                return 1000
            elif choice == '3':
                return 2000
            elif choice == '4':
                return 4000
            else:
                print("Введите число от 1 до 4")
        except KeyboardInterrupt:
            raise


def visualize_results(processor, composite_rgb, target_lat, target_lon, zoom, dominant_colors, nakarte_link,
                      top_matches):
    """
    Визуализация результатов с мозаикой, миниатюрами и АДАПТИВНОЙ таблицей эффективности.
    """

    W_composite, H_composite = composite_rgb.size

    # Константы визуализации
    PADDING = 20
    SIDEBAR_W = 450
    FINAL_W = W_composite + SIDEBAR_W + PADDING * 2
    FINAL_H = max(H_composite, 600) + PADDING * 2

    # 🌟 ИЗМЕНЕНИЕ: Смещение таблицы влево на 15 пикселей (140 -> 125)
    TABLE_X_OFFSET = 125
    # Сдвиг таблицы вниз на 25 пикселей
    TABLE_Y_SHIFT = 25

    # Определяем профиль эффективности по широте для отображения таблицы
    environment_profile_key = get_environment_profile_key(target_lat)
    environment_profile = CAMO_EFFECTIVENESS_CLASSES[environment_profile_key]

    # Настройка шрифтов
    try:
        font_small = ImageFont.truetype("arial.ttf", 14)
        font_medium = ImageFont.truetype("arial.ttf", 18)
        font_large = ImageFont.truetype("arial.ttf", 24)
        # Учитывая сокращенный текст, ширина столбца "Эффективность" может быть уменьшена.
        # Выбираем более компактный моноширинный шрифт.
        font_mono = ImageFont.truetype("DejaVuSansMono.ttf", 12)
    except IOError:
        font_small = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_large = ImageFont.load_default()
        font_mono = ImageFont.load_default()

    final_img = Image.new('RGB', (FINAL_W, FINAL_H), (240, 240, 240))
    draw = ImageDraw.Draw(final_img)

    # 1. Вставляем RGB снимок (мозаику)
    final_img.paste(composite_rgb, (PADDING, PADDING))

    # 2. Боковая панель с информацией
    x_start = W_composite + PADDING * 2
    y_start = PADDING

    draw.text((x_start, y_start), "РЕЗУЛЬТАТЫ АНАЛИЗА", fill=(0, 0, 0), font=font_large)
    y_start += 40

    draw.text((x_start, y_start),
              f"Координаты: {processor.format_coordinate(target_lat)} {processor.format_coordinate(target_lon, is_lat=False)}",
              fill=(50, 50, 50), font=font_small)
    y_start += 20
    draw.text((x_start, y_start), f"Zoom: {zoom}", fill=(50, 50, 50), font=font_small)
    y_start += 25
    draw.text((x_start, y_start), f"Карта: {nakarte_link}", fill=(0, 0, 255), font=font_small)
    y_start += 30

    # Сохраняем начальную Y-координату для блока цветов и таблицы
    color_block_y_start = y_start

    # 3. Доминирующие цвета окружения (Левая часть)
    draw.text((x_start, color_block_y_start), "Доминирующие Цвета Среды:", fill=(0, 0, 0), font=font_medium)
    y_color_block = color_block_y_start + 25

    color_box_size = 40
    color_box_y = y_color_block

    for i, (color_rgb, percentage) in enumerate(dominant_colors):
        box_x = x_start + (i % 2) * (color_box_size + 10)  # 2 колонки
        box_y = color_box_y + (i // 2) * (color_box_size + 20)

        draw.rectangle([box_x, box_y, box_x + color_box_size, box_y + color_box_size], fill=tuple(color_rgb))
        draw.text((box_x, box_y + color_box_size + 2), f"{percentage:.1f}%", fill=(0, 0, 0), font=font_small)

    max_y_colors = color_box_y + (math.ceil(len(dominant_colors) / 2) * (color_box_size + 25))

    # 4. Адаптивная таблица эффективности (Правая часть)
    x_table = x_start + TABLE_X_OFFSET  # 🌟 ПРИМЕНЕНИЕ СДВИГА ПО X
    y_table = color_block_y_start + TABLE_Y_SHIFT

    draw.text((x_table, y_table), "КЛАССЫ ЭФФЕКТИВНОСТИ DELTA E:", fill=(0, 0, 0), font=font_medium)
    y_table += 25

    # Заголовок для выбранного профиля
    draw.text((x_table, y_table), environment_profile["name"] + ":", fill=(50, 50, 50), font=font_small)
    y_table += 18

    # 🌟 Сокращена ширина столбца для 'Эффективность'
    header_line = f" {'Оценка Delta E':<15} | {'Эффективность':<15}"
    draw.text((x_table, y_table), header_line, fill=(0, 0, 0), font=font_mono)
    y_table += 18

    try:
        line_length = draw.textlength(header_line, font=font_mono)
        draw.line([x_table, y_table, x_table + line_length, y_table], fill=(0, 0, 0), width=1)
    except Exception:
        # Уменьшено количество символов для разделителя
        draw.text((x_table, y_table), "-" * 38, fill=(0, 0, 0), font=font_mono)
    y_table += 5

    # Вывод порогов для выбранного профиля
    for limit, description in environment_profile["thresholds"]:
        score_str = f"< {limit:.1f}" if limit != float('inf') else f">= {environment_profile['thresholds'][-2][0]:.1f}"
        # 🌟 Использование сокращенной ширины
        line = f" {score_str:<15} | {description:<15}"

        color_code = get_score_color(limit - 0.01 if limit != float('inf') else limit, target_lat)
        draw.text((x_table, y_table), line, fill=color_code, font=font_mono)

        y_table += 18

    # Смещение для следующего блока (ТОП совпадений)
    y_start = max(max_y_colors, y_table) + 20

    # 5. Лучшие совпадения камуфляжа (с изображениями и цветами)
    draw.text((x_start, y_start), f"ТОП {N_TOP_MATCHES} СОВПАДЕНИЙ КАМУФЛЯЖА:", fill=(0, 0, 0), font=font_medium)
    y_start += 30

    for i, match in enumerate(top_matches):
        camo_name = match['filename'].split('.')[0]
        score = match['final_score']

        # Цвет score соответствует профилю местности, в котором он тестируется
        score_color = get_score_color(score, target_lat)

        camo_img = _get_camo_image(match['path'], CAMO_VIZ_SIZE)
        img_x = x_start
        img_y = y_start
        final_img.paste(camo_img, (img_x, img_y))

        text_x = img_x + CAMO_VIZ_SIZE + 15

        draw.text((text_x, img_y), f"{i + 1}. {camo_name.upper()}", fill=(0, 0, 0), font=font_medium)

        draw.text((text_x, img_y + 18), f"Delta E Score: {score:.2f}",
                  fill=score_color, font=font_small)

        camo_colors = match.get('camo_dominants', [])
        camo_color_box_size = 15
        camo_color_x = text_x
        camo_color_y = img_y + 40

        for color_rgb, percentage in camo_colors:
            draw.rectangle(
                [camo_color_x, camo_color_y, camo_color_x + camo_color_box_size, camo_color_y + camo_color_box_size],
                fill=tuple(color_rgb))
            camo_color_x += camo_color_box_size + 5

        y_start += CAMO_VIZ_SIZE + 20

    return final_img


def run_analysis(processor: ESRIImageryProcessor, camo_properties: Dict):
    """Основная логика анализа."""
    try:
        lat, lon = get_user_coordinates()
        area_width = get_area_width()
    except Exception:
        return

    try:
        print(f"\n{'=' * 60}")
        print(f"Запуск ЦВЕТОВОГО анализа (RGB/Delta E 2000)...")
        start_zoom = processor.get_best_zoom_level(area_width)

        zoom = processor.find_available_zoom(lat, lon, start_zoom)
        nakarte_link = processor.create_nakarte_link(lat, lon, zoom)
        print(f"Выбранный Zoom: {zoom}")

        target_zone = get_latitude_zone(lat)
        filtered_camo_properties = {}
        print(f"Целевая зона: {target_zone}. Применяется фильтрация...")

        for filename, props in camo_properties.items():
            allowed_zones_str = props.get("LATITUDE_RANGE", "ANY")
            allowed_zones = allowed_zones_str.split(',')

            if "ANY" in allowed_zones or target_zone in allowed_zones:
                filtered_camo_properties[filename] = props

        patterns_to_check = filtered_camo_properties.items()
        print(f"Отобрано {len(filtered_camo_properties)} камуфляжей для анализа.")

        print("\nЗагрузка RGB снимков (ESRI World Imagery) для цветового анализа...")
        composite_rgb, tiles_info = processor.get_composite_image(lat, lon, zoom)

        dominant_colors = processor.get_dominant_colors(tiles_info, n_colors=N_DOMINANT_COLORS)

        if not dominant_colors:
            print("Цветовой анализ невозможен (нет данных).")
            # Передаем только необходимые аргументы
            result_image = visualize_results(
                processor,
                composite_rgb,
                lat, lon, zoom, [], nakarte_link,
                top_matches=[]
            )
            filename = processor.create_safe_filename(lat, lon)
            result_image.save(filename)
            print(f"\nРезультат сохранен в {os.path.abspath(filename)} (нет доминирующих цветов для анализа).")
            result_image.show()
            return

        final_matches = []

        for filename, camo_props in patterns_to_check:
            camo_dominants = camo_props.get("DOMINANT_COLORS")
            camo_type = camo_props.get('Type', 'N/A')

            if not camo_dominants: continue

            color_score = processor._calculate_color_match_score(dominant_colors, camo_dominants)

            if color_score > processor.MAX_DELTA_E_CUTOFF * 1.5: continue

            final_score = color_score

            final_matches.append({
                'filename': filename,
                'path': camo_props['PATH'],
                'final_score': final_score,
                'camo_dominants': camo_dominants,
                'type': camo_type
            })

        final_matches.sort(key=lambda x: x['final_score'])
        top_matches = final_matches[:N_TOP_MATCHES]

        # Визуализация
        result_image = visualize_results(
            processor,
            composite_rgb,
            lat, lon, zoom, dominant_colors, nakarte_link,
            top_matches=top_matches
        )

        print(f"\n--- ТОП {N_TOP_MATCHES} СОВПАДЕНИЙ (Delta E 2000) ---")
        for match in top_matches:
            effectiveness = get_effectiveness_class(match['final_score'], lat)  # Используем lat для определения класса
            print(f"| {match['filename']:<20} | E00: {match['final_score']:.2f} | Класс: {effectiveness}")
        print("---------------------------------------")

        filename = processor.create_safe_filename(lat, lon)
        result_image.save(filename)
        print(f"\nРезультат сохранен в {os.path.abspath(filename)}")
        result_image.show()

    except Exception as e:
        print(f"Общая ошибка анализа: {e}")
        logger.error(f"Ошибка в run_analysis: {e}", exc_info=True)


def main():
    """Главная функция программы."""
    processor = ESRIImageryProcessor()

    print("\n" + "=" * 60)
    print("ШАГ 1: Загрузка и анализ базы данных камуфляжей с GitHub...")

    CAMO_PROPERTIES = load_virtual_camo_data()

    if not CAMO_PROPERTIES:
        print("\nВНИМАНИЕ: База данных камуфляжей не загружена. Проверьте ссылки и формат файлов.")
        print("Анализ невозможен. Программа завершена.")
        return

    print(f"База данных успешно загружена. Доступно {len(CAMO_PROPERTIES)} камуфляжей для анализа.")

    while True:
        try:
            run_analysis(processor, CAMO_PROPERTIES)

            print("\n" + "=" * 60)
            restart_choice = input(
                "Анализ завершен. Нажмите Enter, чтобы провести **новый анализ координат**, или введите 'exit' для выхода: ").strip().lower()

            if restart_choice == 'exit':
                print("Выход из программы. До свидания!")
                break

        except KeyboardInterrupt:
            print("\nОбнаружено прерывание. Выход из программы.")
            break
        except Exception as e:
            print(f"\nПроизошла непредвиденная ошибка в цикле анализа: {e}. Пожалуйста, попробуйте снова.")
            logger.error(f"Ошибка в цикле main: {e}", exc_info=True)
            continue


if __name__ == "__main__":
    main()