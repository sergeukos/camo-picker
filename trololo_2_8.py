# Tactical Engine V12.1 | "Luminance Spark"
# SPARK LOGIC | BRIGHTNESS PRIORITY | L-CAP PROTECTION | ZOOM 7x7

import requests
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from io import BytesIO
import math
import mercantile
import re
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans
from collections import Counter
from skimage import color
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import webbrowser
from tqdm import tqdm

# --- STATIC CONFIG ---
MATRIX_SIZE = 96
GITHUB_BASE_URL = "https://raw.githubusercontent.com/sergeukos/camo-picker/main/CamoDatabase/"
GITHUB_METADATA_URL = GITHUB_BASE_URL + "camo_metadata.json"

HIGH_VIS_STANDARDS = {
    "Safety Orange (SOS)": [255, 79, 0],
    "Neon Lemon (Hi-Vis)": [212, 255, 0],
    "Electric Blue": [0, 100, 255],
    "Emergency Magenta": [255, 0, 144],
    "Signal White": [255, 255, 255],
    "Solar Yellow": [255, 210, 0]
}


class TacticalCoreV12_1:
    def __init__(self):
        print("[*] Initializing Engine V12.1 | Luminance Spark Edition...")
        self.session = requests.Session()
        retries = Retry(total=20, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries, pool_connections=50, pool_maxsize=50))
        self.db = {}

    def _apply_optical_pivot(self, doms, avg_l, mode='TACTICAL'):
        """
        V12.1: Добавлен L-Cap. Если на снимке есть зелень, ограничиваем макс. яркость,
        чтобы не вылетал 'Alpine' в летнем лесу.
        """
        pivoted = []
        # Проверяем наличие "зеленки" в анализе (упрощенно по Hue/Saturation в будущем, сейчас по Lab)
        has_greenery = any(d[0][1] > d[0][0] and d[0][1] > d[0][2] for d in doms)

        for i, (rgb, weight) in enumerate(doms):
            lab = color.rgb2lab(np.array([[rgb]], dtype=np.float32) / 255.0).reshape(3)
            if mode == 'TACTICAL':
                # Сдвиг яркости
                lab[0] = np.clip(lab[0] + (i * 10 - 15), 10, 95)
                # Летний предохранитель: не даем белому сиять ярче 85, если есть зелень
                if has_greenery and lab[0] > 85: lab[0] = 85
                lab[1:] *= 1.3
            else:
                lab[0] = 95 if lab[0] < 50 else 15
            new_rgb = (color.lab2rgb(lab.reshape(1, 1, 3))[0, 0] * 255).astype(np.uint8).tolist()
            pivoted.append([new_rgb, weight])
        return pivoted

    def _draw_seamless(self, draw, shape, coords, fill):
        x1, y1, x2, y2 = coords
        for ox in [0, -MATRIX_SIZE, MATRIX_SIZE]:
            for oy in [0, -MATRIX_SIZE, MATRIX_SIZE]:
                if shape == 'rect':
                    draw.rectangle([x1 + ox, y1 + oy, x2 + ox, y2 + oy], fill=fill)
                elif shape == 'ellipse':
                    draw.ellipse([x1 + ox, y1 + oy, x2 + ox, y2 + oy], fill=fill)

    def generate_pixel_apex(self, doms, entropy):
        """
        PIXEL APEX V12.1: SPARK & LAYERED STRUCTURE
        Решает проблему "китайского однотона" через приоритет светлых пятен.
        """
        doms.sort(key=lambda x: x[1], reverse=True)

        # 1. Базовый раскол (Фон)
        canvas = Image.new('RGB', (MATRIX_SIZE, MATRIX_SIZE), tuple(map(int, doms[0][0])))
        draw = ImageDraw.Draw(canvas)

        c2 = tuple(map(int, doms[1][0]))
        for _ in range(np.random.randint(6, 10)):
            x, y = np.random.randint(0, MATRIX_SIZE), np.random.randint(0, MATRIX_SIZE)
            bw, bh = np.random.randint(25, 45), np.random.randint(15, 30)
            self._draw_seamless(draw, 'rect', [x, y, x + bw, y + bh], fill=c2)

        # 2. СОРТИРОВКА ПО ЯРКОСТИ (L)
        # Рисуем от самых темных к самым светлым, чтобы "искры" ложились сверху
        others = doms[1:5]
        others.sort(key=lambda x: color.rgb2lab(np.array([[x[0]]], dtype=np.float32) / 255.0)[0, 0, 0])

        for rgb, weight in others:
            c = tuple(map(int, rgb))
            lab_l = color.rgb2lab(np.array([[rgb]], dtype=np.float32) / 255.0)[0, 0, 0]

            # Буст для светлых пятен (искр)
            is_spark = lab_l > 58
            boost = 2.8 if is_spark else 1.0
            num_clusters = int(np.clip((weight / 4) * boost, 3, 14))

            for _ in range(num_clusters):
                cx, cy = np.random.randint(0, MATRIX_SIZE), np.random.randint(0, MATRIX_SIZE)

                # Искры делаем чуть мельче и "острее"
                grain_w = 4 if is_spark else (8 if entropy < 18 else 4)
                grain_h = 2 if is_spark else (4 if entropy < 18 else 2)

                for _ in range(np.random.randint(8, 18)):
                    offset_x = (np.random.randint(-14, 14) // grain_w) * grain_w
                    offset_y = (np.random.randint(-10, 10) // grain_h) * grain_h
                    px, py = cx + offset_x, cy + offset_y
                    self._draw_seamless(draw, 'rect', [px, py, px + grain_w, py + grain_h], fill=c)

        return np.array(canvas)

    def generate_spots_850(self, doms):

        doms.sort(key=lambda x: x[1], reverse=True)

        # Вычисляем среднюю яркость всех доминант для ориентира
        all_labs = [color.rgb2lab(np.array([[d[0]]], dtype=np.float32) / 255.0)[0, 0, 0] for d in doms]
        avg_env_l = np.mean(all_labs)

        canvas = Image.new('RGB', (MATRIX_SIZE, MATRIX_SIZE), tuple(map(int, doms[0][0])))

        for rgb, weight in doms[1:]:
            # --- ЛОКАЛЬНАЯ КОРРЕКЦИЯ КОНТРАСТА ---
            lab = color.rgb2lab(np.array([[rgb]], dtype=np.float32) / 255.0).reshape(3)

            # Если цвет слишком светлый относительно среднего по палате - приземляем его
            if lab[0] > avg_env_l + 15:
                lab[0] = avg_env_l + 15  # Лимит контраста сверху

            # Слегка приглушаем насыщенность для Spots (эффект пыли)
            lab[1:] *= 0.8

            c_corrected = (color.lab2rgb(lab.reshape(1, 1, 3))[0, 0] * 255).astype(np.uint8).tolist()
            c = tuple(map(int, c_corrected))

            # РИСОВАНИЕ ПЯТЕН (Геометрия без изменений)
            mask = Image.new('L', (MATRIX_SIZE, MATRIX_SIZE), 0);
            d = ImageDraw.Draw(mask)
            num_seeds = int(weight * 1.8) if weight > 7 else int(weight * 8)
            for _ in range(max(1, num_seeds // 2)):
                rx, ry = np.random.randint(0, MATRIX_SIZE), np.random.randint(0, MATRIX_SIZE)
                for _ in range(np.random.randint(2, 5)):
                    ox, oy = rx + np.random.randint(-12, 12), ry + np.random.randint(-12, 12)
                    size = np.random.randint(20, 45) if weight > 15 else np.random.randint(5, 15)
                    self._draw_seamless(d, 'ellipse', [ox - size // 2, oy - size // 2, ox + size // 2, oy + size // 2],
                                        fill=255)

            mask = mask.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(2.5, 4.0))).point(
                lambda p: 255 if p > 135 else 0)
            canvas.paste(Image.new('RGB', (MATRIX_SIZE, MATRIX_SIZE), c), (0, 0), mask)

        return np.array(canvas)

    def generate_heritage_woodland(self, doms):
        canvas = Image.new('RGB', (MATRIX_SIZE, MATRIX_SIZE), tuple(map(int, doms[0][0])))
        for rgb, weight in doms[1:4]:
            c = tuple(map(int, rgb))
            mask = Image.new('L', (MATRIX_SIZE, MATRIX_SIZE), 0);
            d = ImageDraw.Draw(mask)
            for _ in range(3):
                x, y = np.random.randint(0, MATRIX_SIZE), np.random.randint(0, MATRIX_SIZE)
                s = np.random.randint(40, 75)
                self._draw_seamless(d, 'ellipse', [x - s // 2, y - s // 2, x + s // 2, y + s // 2], fill=255)
            mask = mask.filter(ImageFilter.GaussianBlur(radius=6.0)).point(lambda p: 255 if p > 100 else 0)
            canvas.paste(Image.new('RGB', (MATRIX_SIZE, MATRIX_SIZE), c), (0, 0), mask)
        return np.array(canvas)

    def generate_rescue_dazzle(self, env_doms):
        avg_rgb = np.mean([d[0] for d in env_doms], axis=0)
        lab_bg = color.rgb2lab(np.array([[avg_rgb]], dtype=np.float32) / 255.0).reshape(3)
        best_n, max_de = "Signal White", 0
        for n, rgb in HIGH_VIS_STANDARDS.items():
            de = color.deltaE_ciede2000(lab_bg, color.rgb2lab(np.array([[rgb]], dtype=np.float32) / 255.0).reshape(3))
            if de > max_de: max_de, best_n = de, n
        sig_c = tuple(HIGH_VIS_STANDARDS[best_n])
        canvas = Image.new('RGB', (MATRIX_SIZE, MATRIX_SIZE), sig_c)
        d = ImageDraw.Draw(canvas);
        white = (255, 255, 255) if best_n != "Signal White" else (0, 0, 0)
        for size in range(MATRIX_SIZE, 0, -18):
            cf = white if (size // 18) % 2 == 0 else sig_c
            d.regular_polygon((MATRIX_SIZE // 2, MATRIX_SIZE // 2, size // 1.1), 4, rotation=45, fill=cf)
        return np.array(canvas)

    def get_terrain(self, lat, lon, radius_m):
        z = int(np.clip(18 - math.log2(radius_m / 400), 10, 19))
        while z > 0:
            m_deg = 111320 * math.cos(math.radians(lat))
            d = radius_m / m_deg
            ul, lr = mercantile.tile(lon - d, lat + d, z), mercantile.tile(lon + d, lat - d, z)
            x_r, y_r = range(ul.x, lr.x + 1), range(ul.y, lr.y + 1)
            if len(x_r) * len(y_r) > 49: z -= 1; continue
            canvas = Image.new('RGB', (len(x_r) * 256, len(y_r) * 256))
            tile_hashes = []

            def fetch(tx, ty):
                try:
                    r = self.session.get(
                        f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{ty}/{tx}",
                        timeout=10)
                    return Image.open(BytesIO(r.content))
                except:
                    return Image.new('RGB', (256, 256), (60, 65, 60))

            with ThreadPoolExecutor(max_workers=25) as ex:
                tasks = [(i, j, tx, ty) for i, tx in enumerate(x_r) for j, ty in enumerate(y_r)]
                for i, j, tx, ty in tqdm(tasks, desc=f"[Z{z}] Downloading"):
                    tile = fetch(tx, ty)
                    canvas.paste(tile, (i * 256, j * 256))
                    tile_hashes.append(hash(tile.tobytes()))
            if len(set(tile_hashes)) <= 1 and len(tile_hashes) > 1:
                print(f"[!] Stub detected at Z{z}. Scaling up...")
                z -= 1;
                continue
            else:
                break
        pix = np.array(canvas.resize((150, 150))).reshape(-1, 3)
        km = KMeans(n_clusters=6, n_init='auto').fit(pix)
        doms = [[km.cluster_centers_[i].tolist(), (Counter(km.labels_)[i] / 22500) * 100] for i in range(6)]
        avg_l = np.mean([color.rgb2lab(np.array([[d[0]]], dtype=np.float32) / 255.0).reshape(3)[0] for d in doms])
        return doms, canvas, avg_l, np.std(pix)

    def analyze_camo_irl(self, env, camo_data, avg_l, mode='T'):
        ref_spots = self.generate_spots_850(env)
        lab_ref = color.rgb2lab(np.array(ref_spots, dtype=np.float32) / 255.0)
        c_rgb = camo_data if isinstance(camo_data, list) else camo_data['DOMINANTS'][0][0]
        lab_camo = color.rgb2lab(np.full((96, 96, 3), c_rgb, dtype=np.float32) / 255.0)
        de = np.mean(color.deltaE_ciede2000(lab_ref, lab_camo))
        l_diff = lab_camo[0, 0, 0] - avg_l
        verdict = "SIGNAL" if mode == 'R' else (
            "MATCH" if -20 < l_diff < 15 else ("GLOWING" if l_diff > 15 else "BLACK HOLE"))
        return de, l_diff, verdict

    def load_db(self):
        meta = self.session.get(GITHUB_METADATA_URL).json()

        def proc(n, p):
            try:
                img = Image.open(BytesIO(self.session.get(GITHUB_BASE_URL + n).content)).convert('RGB').resize((64, 64))
                pix = np.asarray(img).reshape(-1, 3);
                km = KMeans(n_clusters=4, n_init='auto').fit(pix)
                p.update({'DOMINANTS': [[km.cluster_centers_[i].tolist(), 25] for i in range(4)],
                          'PATH': GITHUB_BASE_URL + n})
                return n, p
            except:
                return None

        with ThreadPoolExecutor(max_workers=30) as ex:
            self.db = {r[0]: r[1] for r in
                       tqdm(ex.map(lambda x: proc(*x), meta.items()), total=len(meta), desc="Syncing DB") if r}


class ReporterV12:
    def draw_all(self, map_img, avg_l, mats, ranks, mode_idx):
        canvas = Image.new('RGB', (1900, 2100), (10, 12, 14));
        draw = ImageDraw.Draw(canvas)
        draw.text((50, 20), f"TACTICAL ENGINE V12.1 | LUMINANCE SPARK | LIGHT: {avg_l:.1f}", fill=(0, 255, 180))
        canvas.paste(map_img.resize((1000, 1000)), (50, 80))

        tact_list = ["Pixel", "Spots", "Heritage"]
        for i, name in enumerate(tact_list):
            if name in mats:
                x = 1100 + (i * 260)
                canvas.paste(Image.fromarray(mats[name]).resize((245, 245), Image.NEAREST), (x, 110))
                draw.rectangle([x, 110, x + 245, 110 + 245], outline=(0, 255, 150), width=2)
                draw.text((x, 80), name.upper(), fill=(0, 255, 180))

        t_ranks = [r for r in ranks if r[4] is not None]
        for i, (name, de, dl, v, p) in enumerate(t_ranks[:7]):
            ry = 410 + (i * 115)
            try:
                canvas.paste(Image.open(BytesIO(requests.get(p).content)).convert('RGB').resize((90, 90)), (1100, ry))
            except:
                pass
            draw.text((1210, ry), f"{name.upper()[:35]}", fill=(255, 255, 255))
            draw.text((1210, ry + 30), f"Delta-E: {de:.2f} | Delta-L: {dl:.1f} | {v}", fill=(160, 160, 160))
            bw = int(np.clip(450 - de * 8, 10, 450))
            draw.rectangle([1210, ry + 75, 1210 + bw, ry + 85], fill=(0, 255, 130) if "MATCH" in v else (255, 100, 0))

        if "Rescue Master" in mats:
            y_rescue_start = 1250
            draw.rectangle([50, y_rescue_start - 20, 1850, y_rescue_start - 15], fill=(255, 120, 0))
            draw.text((50, y_rescue_start), "SEARCH & RESCUE OPERATIONS", fill=(255, 120, 0))
            canvas.paste(Image.fromarray(mats["Rescue Master"]).resize((420, 420), Image.NEAREST),
                         (50, y_rescue_start + 50))
            r_ranks = [r for r in ranks if r[4] is None]
            for i, (name, de, dl, v, p) in enumerate(r_ranks):
                rx = 520 + (i // 3) * 440
                ry = (y_rescue_start + 50) + (i % 3) * 120
                draw.rectangle([rx, ry, rx + 90, ry + 90], fill=tuple(HIGH_VIS_STANDARDS[name]))
                draw.text((rx + 110, ry + 10), name.upper(), fill=(255, 255, 255))
                draw.text((rx + 110, ry + 40), f"VISIBILITY: {de:.1f}", fill=(200, 200, 200))
                bw = int(np.clip(de * 5.5, 10, 300))
                draw.rectangle([rx + 110, ry + 75, rx + 110 + bw, ry + 85], fill=(255, 120, 0))
        canvas.save("output_v12.png");
        webbrowser.open("output_v12.png")


def main():
    core = TacticalCoreV12_1();
    rep = ReporterV12();
    core.load_db()
    while True:
        raw = input("\n[INPUT] URL/Coords: ");
        f = re.findall(r"[-+]?\d*\.\d+|\d+", raw)
        if len(f) >= 2:
            rad = input("[INPUT] Radius [400]: ");
            rad = int(rad) if rad.isdigit() else 400
            m_idx = input("[INPUT] Mode [1:FULL, 2:CAMO, 3:RESCUE]: ")
            env, map_img, avg_l, entropy = core.get_terrain(float(f[-2]), float(f[-1]), rad)
            pivoted = core._apply_optical_pivot(env, avg_l)
            mats, ranks = {}, []
            if m_idx in ['1', '2']: \
                    mats['Pixel'] = core.generate_pixel_apex(pivoted, entropy)
            mats['Spots'] = core.generate_spots_850(pivoted)
            mats['Heritage'] = core.generate_heritage_woodland(pivoted)
            ranks += sorted([(n, *core.analyze_camo_irl(env, d, avg_l, 'T'), d['PATH']) for n, d in core.db.items()],
                            key=lambda x: x[1])
        if m_idx in ['1', '3']:
            mats['Rescue Master'] = core.generate_rescue_dazzle(env)
            ranks += sorted(
                [(n, *core.analyze_camo_irl(env, c, avg_l, 'R'), None) for n, c in HIGH_VIS_STANDARDS.items()],
                key=lambda x: x[1], reverse=True)
        rep.draw_all(map_img, avg_l, mats, ranks, m_idx)


if __name__ == "__main__": main()