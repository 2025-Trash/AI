# 📄 Dataset Labeling Guide

## 1. Dataset Structure

데이터셋은 다음과 같은 폴더 구조를 가집니다.
각 폴더명은 객체(Object) 및 재질(Material) 라벨을 추론하는 기준이 됩니다.

```
dataset/
│
├── cup/
│   ├── paper/
│   │   ├── img1.jpg
│   │   └── ...
│   └── plastic/
│       ├── img2.jpg
│       └── ...
│
├── cupholder/
│   ├── img3.jpg
│   └── ...
│
├── lid/
│   ├── img4.jpg
│   └── ...
│
├── straw/
│   ├── img5.jpg
│   └── ...
│
└── none/
    ├── img6.jpg
    └── ...
```

---

## 2. Label Mapping

### 2.1 Object Labels

| object\_class | object\_label | 데이터 개수 |
| ------------- | ------------- | ------ |
| `cup_paper`   | 0             | 500    |
| `cup_plastic` | 1             | 500    |
| `cupholder`   | 2             | 500    |
| `lid`         | 3             | 500    |
| `straw`       | 4             | 500    |
| `none`        | 5             | 500    |

### 2.2 Material Labels

| material\_type | material\_label | 데이터 개수 |
| -------------- | --------------- | ------ |
| `paper`        | 0               | 1000   |
| `plastic`      | 1               | 1500   |
| `none`         | 2               | 500    |

---

## 3. Labeling Rules

폴더 경로에 따라 라벨이 자동 부여됩니다.

| 폴더 경로 패턴        | object\_class | material\_type |
| --------------- | ------------- | -------------- |
| `cup/paper/*`   | cup\_paper    | paper          |
| `cup/plastic/*` | cup\_plastic  | plastic        |
| `cupholder/*`   | cupholder     | paper          |
| `lid/*`         | lid           | plastic        |
| `straw/*`       | straw         | plastic        |
| `none/*`        | none          | none           |

---

## 4. CSV 파일 구조 (`labels.csv`)

| 컬럼명              | 설명           |
| ---------------- | ------------ |
| `image_path`     | 이미지 파일 절대 경로 |
| `object_class`   | 객체 클래스명      |
| `material_type`  | 재질 클래스명      |
| `object_label`   | 객체 클래스 정수 라벨 |
| `material_label` | 재질 클래스 정수 라벨 |

---

## 5. Example

```
image_path: /content/dataset/dataset/cup/paper/cup_paper_001.jpg
object_class: cup_paper
material_type: paper
object_label: 0
material_label: 0
```

---

## 6. Generate labels.csv Automatically

아래 코드를 실행하면, 폴더 구조를 기반으로 `labels.csv`를 자동 생성할 수 있습니다.

```python
import os
import pandas as pd
from pathlib import Path

# ====== 라벨 매핑 ======
OBJECT_CLASS_NAMES = ['cup_paper', 'cup_plastic', 'cupholder', 'lid', 'straw', 'none']
OBJECT_CLASS_MAP   = {n:i for i,n in enumerate(OBJECT_CLASS_NAMES)}

MATERIAL_CLASS_NAMES = ['paper', 'plastic', 'none']
MATERIAL_CLASS_MAP   = {n:i for i,n in enumerate(MATERIAL_CLASS_NAMES)}

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# ====== 라벨 추론 함수 ======
def infer_labels_from_path(dataset_dir, path):
    base = Path(dataset_dir).resolve()
    rel  = Path(path).resolve().relative_to(base)
    top  = rel.parts[0].lower()
    sub  = rel.parts[1].lower() if len(rel.parts) > 1 else ''

    if top == 'cup':
        if sub == 'paper':
            return ('cup_paper', 'paper')
        elif sub == 'plastic':
            return ('cup_plastic', 'plastic')
    elif top == 'cupholder':
        return ('cupholder', 'paper')
    elif top == 'lid':
        return ('lid', 'plastic')
    elif top == 'straw':
        return ('straw', 'plastic')
    elif top == 'none':
        return ('none', 'none')
    raise ValueError(f"Unknown path pattern: {path}")

# ====== CSV 생성 ======
def generate_labels_csv(dataset_dir, output_csv="labels.csv"):
    image_paths, object_classes, materials = [], [], []
    for root, _, files in os.walk(dataset_dir):
        for fname in files:
            if fname.lower().endswith(IMG_EXTS):
                fpath = os.path.join(root, fname)
                try:
                    obj_cls, mat = infer_labels_from_path(dataset_dir, fpath)
                    image_paths.append(fpath)
                    object_classes.append(obj_cls)
                    materials.append(mat)
                except Exception as e:
                    print("Skip:", e)

    df = pd.DataFrame({
        "image_path": image_paths,
        "object_class": object_classes,
        "material_type": materials
    })
    df["object_label"] = df["object_class"].map(OBJECT_CLASS_MAP)
    df["material_label"] = df["material_type"].map(MATERIAL_CLASS_MAP)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} records to {output_csv}")
    return df

# ====== 실행 ======
dataset_dir = "/path/to/dataset"
generate_labels_csv(dataset_dir, "labels.csv")
```
