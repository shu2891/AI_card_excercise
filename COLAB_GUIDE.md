# Colab 使用教學

這份文件是給「第一次把這個專案拿去 Google Colab 跑」的人看的。

目標很簡單：

- 每個專題用自己的資料夾
- 每個專題訓練自己的 Style LoRA
- 最後只輸出生成圖片 `.png`
- 不使用卡框

---

## 1. 你會得到什麼

這個 repo 目前提供的是一個 **Colab 版工作流**：

1. 掛載 Google Drive
2. 準備訓練資料集
3. 訓練專題專用的 Style LoRA
4. 載入剛剛訓練好的 LoRA
5. 生成圖片並輸出到 Google Drive

你真正會使用的檔案有兩個：

- `notebooks/project_style_lora_colab.ipynb`
- `colab_style_lora_workflow.py`

---

## 2. 專案資料夾結構

每個專題都要放在 Google Drive 裡自己的資料夾。

格式如下：

```text
/content/drive/MyDrive/AI_Card_Project/{PROJECT_NAME}/
  raw_images/
  dataset/
  lora/
  outputs/
```

實際例子：

```text
/content/drive/MyDrive/AI_Card_Project/翻身/
  raw_images/
  dataset/
  lora/
  outputs/
```

各資料夾用途：

- `raw_images/`
  - 放原始圖片
  - 可以有子資料夾，例如 `sky/`、`kitchen/`、`orb/`
- `dataset/`
  - 程式自動整理出的 LoRA 訓練資料
- `lora/`
  - 訓練產出的 checkpoint 和最終權重
- `outputs/`
  - 最後生成出來的 `.png`

---

## 3. raw_images 要怎麼放

你可以用兩種方式整理：

### 方法 A：全部直接丟在 `raw_images/`

例如：

```text
raw_images/
  天空魚.png
  天空魚焦.png
  廚房魚.png
  廚房魚生.png
  蛋魚正常.png
```

### 方法 B：分子資料夾

例如：

```text
raw_images/
  sky/
    天空魚.png
    天空魚焦.png
  kitchen/
    廚房魚.png
    廚房魚生.png
  orb/
    蛋魚正常.png
```

目前程式會根據：

- 檔名中的 `生 / 正常 / 焦`
- 檔名或父資料夾中的 `天空 / 廚房 / 蛋 / 馬路 / orb`

去自動推測 caption。

---

## 4. 第一次使用前要做什麼

### Step 1：打開 Colab notebook

建議直接把 repo clone 到 Colab 後，打開：

- `notebooks/project_style_lora_colab.ipynb`

### Step 2：確認 Colab GPU

在 Colab 選：

- `Runtime`
- `Change runtime type`
- `Hardware accelerator`
- 選 `GPU`

沒有 GPU 就不要開始訓練。

### Step 3：把原始圖片先放進 Google Drive

你要先在 Google Drive 建好資料夾，例如：

```text
MyDrive/AI_Card_Project/翻身/raw_images/
```

然後把原始圖放進去。

---

## 5. notebook 的五段流程

這個 notebook 已經拆成五段。

你只要從上往下跑。

### 第 1 段：安裝

這段會做：

- 掛載 Google Drive
- 安裝 Python 套件
- clone repo 到 Colab 工作區

如果你今天是第一次開新的 Colab session，這段一定要跑。

---

### 第 2 段：設定專題

這段你要改幾個值：

- `PROJECT_NAME`
- `PROMPT`
- `SEED`
- `NUM_IMAGES`

例如：

```python
PROJECT_NAME = '翻身'
PROMPT = 'fishseriesstyle, same series, pastel chalk fish, single fish hero, mythic regal power'
SEED = 42
NUM_IMAGES = 4
```

這段會自動組出：

- `base_dir`
- `source_dir`
- `dataset_dir`
- `lora_dir`
- `output_dir`

所以你不用自己手動拼路徑。

---

### 第 3 段：準備資料

這段會執行 dataset preparation。

它會：

- 讀取 `raw_images/`
- 複製圖片到 `dataset/images/`
- 建立 `dataset/metadata.jsonl`

跑完後會回傳 summary，包含：

- 總資料數
- 原始圖片數量
- 範例 caption

如果這一步失敗，通常是：

- `raw_images/` 裡沒有圖
- 路徑打錯

---

### 第 4 段：訓練

這段會真正開始訓練 LoRA。

它會使用：

- `stabilityai/stable-diffusion-xl-base-1.0`
- Hugging Face `diffusers` 官方 SDXL LoRA 訓練腳本

訓練輸出會放到：

```text
/content/drive/MyDrive/AI_Card_Project/{PROJECT_NAME}/lora/
```

如果中途斷線，資料通常還在 Drive。

但 Colab session 中斷後，要重新掛載並重新設定環境才能續做。

---

### 第 5 段：生成

這段會：

1. 自動抓最新的 LoRA checkpoint
2. 載入 LoRA
3. 用你給的 prompt 生成圖片
4. 存成 `.png`

輸出位置：

```text
/content/drive/MyDrive/AI_Card_Project/{PROJECT_NAME}/outputs/
```

---

## 6. 你平常只需要改哪幾個參數

最常改的是這幾個：

### `PROJECT_NAME`

控制你現在跑的是哪個專題。

例如：

- `翻身`
- `廚房`
- `天空`

### `PROMPT`

控制生成主題。

例如：

```python
'fishseriesstyle, same series, pastel chalk fish, single fish hero, cloud motifs'
```

或：

```python
'fishseriesstyle, same series, pastel chalk fish, single fish hero, kitchen motifs'
```

### `SEED`

控制隨機性。

- 同樣的 seed 比較容易得到可重現結果
- 換 seed 可以測更多變體

### `NUM_IMAGES`

一次要生成幾張圖。

---

## 7. 什麼時候要重訓，什麼時候不用

### 需要重訓

當你有以下變動時，建議重訓：

- 加了很多新風格圖片
- 專題畫風明顯不同
- 想做另一個新專題

### 不一定要重訓

如果只是：

- 改 prompt
- 改 seed
- 多生成幾張圖

那通常直接用現有 LoRA 就可以。

---

## 8. 推薦的工作方式

比較穩的流程是：

1. 先整理 `raw_images/`
2. 跑 dataset preparation
3. 先訓一版 LoRA
4. 用短 prompt 測試
5. 覺得系列感不夠，再補資料或再訓

不要一開始就把 prompt 寫太長。

SDXL 很容易截斷 prompt，真正重要的風格詞要放前面。

---

## 9. 常見問題

### 問題 1：LoRA 訓練很慢

這是正常的。

Colab GPU 強度不固定，而且 SDXL LoRA 本來就不算輕。

你可以先：

- 降低 `MAX_TRAIN_STEPS`
- 先測少量資料

---

### 問題 2：生成結果不像同系列

先檢查：

- 原始資料夠不夠集中
- 畫風是不是混太多種
- prompt 有沒有太長

如果風格資料本身很散，LoRA 也很難學穩。

---

### 問題 3：Colab 斷線

這也是正常風險。

建議：

- 重要輸出全部存在 Google Drive
- 不要把訓練成果只放在 `/content/`

---

### 問題 4：生成圖在哪裡

在：

```text
/content/drive/MyDrive/AI_Card_Project/{PROJECT_NAME}/outputs/
```

---

## 10. 建議你第一次這樣跑

如果你是第一次操作，我建議：

1. 建一個專題資料夾，例如 `翻身`
2. 丟少量但風格一致的圖片進 `raw_images/`
3. 把 `NUM_IMAGES` 先設成 `2`
4. 跑完整個 notebook
5. 確認真的有產出到 `outputs/`

先確認流程通，再追求品質。

---

## 11. 最後一句

你現在可以把這個 repo 當成：

- 每個專題一個資料夾
- 每個專題一個 LoRA
- 每個專題一份 Colab notebook 工作流

這樣最清楚，也最不容易把資料混掉。
