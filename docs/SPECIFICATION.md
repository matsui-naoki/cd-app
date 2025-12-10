# CD Analyzer 機能仕様書

## 概要

本仕様書は、電池充放電データ解析・可視化アプリケーション「CD Analyzer」の実装すべき機能を定義します。
BioLogic EC-Lab の測定データ（.mpt, .mpr, .mps）を読み込み、学会・論文に使用可能な高品質な図を出力することを目指します。

---

## 1. データ入力・解析機能

### 1.1 対応ファイル形式

| 形式 | 説明 | 対応状況 |
|------|------|----------|
| `.mpt` | BioLogic テキスト形式 | 実装済み |
| `.mpr` | BioLogic バイナリ形式 | 実装済み |
| `.mps` | BioLogic 設定ファイル（測定シーケンス） | 実装済み |

### 1.2 測定手法の自動判別

ファイルヘッダーまたは `.mps` ファイルから測定手法を自動判別：

| 手法 | 識別子 | Igor実装 | 現状 |
|------|--------|----------|------|
| GCPL (Galvanostatic Cycling) | `Galvanostatic Cycling with Potential Limitation` | ✅ | ✅ |
| CCCV (定電流定電圧) | `Modulo Bat` | ✅ | 部分的 |
| PEIS (インピーダンス) | `Potentio Electrochemical Impedance Spectroscopy` | ✅ | ✅ |
| GEIS | `Galvanostatic Electrochemical Impedance Spectroscopy` | ✅ | 未実装 |
| OCV (開回路電圧) | `Open Circuit Voltage` | ✅ | ✅ |
| CV (サイクリックボルタンメトリー) | `Cyclic Voltammetry` | - | 未実装 |
| CC (定電流) | `Constant Current` | ✅ | 未実装 |
| BCD | Battery Capacity Determination | ✅ | 未実装 |

### 1.3 データ前処理

#### 1.3.1 Ns (Number of Sequence) によるデータ分割
- **機能**: Ns値の変化点でデータを分割し、各半サイクル（充電/放電）を独立した波形として取得
- **Igor実装**: `CCD_loadDataBiologic_for_UL_split()` - Ns changes フラグを監視
- **用途**: サイクルごとの充放電曲線プロット、dQ/dV分析

#### 1.3.2 容量計算
- **機能**: 電流・時間データから容量を計算
- **単位変換**:
  - `mAh/g` = 容量(mAh) / 活物質重量(g)
  - `mAh/cm²` = 容量(mAh) / 電極面積(cm²)
- **Igor実装**: `CCCV_loadDataBiologic_for_UL_for_split_refactoring()` で `mass_of_active_material` を使用

#### 1.3.3 電流密度正規化
- **機能**: 電流を電極面積で割り、電流密度 (mA/cm²) に変換
- **Igor実装**: `wave_mA /= 0.63585` (φ9mmの面積)
- **必要パラメータ**: 電極面積 (cm²)

---

## 2. サンプル情報設定

### 2.1 必須パラメータ

| パラメータ | 単位 | 説明 | Igor変数 |
|------------|------|------|----------|
| 活物質重量 | mg | 正極活物質の重量 | `mass_of_cathode_mg` |
| 活物質比率 | - | 合剤中の活物質比率 (0-1) | `ratio_of_active_material` |
| 電極面積 | cm² | 電極の有効面積 | (固定値: 0.636 for φ9mm) |
| サンプル名 | - | グラフラベル用 | `sample_name_UL` |

### 2.2 オプションパラメータ

| パラメータ | 単位 | 説明 |
|------------|------|------|
| 理論容量 | mAh/g | C-rate計算用 |
| 電池容量 | Ah | セル設計用 |
| 温度 | °C | 測定条件記録 |
| 追加情報 | - | ラベル用 (例: "Torque_2Nm") |

---

## 3. 可視化機能

### 3.1 充放電曲線 (V-t プロット)

**目的**: 電圧と電流の時間変化を表示

| 軸 | データ | 単位 |
|----|--------|------|
| X軸 | 時間 | h |
| Y軸(左) | 電圧 | V |
| Y軸(右) | 電流密度 | mA/cm² |

**Igor実装機能**:
- `CCD_make_graph_UL()`: 複数ファイルの波形を一括プロット
- `CCD_Graph_for_UL_split()`: Nsごとに分割してプロット
- 電圧(赤系)と電流(青系)の色分け

**必要な改善**:
- [ ] サイクルごとの色分け (レインボーカラー)
- [ ] 電流密度を右Y軸に表示
- [ ] 時間軸のオフセット調整機能

### 3.2 充放電曲線 (V-Q プロット)

**目的**: 電圧と容量の関係を表示

| 軸 | データ | 単位 |
|----|--------|------|
| X軸 | 容量 | mAh/g または mAh/cm² |
| Y軸 | 電圧 | V |

**Igor実装機能**:
- `CCCV_make_graph_UL()`: 容量 vs 電圧プロット
- サイクルごとの分離表示
- Rest期間/PEIS期間のデータ除外

**必要な改善**:
- [ ] 充電/放電曲線の色分け (充電:赤、放電:青)
- [ ] サイクル選択機能 (特定サイクルのみ表示)
- [ ] 容量単位切り替え (mAh/g ↔ mAh/cm²)

### 3.3 dQ/dV プロット

**目的**: 微分容量曲線による反応電位の同定

| 軸 | データ | 単位 |
|----|--------|------|
| X軸 | 電圧 | V |
| Y軸 | dQ/dV | mAh/g·V または mAh·V⁻¹ |

**Igor実装機能**:
- `dQdV_UL()`: dQ/dVデータの読み込みと描画
- `CCD_loadDataBiologic_dQdV_UL()`: Nsごとにデータを分割
- `d(Q-Qo)/dE/mA.h/V` カラムの使用

**必要な改善**:
- [ ] スムージング強度の調整
- [ ] 充電/放電の分離表示
- [ ] ピーク自動検出・ラベリング

### 3.4 サイクル特性プロット

**目的**: 容量維持率とクーロン効率の推移を表示

| グラフ | X軸 | Y軸 |
|--------|-----|-----|
| 容量推移 | サイクル数 | 容量 (mAh/g) |
| クーロン効率 | サイクル数 | 効率 (%) |
| 容量維持率 | サイクル数 | 維持率 (%) |

**Igor実装機能**:
- `calculate_capacity_and_make_CCCV_graph_UL()`:
  - 充電容量・放電容量の計算
  - クーロン効率 = 放電容量/充電容量 × 100
  - 容量維持率 = 放電容量[n]/放電容量[1] × 100
- `calcRetention_UL()`: 80%, 70%到達サイクル数の計算
- `getCyclesAtRetentionOf()`: 指定維持率でのサイクル数取得

**必要な改善**:
- [ ] 充電/放電容量の両方表示
- [ ] 効率の右Y軸表示
- [ ] 80%/70%維持サイクル数の自動表示

### 3.5 EIS (インピーダンス) プロット

**目的**: 電気化学インピーダンスの解析

| グラフ | X軸 | Y軸 | 備考 |
|--------|-----|-----|------|
| ナイキスト | Z' (Ω·cm²) | -Z" (Ω·cm²) | アスペクト比 1:1 必須 |
| ボード (|Z|) | log(freq) | |Z| (Ω) | |
| ボード (δ) | log(freq) | Phase (°) | |

**Igor実装機能**:
- `PEIS_loadDataBiologic_for_UL()`: PEISデータの読み込み
- `create_Bode_plot_from_active_grpah()`: ボードプロット作成
  - `|Z| = sqrt(Re(Z)² + Im(Z)²)`
  - `δ = -atan(Im(Z)/|Z|) × 180/π`

**必要な改善**:
- [ ] ナイキストプロットのアスペクト比固定
- [ ] 複数測定の重ね描き
- [ ] 周波数ラベル表示

---

## 4. グラフスタイル設定

### 4.1 論文用スタイル仕様

**Igor実装** (`CCD_graph_style_TN_modified()`, `dQdV_graph_style_UL()`):

```
フォント: Helvetica, 12pt
グラフサイズ: 幅212pt × 高さ156pt
マージン: left=34, bottom=28, right=45, top=6
軸: tick内向き, ミラー有り, btLen=4
軸ラベル位置: lblPos(left)=35
```

### 4.2 スタイルオプション

| 項目 | オプション | デフォルト |
|------|------------|------------|
| 線幅 | 0.5-3 pt | 1 pt |
| マーカー | ○●□■△▲ etc. | - |
| マーカーサイズ | 1-10 | 2 |
| カラーパレット | Inocolor, Rainbow等 | Inocolor |
| 背景色 | 白/透明 | 白 |

### 4.3 軸ラベル仕様

| 測定 | X軸ラベル | Y軸ラベル |
|------|-----------|-----------|
| V-t | Time / h | Voltage *V* / V |
| V-Q | Capacity *C* / mAh g⁻¹ | Voltage *V* / V |
| dQ/dV | Voltage *V* / V | d*Q*·d*V*⁻¹ / mAh·V⁻¹ |
| サイクル | Cycle number | Capacity *C* / mAh g⁻¹ |
| EIS | *Z*' / Ω·cm² | −*Z*" / Ω·cm² |

---

## 5. グラフ操作機能

### 5.1 軸範囲操作

**Igor実装機能**:
- `ButtonProc_CCD_UL_autorange_Y()`: Y軸オートレンジ
- `ButtonProc_CCD_UL_autorange_X()`: X軸オートレンジ
- `ButtonProc_expand_Y()`: Y軸範囲を1%拡張
- `ButtonProc_shift_X_axis_CCC_UL()`: X軸シフト
- `load_range()`: カーソル間の範囲を読み取り

### 5.2 トレース操作

**Igor実装機能**:
- `ButtonProc_hid_UL()`: 選択トレースを非表示
- `ButtonProcShowAll_UL()`: 全トレース表示
- `ButtonProc_rmv_right_traces_UL()`: 右軸トレース削除
- `InoColors()`: カラーパレット適用
- `change_marker_UL()`: マーカー変更

### 5.3 アノテーション

**Igor実装機能**:
- `ButtonProc_add_legend_CCD_UL()`: 凡例追加
- `ButtonProc_add_tag_UL()`: データポイントにタグ追加
- `ButtonProc_load_annotation()`: アノテーションパネル読み込み
- サンプル名・電流密度の自動ラベリング

---

## 6. データエクスポート

### 6.1 グラフエクスポート

| 形式 | 用途 | 対応状況 |
|------|------|----------|
| PNG | プレゼンテーション | 実装予定 |
| SVG | 論文投稿 (ベクター) | 実装予定 |
| PDF | 印刷用 | 実装予定 |

**解像度設定**: 300 dpi以上 (印刷品質)

### 6.2 データエクスポート

| 形式 | 内容 | 対応状況 |
|------|------|----------|
| CSV | 生データ・計算結果 | 実装済み |
| Igor Text (.itx) | Igor Pro用波形データ | 実装済み |
| Excel | サマリーテーブル | 未実装 |

### 6.3 レポート出力

- サイクル容量一覧表
- 効率・維持率サマリー
- 測定条件情報

---

## 7. 実装優先度

### Phase 1: 必須機能 (高優先度)

1. **サンプル情報入力UI**
   - 活物質重量、比率、電極面積の入力
   - 容量単位の選択 (mAh/g or mAh/cm²)

2. **V-Q プロット強化**
   - サイクル選択機能
   - 充電/放電の色分け

3. **サイクル特性プロット**
   - 容量 vs サイクル数
   - クーロン効率 vs サイクル数
   - 容量維持率の計算・表示

4. **dQ/dV プロット**
   - Nsによるデータ分割
   - スムージング調整

### Phase 2: 重要機能 (中優先度)

5. **グラフスタイル設定**
   - 論文用プリセット
   - 軸ラベルのフォーマット

6. **軸操作機能**
   - オートレンジ
   - マニュアル範囲設定
   - 軸シフト

7. **図のエクスポート**
   - SVG/PNG出力
   - 解像度設定

### Phase 3: 追加機能 (低優先度)

8. **複数ファイル比較**
   - オーバーレイプロット
   - 差分表示

9. **EIS強化**
   - ボードプロット
   - 等価回路フィッティング (eis-app連携)

10. **バッチ処理**
    - フォルダ内全ファイル処理
    - レポート自動生成

---

## 8. UI設計方針

### 8.1 ワークフロー

```
1. ファイルアップロード
   ↓
2. 測定手法の自動判別 (または手動選択)
   ↓
3. サンプル情報入力
   ↓
4. プロットタイプ選択
   ↓
5. グラフ生成・調整
   ↓
6. エクスポート
```

### 8.2 サイドバー構成

```
📁 データ
  - ファイルアップロード
  - ファイル一覧
  - 測定手法表示

⚙️ サンプル情報
  - 活物質重量
  - 活物質比率
  - 電極面積
  - 容量単位選択

📊 表示設定
  - プロットタイプ
  - サイクル選択
  - スムージング

🎨 スタイル
  - カラーパレット
  - 線幅/マーカー
  - 軸ラベル

💾 エクスポート
  - 図の保存
  - データ保存
```

---

## 付録A: Igor IPF 関数一覧

### データ読み込み関連
- `CCD_loadDataBiologic_for_UL()` - CCD用mptファイル読み込み
- `CCD_loadDataBiologic_for_UL_split()` - Ns分割読み込み
- `CCCV_loadDataBiologic_for_UL_for_split_refactoring()` - CCCV用読み込み
- `PEIS_loadDataBiologic_for_UL()` - PEIS用読み込み
- `dQdV_loadDataBiologic_UL()` - dQ/dV用読み込み
- `CCD_GetHeaderLine_BioLogic_UL()` - ヘッダー解析

### グラフ作成関連
- `CCD_make_graph_UL()` - V-tグラフ作成
- `CCCV_make_graph_UL()` - V-Qグラフ作成
- `dQdV_graph_UL_all()` - dQ/dVグラフ作成
- `calculate_capacity_and_make_CCCV_graph_UL()` - サイクル特性グラフ
- `create_Bode_plot_from_active_grpah()` - ボードプロット作成

### スタイル関連
- `CCD_graph_style_TN_modified()` - CCDスタイル適用
- `CCCV_graph_style_UL()` - CCCVスタイル適用
- `dQdV_graph_style_UL()` - dQ/dVスタイル適用
- `InoColors()` - カラーパレット適用

### 計算関連
- `calculate_charge_discharge_capacity_UL()` - 充放電容量計算
- `calcRetention_UL()` - 容量維持率計算
- `getCyclesAtRetentionOf()` - 指定維持率でのサイクル数

---

## 付録B: 用語集

| 用語 | 説明 |
|------|------|
| Ns | Number of Sequence - 測定シーケンス番号 |
| GCPL | Galvanostatic Cycling with Potential Limitation |
| CCCV | Constant Current Constant Voltage |
| PEIS | Potentio Electrochemical Impedance Spectroscopy |
| dQ/dV | 微分容量 (Differential Capacity) |
| CE | Coulombic Efficiency (クーロン効率) |
| ASR | Area Specific Resistance (面積比抵抗) |

---

*仕様書バージョン: 1.0*
*作成日: 2024-12-10*
