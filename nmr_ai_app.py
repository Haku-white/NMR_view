import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import tempfile
import os
import pickle
import numpy as np
import scipy

# nmrglueの古い依存関係(scipy.pi)エラーに対する一時的なモンキーパッチ
try:
    scipy.pi = np.pi
    scipy.fftpack = scipy.fft
except Exception:
    pass

try:
    import nmrglue as ng
except ImportError:
    st.error("nmrglueライブラリがインストールされていません。")
    st.stop()

st.set_page_config(page_title="1H-NMR Data Visualization", layout="wide")

st.markdown("""
<style>
    div[data-testid="stFileUploader"] section {
        padding: 5rem 2rem;
        min-height: 250px;
        border-width: 2px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("#### JEOL <sup>1</sup>H-NMR スペクトル 可視化・比較ツール", unsafe_allow_html=True)
st.caption("JEOLのRAWデータ(.jdf)を読み込み、複数のNMRスペクトルを重ねて描画・比較します。")
st.divider()

import re

def get_file_info(filename):
    base = filename[:-4] if filename.lower().endswith(".jdf") else filename
    # 例: _proton-1-1 などを検出 (グループ2が最後のn)
    match = re.search(r'_proton-(\d+)-(\d+)$', base, re.IGNORECASE)
    is_fid = False
    series_name = base
    if match:
        if int(match.group(2)) == 1:
            is_fid = True
        series_name = base[:match.start()]
    return series_name, is_fid

def downsample_minmax(df, x_col, y_col, target_points):
    """
    データポイント数を削減しつつ、ピークトップ(最大・最小)の形状を保持するデシメーション。
    """
    n_original = len(df)
    n_chunks = max(1, target_points // 2)
    chunk_size = max(1, n_original // n_chunks)
    trunc_length = (n_original // chunk_size) * chunk_size

    y_values = df[y_col].values[:trunc_length]
    y_reshaped = y_values.reshape(-1, chunk_size)

    max_idx_rel = np.argmax(y_reshaped, axis=1)
    min_idx_rel = np.argmin(y_reshaped, axis=1)

    base_idx = np.arange(0, trunc_length, chunk_size)
    max_idx_abs = base_idx + max_idx_rel
    min_idx_abs = base_idx + min_idx_rel

    selected_indices = np.unique(np.concatenate([max_idx_abs, min_idx_abs]))
    selected_indices.sort()

    return df.iloc[selected_indices].reset_index(drop=True)

@st.cache_data(show_spinner=False)
def parse_nmr_file(file_content, filename, apply_fft=False):
    """
    メモリ上のバイナリデータ(.jdf)をnmrglueで読み込み、
    Chemical Shift (ppm) と Intensity の DataFrame を返す
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jdf") as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        dic, data = ng.jeol.read(tmp_path)

        try:
            os.remove(tmp_path)
        except:
            pass

        udic = ng.jeol.guess_udic(dic, data)
        # 1Dデータであることを確認
        if udic['ndim'] != 1:
            raise ValueError("1DのNMRデータのみ対応しています。")

        if apply_fft:
            # 強制的にFIDとして前処理(窓関数->ゼロ詰め->FT->自動位相補正)を行う
            if np.iscomplexobj(data):
                data = ng.process.proc_base.em(data)
                original_size = udic[0].get('size', data.shape[-1])
                new_size = original_size * 2
                data = ng.proc_base.zf_size(data, new_size)

                uc = ng.fileio.fileiobase.unit_conversion(
                    new_size,
                    udic[0].get('complex', True),
                    udic[0].get('sw', 1000.0),
                    udic[0].get('obs', 400.0),
                    udic[0].get('car', 2000.0)
                )
                ppm_scale = uc.ppm_scale()

                data = ng.proc_base.fft(data)
                data = ng.process.proc_autophase.autops(data, 'acme')
                data = ng.proc_base.di(data)
                data = ng.proc_base.rev(data)
                intensity = data
            else:
                raise ValueError("実数データに対してフーリエ変換は実行できません。")
        else:
            # 既にFT済みのデータとして実部を取得
            if np.iscomplexobj(data):
                intensity = data.real
            else:
                intensity = data

            uc = ng.fileiobase.uc_from_udic(udic)
            ppm_scale = uc.ppm_scale()

        df = pd.DataFrame({
            "Chemical Shift [ppm]": ppm_scale,
            "Intensity": intensity
        })

        target_points = 15000
        if len(df) > target_points:
            df = downsample_minmax(df, "Chemical Shift [ppm]", "Intensity", target_points)

        return df

    except Exception as e:
        raise Exception(f"JDFファイルのパースに失敗しました: {e}")

@st.cache_data(show_spinner=False)
def normalize_data_by_peak(df, peak_min, peak_max):
    mask = (df["Chemical Shift [ppm]"] >= min(peak_min, peak_max)) & (df["Chemical Shift [ppm]"] <= max(peak_min, peak_max))
    sub_df = df[mask]

    if not sub_df.empty:
        max_val = sub_df["Intensity"].max()
        if max_val != 0:
            df["Intensity_Norm_Base"] = df["Intensity"] / max_val
        else:
            df["Intensity_Norm_Base"] = df["Intensity"]
    else:
        max_val = df["Intensity"].max()
        if max_val != 0:
            df["Intensity_Norm_Base"] = df["Intensity"] / max_val
        else:
            df["Intensity_Norm_Base"] = df["Intensity"]

    return df

# --- Main App ---
col_up1, col_up2 = st.columns(2)
with col_up1:
    uploaded_files = st.file_uploader("JEOL NMRデータファイル (.jdf) をドラッグ＆ドロップ", type=["jdf"], accept_multiple_files=True)
with col_up2:
    project_file = st.file_uploader("💾 プロジェクトファイルの復元 (.nmr)", type=["nmr"])

if not uploaded_files and project_file is None:
    st.markdown("""
    <style>
        div[data-testid="stFileUploader"] section {
            padding: 5rem 2rem;
            min-height: 250px;
            border-width: 2px;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        div[data-testid="stFileUploader"] section {
            padding: 1rem 1rem;
            min-height: 80px;
        }
    </style>
    """, unsafe_allow_html=True)

data_dict = {}

if project_file is not None:
    try:
        project_data = pickle.loads(project_file.getvalue())
        data_dict = project_data.get('data_dict', {})
        loaded_state = project_data.get('session_state', {})

        if st.session_state.get('last_loaded_project') != project_file.name:
            st.session_state.clear()
            for k, v in loaded_state.items():
                st.session_state[k] = v
            st.session_state['last_loaded_project'] = project_file.name
            st.rerun()
    except Exception as e:
        st.error(f"プロジェクトの読み込みに失敗しました: {e}")
elif uploaded_files:
    for file in uploaded_files:
        try:
            series_name, default_is_fid = get_file_info(file.name)

            # 初期状態の取得 (まだ設定されていなければファイル名から推定したフラグを使用)
            if f"apply_fft_{file.name}" not in st.session_state:
                st.session_state[f"apply_fft_{file.name}"] = default_is_fid

            apply_fft_flag = st.session_state[f"apply_fft_{file.name}"]

            file_bytes = file.getvalue()
            df_nmr = parse_nmr_file(file_bytes, file.name, apply_fft=apply_fft_flag)

            if not df_nmr.empty:
                data_dict[file.name] = {
                    'series_name': series_name,
                    'df_nmr': df_nmr
                }
            else:
                st.warning(f"{file.name}: スペクトルデータが抽出できませんでした。")
        except Exception as e:
            st.error(f"{file.name} の読み込みに失敗しました: {e}")

else:
    st.info("💡 データファイル (.jdf) または保存したプロジェクト (.nmr) をアップロードしてください。")

if data_dict:
    # --- Sidebar settings ---
    if st.sidebar.button("🔄 設定を初期状態に戻す"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.sidebar.divider()
    st.sidebar.header("📑 凡例・グラフの表示順")

    file_names = list(data_dict.keys())
    if 'ordered_files' not in st.session_state or set(st.session_state.ordered_files) != set(file_names):
        st.session_state.ordered_files = file_names

    st.sidebar.markdown("ファイルの順番を「▲」「▼」で入れ替え、✔で表示/非表示を切り替えられます。")

    ordered_files = []
    for i, filename in enumerate(st.session_state.ordered_files):
        col1, col2, col3 = st.sidebar.columns([6, 2, 2])
        with col1:
            is_checked = st.checkbox(data_dict[filename]['series_name'], value=st.session_state.get(f"vis_{filename}", True), key=f"vis_chk_{filename}")
            st.session_state[f"vis_{filename}"] = is_checked
            if is_checked:
                ordered_files.append(filename)
        with col2:
            if i > 0:
                if st.button("▲", key=f"up_{filename}"):
                    st.session_state.ordered_files[i-1], st.session_state.ordered_files[i] = st.session_state.ordered_files[i], st.session_state.ordered_files[i-1]
                    st.rerun()
        with col3:
            if i < len(st.session_state.ordered_files) - 1:
                if st.button("▼", key=f"down_{filename}"):
                    st.session_state.ordered_files[i+1], st.session_state.ordered_files[i] = st.session_state.ordered_files[i], st.session_state.ordered_files[i+1]
                    st.rerun()

    st.sidebar.header("🎨 グラフ全体設定")

    st.sidebar.subheader("横軸範囲の指定 (ppm)")
    col_x_min, col_x_max = st.sidebar.columns(2)
    with col_x_max:
        x_range_max = st.number_input("x左端 (最大値)", value=8.0, step=0.5, format="%.2f", key="ui_x_max")
    with col_x_min:
        x_range_min = st.number_input("x右端 (最小値)", value=-1.0, step=0.5, format="%.2f", key="ui_x_min")

    if x_range_min >= x_range_max:
        st.sidebar.error("左端(最大値)は右端(最小値)より大きい値を指定してください。")
        x_range_max = x_range_min + 1.0

    tick_interval = st.sidebar.number_input("横軸メモリの間隔", min_value=0.1, max_value=10.0, value=1.0, step=0.5, key="ui_tick_interval")

    st.sidebar.subheader("📏 データ標準化 (指定ピーク基準)")
    norm_enable = st.sidebar.checkbox("指定した範囲のピークで強度を揃える", value=False, key="norm_enable")
    norm_target_value = st.sidebar.number_input("基準となる強度 (揃える値)", value=0.5, step=0.2, format="%.2f", disabled=not norm_enable, key="ui_norm_target")
    col_norm_max, col_norm_min = st.sidebar.columns(2)
    with col_norm_max:
        norm_peak_max = st.number_input("ピーク範囲 Max", value=7.5, step=0.1, format="%.2f", disabled=not norm_enable, key="ui_norm_max")
    with col_norm_min:
        norm_peak_min = st.number_input("ピーク範囲 Min", value=7.0, step=0.1, format="%.2f", disabled=not norm_enable, key="ui_norm_min")

    st.sidebar.divider()

    st.sidebar.subheader("凡例の設定")
    show_legend_ui = st.sidebar.checkbox("グラフ右上の凡例を表示", value=True, key="ui_show_legend")

    st.sidebar.divider()
    with st.sidebar.expander("🛠 詳細設定 (表示範囲・位置・フォント・保存)"):
        st.subheader("縦軸(強度)の表示範囲指定")
        y_manual_range = st.checkbox("手動で縦軸(強度)の表示範囲を設定する", value=False, key="y_manual_range")
        col_y_min, col_y_max = st.columns(2)
        with col_y_max:
            y_range_max = st.number_input("y上端 (最大値)", value=1.2, step=0.1, format="%.2f", disabled=not y_manual_range, key="ui_y_max")
        with col_y_min:
            y_range_min = st.number_input("y下端 (最小値)", value=-0.1, step=0.1, format="%.2f", disabled=not y_manual_range, key="ui_y_min")

        st.divider()
        st.subheader("凡例の配置調整")
        legend_x = st.slider("凡例の横位置 (X座標)", min_value=0.0, max_value=1.1, value=st.session_state.get("legend_pos_x", 0.99), step=0.01, key="ui_legend_x")
        legend_y = st.slider("凡例の縦位置 (Y座標)", min_value=0.0, max_value=1.1, value=st.session_state.get("legend_pos_y", 0.99), step=0.01, key="ui_legend_y")

        st.divider()
        st.subheader("フォント設定")
        axis_title_font_size = st.number_input("軸タイトルの文字サイズ", min_value=10, max_value=40, value=20, step=1, key="ui_title_font")
        axis_tick_font_size = st.number_input("軸ラベル(数値)の文字サイズ", min_value=10, max_value=40, value=18, step=1, key="ui_tick_font")

        st.divider()
        show_yaxis = st.checkbox("縦軸（強度数値）を表示", value=False, key="ui_show_yaxis")

        st.divider()
        st.header("💾 保存設定")
        img_format = st.radio("カメラアイコンでの保存形式", ["svg", "png"], index=0, key="ui_img_format")
        col_w, col_h = st.columns(2)
        with col_w:
            export_width = st.number_input("出力幅 (px)", min_value=200, max_value=3000, value=800, step=50, key="ui_export_width")
        with col_h:
            export_height = st.number_input("出力高さ (px)", min_value=200, max_value=3000, value=400, step=50, key="ui_export_height")

    st.sidebar.divider()
    st.sidebar.header("📝 テキストメモ (アノテーション)")
    st.sidebar.markdown("グラフ上に任意のテキストを追加できます。")
    num_annotations = st.sidebar.number_input("追加するメモの数", min_value=0, max_value=10, value=0, step=1, key="ui_num_annotations")
    annotations = []
    for i in range(num_annotations):
        with st.sidebar.expander(f"メモ {i+1}"):
            show_ann = st.checkbox("このメモを表示する", value=True, key=f"ann_show_{i}")
            text = st.text_input(f"表示テキスト", value=f"Memo {i+1}", key=f"ann_text_{i}")
            font_size = st.number_input("文字サイズ", min_value=8, max_value=40, value=14, step=1, key=f"ann_s_{i}")
            color_ann = st.color_picker("文字色", value="#000000", key=f"ann_c_{i}")

            ann_x = st.slider("X座標（ppm）", min_value=float(x_range_min), max_value=float(x_range_max), value=float(x_range_max - (x_range_max - x_range_min)/2), step=0.1, key=f"ui_ann_x_{i}")
            ann_y = st.slider("Y座標 (高さ)", min_value=-0.5, max_value=10.0, value=st.session_state.get(f"ann_pos_y_{i}", 0.5), step=0.1, key=f"ui_ann_y_{i}")

            if show_ann:
                st.session_state[f"ann_pos_y_{i}"] = ann_y
                annotations.append({
                    'idx': i,
                    'text': text,
                    'x': ann_x,
                    'y': ann_y,
                    'font_size': font_size,
                    'color': color_ann
                })

    st.sidebar.header("⚙ 各データの設定 (色・Y軸オフセット)")

    file_settings = {}

    for i, filename in enumerate(ordered_files):
        data = data_dict[filename]
        series_name = data['series_name']
        with st.sidebar.expander(f"設定: {series_name}"):
            custom_legend_name = st.text_input("凡例の表示名", value=st.session_state.get(f"legend_name_{filename}", series_name), key=f"legend_input_{filename}")
            st.session_state[f"legend_name_{filename}"] = custom_legend_name

            apply_fft_current = st.session_state.get(f"apply_fft_{filename}", False)
            apply_fft_new = st.checkbox("未処理(FID)データとしてフーリエ変換を実行する", value=apply_fft_current, key=f"ui_fft_chk_{filename}")
            if apply_fft_new != apply_fft_current:
                st.session_state[f"apply_fft_{filename}"] = apply_fft_new
                st.rerun()

            color = st.color_picker("線の色", value="#000000", key=f"color_{filename}")

            scale_intensity = st.number_input("強度倍率 (スケール)", value=1.0, step=0.1, format="%.2f", key=f"scale_{filename}")

            max_y_offset = 20.0
            step_y = 0.1

            default_y = float(-i * 1.5)

            slider_key = f"y_offset_pos_{i}"
            if slider_key not in st.session_state:
                st.session_state[slider_key] = default_y

            y_offset = st.slider("Y軸 近似オフセット間隔", min_value=float(-max_y_offset), max_value=float(max_y_offset), value=float(st.session_state[slider_key]), step=float(step_y), key=slider_key)

            st.markdown("##### 📍 ピークトップ線(任意)")
            peak_enable = st.checkbox("ピークトップ（縦線）を表示", value=False, key=f"peak_enable_{filename}")
            peak_color = st.color_picker("線の色", value="#808080", key=f"peak_color_{filename}")
            peak_dash = st.selectbox("線の種類", ["solid", "dash", "dot", "dashdot"], index=0, key=f"peak_dash_{filename}")

            file_settings[filename] = {
                'color': color,
                'scale_intensity': scale_intensity,
                'y_offset': y_offset,
                'peak_enable': peak_enable,
                'peak_color': peak_color,
                'peak_dash': peak_dash
            }

    # --- Plotting ---
    st.markdown("---")
    st.subheader("📈 1H-NMR スペクトル")
    fig = go.Figure()

    global_y_min = float('inf')
    global_y_max = float('-inf')

    traces_to_add = []
    shapes_to_add = []

    for filename in ordered_files:
        data = data_dict[filename]
        df = data['df_nmr'].copy()

        mapped_series_name = st.session_state.get(f"legend_name_{filename}", data['series_name'])
        settings = file_settings[filename]

        if norm_enable:
            df = normalize_data_by_peak(df, float(norm_peak_min), float(norm_peak_max))
        else:
            df["Intensity_Norm_Base"] = df["Intensity"]

        x_min_val = float(x_range_min)
        x_max_val = float(x_range_max)

        df_sorted = df.sort_values("Chemical Shift [ppm]").reset_index(drop=True)
        x_col = df_sorted["Chemical Shift [ppm]"]
        y_col = df_sorted["Intensity_Norm_Base"]

        def interp_boundary(x_col, y_col, x_bound, side):
            if side == 'left': # 検索対象が X軸の小さい側（右側端点）
                idx_after = x_col.searchsorted(x_bound, side='right')
                if idx_after == 0 or idx_after >= len(x_col):
                    return None
                x0, x1 = x_col.iloc[idx_after - 1], x_col.iloc[idx_after]
                y0, y1 = y_col.iloc[idx_after - 1], y_col.iloc[idx_after]
            else:  # right （検索対象が X軸の大きい側, 左側端点）
                idx_before = x_col.searchsorted(x_bound, side='left') - 1
                if idx_before < 0 or idx_before + 1 >= len(x_col):
                    return None
                x0, x1 = x_col.iloc[idx_before], x_col.iloc[idx_before + 1]
                y0, y1 = y_col.iloc[idx_before], y_col.iloc[idx_before + 1]
            if x1 == x0:
                return None
            y_interp = y0 + (y1 - y0) * (x_bound - x0) / (x1 - x0)
            return pd.DataFrame({"Chemical Shift [ppm]": [x_bound], "Intensity_Norm_Base": [y_interp]})

        left_pt = interp_boundary(x_col, y_col, x_min_val, 'left')
        right_pt = interp_boundary(x_col, y_col, x_max_val, 'right')

        plot_mask = (df_sorted["Chemical Shift [ppm]"] >= x_min_val) & (df_sorted["Chemical Shift [ppm]"] <= x_max_val)
        df_inner = df_sorted[plot_mask]

        parts = [p for p in [left_pt, df_inner, right_pt] if p is not None and not p.empty]
        if parts:
            df_plot = pd.concat(parts, ignore_index=True).sort_values("Chemical Shift [ppm]").reset_index(drop=True)
        else:
            df_plot = pd.DataFrame(columns=["Chemical Shift [ppm]", "Intensity_Norm_Base"])

        if not df_plot.empty:
            x_data = df_plot["Chemical Shift [ppm]"]
            y_base = df_plot["Intensity_Norm_Base"]

            target_val = float(norm_target_value) if norm_enable else 1.0
            actual_y_offset = settings['y_offset'] * target_val

            y_data = (y_base * target_val * settings['scale_intensity']) + actual_y_offset

            if norm_enable:
                # 上限は「基準値=1.0」を前提とした固定値にし、下限は設定された基準値に連動させる
                # これにより、基準値を小さくした時に枠が縮むのではなく、グラフだけが小さく表示される
                view_max = 1.0 * settings['scale_intensity']
                view_min = -0.05 * target_val * settings['scale_intensity']
            else:
                view_max = y_base.max() * settings['scale_intensity']
                view_min = y_base.min() * settings['scale_intensity']

            # スケール・オフセットなどを反映した後の「最適フレーム」の上下限
            c_min, c_max = view_min + actual_y_offset, view_max + actual_y_offset

            if c_min < global_y_min: global_y_min = c_min
            if c_max > global_y_max: global_y_max = c_max

            peak_x = None
            mask = (x_data >= float(x_min_val)) & (x_data <= float(x_max_val))
            if any(mask):
                max_y_idx = df_plot.loc[mask, "Intensity_Norm_Base"].idxmax()
                peak_x = df_plot.loc[max_y_idx, "Chemical Shift [ppm]"]
                peak_y = y_data.loc[max_y_idx]

            traces_to_add.append(go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines',
                line_shape='spline',
                name=mapped_series_name,
                line=dict(color=settings['color'], width=1.5),
                cliponaxis=True,
                hoverlabel=dict(namelength=-1)
            ))

            if settings['peak_enable'] and peak_x is not None:
                shapes_to_add.append({
                    'x': peak_x,
                    'y': peak_y,
                    'color': settings['peak_color'],
                    'dash': settings['peak_dash']
                })

    if global_y_min == float('inf'):
        global_y_min = 0.0
        global_y_max = 1.0

    y_range_pad_top = (global_y_max - global_y_min) * 0.05
    y_range_pad_bottom = (global_y_max - global_y_min) * 0.05
    if y_range_pad_top == 0:
        y_range_pad_top = 0.1
        y_range_pad_bottom = 0.02

    if y_manual_range:
        y_axis_min = float(y_range_min)
        y_axis_max = float(y_range_max)
    else:
        y_axis_min = global_y_min - y_range_pad_bottom
        y_axis_max = global_y_max + y_range_pad_top

    for trace in traces_to_add:
        # SVGエクスポート時などにグラフが枠外へ飛び出さないよう、Y軸の上下限で物理的にデータをクリップ（上書き）する
        trace.y = np.clip(np.array(trace.y, dtype=float), a_min=y_axis_min, a_max=y_axis_max)
        fig.add_trace(trace)

    for shape in shapes_to_add:
        # ピークラインの縦線も上限・下限を超えないようにクリップ
        clipped_shape_y = max(y_axis_min, min(shape['y'], y_axis_max))
        fig.add_shape(
            type="line",
            x0=shape['x'], x1=shape['x'],
            y0=y_axis_min, y1=clipped_shape_y,
            yref="y", xref="x",
            line=dict(width=1.0, dash=shape['dash'], color=shape['color']),
            opacity=0.8
        )

    show_grid = False
    show_border = False
    show_legend_final = (len(data_dict) >= 2) and show_legend_ui

    st.session_state["legend_pos_x"] = legend_x
    st.session_state["legend_pos_y"] = legend_y

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=show_legend_final,
        legend=dict(
            yanchor="top", y=legend_y,
            xanchor="right", x=legend_x,
            font=dict(size=14, color="black", family="Arial, sans-serif"),
            bgcolor="rgba(255,255,255,0.7)"
        )
    )

    for trace in fig.data:
        trace.name = f"<b>{trace.name}</b>"

    for ann in annotations:
        fig.add_annotation(
            text=f"<b>{ann['text']}</b>",
            xref="x", yref="y",
            x=ann['x'], y=ann['y'],
            showarrow=False,
            font=dict(size=ann['font_size'], color=ann['color'])
        )

    x_ticks = np.arange(x_range_min, x_range_max + (tick_interval/1000), tick_interval)
    x_tick_texts = [f"<b>{val:g}</b>" for val in x_ticks]

    fig.update_xaxes(
        range=[x_range_max, x_range_min],
        tickmode='array',
        tickvals=x_ticks,
        ticktext=x_tick_texts,
        ticks='inside',
        tickcolor='black',
        tickwidth=2.5,
        ticklen=8,
        title_text="<b>Chemical Shift [ppm]</b>",
        title_font=dict(size=axis_title_font_size, color='black'),
        title_standoff=12,
        tickfont=dict(size=axis_tick_font_size, color='black'),
        automargin=True,
        showline=True,
        linecolor='black',
        linewidth=3.5,
        showgrid=show_grid,
        gridwidth=1,
        gridcolor='#e0e0e0',
        mirror=show_border,
        zeroline=False
    )

    fig.update_yaxes(
        range=[y_axis_min, y_axis_max],
        showticklabels=show_yaxis,
        title_text="" if not show_yaxis else "<b>Normalized Intensity</b>",
        showline=show_border,
        linewidth=1,
        linecolor='black',
        showgrid=show_grid,
        gridwidth=1,
        gridcolor='#e0e0e0',
        mirror=show_border,
        zeroline=False
    )

    fig.update_traces(cliponaxis=True)

    if len(data_dict) == 1:
        series_info_base = list(data_dict.values())[0]['series_name']
    else:
        series_names_list = [v['series_name'] for v in data_dict.values()]
        series_info_base = "_".join(series_names_list[:3])
        if len(series_names_list) > 3:
            series_info_base += "_etc"

    config = {
        'toImageButtonOptions': {
            'format': img_format,
            'filename': series_info_base,
            'height': export_height,
            'width': export_width,
            'scale': 1
        },
        'edits': {
            'annotationPosition': True,
            'annotationText': False,
            'legendPosition': True
        },
        'displaylogo': False,
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape']
    }

    st.plotly_chart(fig, use_container_width=True, config=config)

# --- プロジェクト保存 ---
if data_dict:
    st.sidebar.divider()
    st.sidebar.header("📦 プロジェクトの保存")

    def create_project_file():
        state_to_save = {k: v for k, v in st.session_state.items()
                         if not k.startswith("FormSubmitter")
                         and k != "last_loaded_project"
                         and not k.startswith("uploaded_")
                         and not k.startswith("up_")
                         and not k.startswith("down_")
                         and not k.startswith("btn_")}

        project_data = {
            'data_dict': data_dict,
            'session_state': state_to_save
        }
        return pickle.dumps(project_data)

    if len(data_dict) == 1:
        series_info = list(data_dict.values())[0]['series_name']
        dl_filename = f"{series_info}.nmr"
    else:
        series_names = [v['series_name'] for v in data_dict.values()]
        series_info = "_".join(series_names[:3])
        if len(series_names) > 3:
            series_info += "_etc"
        dl_filename = f"{series_info}.nmr"

    st.sidebar.download_button(
        label="📥 現在の状態を復元用ファイル(.nmr)として保存",
        data=create_project_file(),
        file_name=dl_filename,
        mime="application/octet-stream"
    )
