"""
6S 真实场景相似度测试 - 模拟固定机位 ROI 区域物体在场/不在场
"""
import os
import sys
import numpy as np
import cv2
import onnxruntime as ort

MODEL_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "dinov2_vits14.onnx"))
TEST_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "电脑测试"))

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def preprocess(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224))
    normalized = (resized.astype(np.float32) - MEAN) / STD
    blob = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
    return blob


def extract(session, input_name, output_name, image):
    blob = preprocess(image)
    out = session.run([output_name], {input_name: blob})[0]
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / (norm + 1e-8)


def cosine_sim(emb1, emb2):
    return float(np.dot(emb1.flatten(), emb2.flatten()))


def load_image(filename):
    path = os.path.join(TEST_DIR, filename)
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    return img


def main():
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log(f"模型: {MODEL_PATH}")
    log(f"测试目录: {TEST_DIR}")
    log()

    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 测试场景定义
    scenarios = [
        ("baseline.png", "基准图（电脑在场）"),
        ("present_another_frame.png", "在场另一帧（轻微角度/光照变化）"),
        ("removed_only_background.png", "物体移走只剩背景"),
        ("occluded_by_person.png", "有人站在电脑前遮挡"),
        ("present_with_clutter.png", "物体在但附近有杂物"),
    ]

    # 加载图片
    images = {}
    log("=== 图片信息 ===")
    for fname, desc 在 scenarios:
        img = load_image(fname)
        if img is not None:
            images[desc] = img
            log(f"  {desc}: {img.shape[1]}x{img.shape[0]}")
        else:
            log(f"  {desc}: 读取失败!")
    log()

    # 提取特征
    log("=== 全图特征提取 ===")
    embs = {}
    for desc, img in images.items():
        emb = extract(session, input_name, output_name, img)
        embs[desc] = emb
        log(f"  {desc}: shape={emb.shape}")
    log()

    # 对比
    base_name = "基准图（电脑在场）"
    base = embs[base_name]

    comparisons = [
        ("在场另一帧（轻微角度/光照变化）", "同一物体在场（轻微角度/光照变化）"),
        ("物体移走只剩背景", "物体被移走后只剩背景"),
        ("有人站在电脑前遮挡", "有人站在 ROI 前遮挡物体"),
        ("物体在但附近有杂物"， "物体在但 ROI 里多了无关杂物"),
    ]

    log("=" * 60)
    log("全图对比结果:")
    log("=" * 60)
    results = {}
    for target_name, desc in comparisons:
        if target_name in embs:
            sim = cosine_sim(base, embs[target_name])
            results[target_name] = sim
            log(f"  {desc}")
            log(f"    基准 vs {target_name} = {sim:.4f}")
            log()

    # 汇总
    sim_present = results.get("在场另一帧（轻微角度/光照变化）", 0)
    sim_absent = results.get("物体移走只剩背景", 0)
    sim_occluded = results.get("有人站在电脑前遮挡", 0)
    sim_clutter = results.get("物体在但附近有杂物", 0)

    gap_present_absent = sim_present - sim_absent
    gap_present_occluded = sim_present - sim_occluded
    gap_present_clutter = sim_present - sim_clutter

    log("=" * 60)
    log("汇总分析:")
    log("=" * 60)
    log(f"  在场另一帧 相似度: {sim_present:.4f}")
    log(f"  物体移走   相似度: {sim_absent:.4f}")
    log(f"  有人遮挡   相似度: {sim_occluded:.4f}")
    log(f"  附近杂物   相似度: {sim_clutter:.4f}")
    log()
    log(f"  核心落差（在场 vs 移走）: {gap_present_absent:.4f}")
    log(f"  在场 vs 遮挡落差: {gap_present_occluded:.4f}")
    log(f"  在场 vs 杂物落差: {gap_present_clutter:.4f}")
    log()

    log("=" * 60)
    log("阈值建议与方案可行性:")
    log("=" * 60)

    if sim_present > sim_absent:
        mid = (sim_present + sim_absent) / 2
        safe_low = sim_absent + gap_present_absent * 0.1
        safe_high = sim_present - gap_present_absent * 0.1
        log(f"  安全区间: ({sim_absent:.4f}, {sim_present:.4f})")
        log(f"  推荐阈值（中点）: {mid:.4f}")
        log(f"  保守阈值（靠近在场端）: {safe_high:.4f}")
        log()

        if gap_present_absent > 0.15:
            log(f"  结论: 落差 {gap_present_absent:.4f} > 0.15，阈值好定，方案可行")
        elif gap_present_absent > 0.1:
            log(f"  结论: 落差 {gap_present_absent:.4f}，中等，阈值需现场微调")
        else:
            log(f"  结论: 落差 {gap_present_absent:.4f} < 0.1，落差较小")
            log(f"  建议: ROI 需要紧贴物体本体，减少背景占比")
    else:
        log(f"  警告: 不在场相似度({sim_absent:.4f}) >= 在场相似度({sim_present:.4f})")
        log(f"  全图方案可能不可行，需缩小 ROI 紧贴物体")

    # 保存日志
    log_path = os.path.join(os.path.dirname(__file__), "real_scenario_test.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\n日志已保存: {log_path}")


if __name__ == "__main__":
    main()
