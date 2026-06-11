"""
DINOv2 ONNX 模型验证脚本
验证内容：
1. 打印模型输入/输出 shape
2. 跑通"在/不在"两张图的相似度对比
"""
import os
import sys
import numpy as np
import cv2
import onnxruntime as ort

# 模型路径
MODEL_PATH = os.path.join(os.path.dirname(__file__), "dinov2_vits14.onnx")
MODEL_PATH = os.path.normpath(MODEL_PATH)

# 测试图片目录
INPUTS_DIR = os.path.join(os.path.dirname(__file__), "inputs")
INPUTS_DIR = os.path.normpath(INPUTS_DIR)


def print_model_info(session: ort.InferenceSession):
    """打印模型输入输出信息"""
    print("=" * 60)
    print("模型输入信息:")
    for inp in session.get_inputs():
        print(f"  名称: {inp.name}")
        print(f"  形状: {inp.shape}")
        print(f"  类型: {inp.type}")
    print()
    print("模型输出信息:")
    for out in session.get_outputs():
        print(f"  名称: {out.name}")
        print(f"  形状: {out.shape}")
        print(f"  类型: {out.type}")
    print("=" * 60)


def preprocess_image(image: np.ndarray, input_size=(224, 224),
                     mean=(123.675, 116.28, 103.53),
                     std=(58.395, 57.12, 57.375)) -> np.ndarray:
    """
    DINOv2 预处理（与导出项目一致）
    - resize 到 224x224
    - BGR -> RGB
    - ImageNet 均值方差归一化（0-255 范围）
    - HWC -> CHW
    - 添加 batch 维度
    """
    # BGR -> RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # resize
    resized = cv2.resize(image_rgb, input_size, interpolation=cv2.INTER_LINEAR)
    # 归一化（均值方差在 0-255 范围）
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    normalized = (resized.astype(np.float32) - mean_arr) / std_arr
    # HWC -> CHW, 添加 batch 维度
    blob = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
    return blob


def extract_embedding(session: ort.InferenceSession, image: np.ndarray) -> np.ndarray:
    """提取单张图片的特征向量"""
    input_name = session.get_inputs()[0].name
    blob = preprocess_image(image)
    output = session.run(None, {input_name: blob})[0]
    return output


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """计算余弦相似度（L2 归一化后点积）"""
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    return float(np.dot(emb1_norm.flatten(), emb2_norm.flatten()))


def main():
    # 1. 加载模型
    print(f"模型路径: {MODEL_PATH}")
    if not os.path.isfile(MODEL_PATH):
        print(f"错误: 模型文件不存在!")
        sys.exit(1)

    print(f"模型文件大小: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.1f} MB")
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

    # 2. 打印模型信息
    print_model_info(session)

    # 3. 测试图片列表
    test_pairs = [
        ("bird_1.jpeg", "bird_2.jpeg", "同类（鸟 vs 鸟）"),
        ("dog_1.jpeg", "dog_2.jpeg", "同类（狗 vs 狗）"),
        ("bird_1.jpeg", "dog_1.jpeg", "异类（鸟 vs 狗）"),
        ("turtle_1.jpeg"， "crap_1.jpeg", "异类（龟 vs 蟹）"),
        ("dog_1.jpeg", "turtle_2.jpeg", "异类（狗 vs 龟）"),
    ]

    print("\n" + "=" * 60)
    print("相似度对比测试:")
    print("=" * 60)

    # 预提取所有特征
    image_files = set()
    for a, b, _ in test_pairs:
        image_files.add(a)
        image_files.add(b)

    embeddings = {}
    for fname in sorted(image_files):
        fpath = os.path.join(INPUTS_DIR, fname)
        if not os.path.isfile(fpath):
            print(f"  跳过: {fname} (文件不存在)")
            continue
        img = cv2.imread(fpath)
        if img is None:
            print(f"  跳过: {fname} (读取失败)")
            continue
        emb = extract_embedding(session, img)
        embeddings[fname] = emb
        print(f"  已提取特征: {fname} -> shape={emb.shape}")

    print()

    # 对比
    for img_a, img_b, desc in test_pairs:
        if img_a not in embeddings or img_b not in embeddings:
            print(f"  {desc}: 跳过（缺少图片）")
            continue
        sim = cosine_similarity(embeddings[img_a], embeddings[img_b])
        print(f"  {img_a} vs {img_b}")
        print(f"    {desc} -> 余弦相似度 = {sim:.4f}")

    print("\n验证完成!")


if __name__ == "__main__":
    main()
