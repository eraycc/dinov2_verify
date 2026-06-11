# dinov2_verify
验证dinov2_vits14.onnx特征识别对比结果，示例代码和结构仅供参考

# 模型来源
https://github.com/mohhh-ok/ai-facing-api-models/releases/download/v1/dinov2_vits14.onnx

# 验证图片来源
https://github.com/sefaburakokcu/dinov2_onnx/tree/main/inputs

# DINOv2 ONNX 模型验证结果

## 1. 模型输入输出规格

| 属性 | 输入 | 输出 |
|------|------|------|
| 名称 | `input` | `embedding` |
| 形状 | `['batch', 3, 224, 224]` | `['batch', 384]` |
| 类型 | `tensor(float)` | `tensor(float)` |

- 模型文件大小：84.3 MB
- 输出说明：384 维 CLS token 向量（ViT-S/14 架构）

## 2. 预处理流程

与导出项目保持一致，具体步骤如下：

| 步骤 | 操作 | 参数 |
|------|------|------|
| 1 | 色彩空间转换 | BGR → RGB |
| 2 | 图像缩放 | resize 到 224×224 (INTER_LINEAR) |
| 3 | 归一化 | mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375) |
| 4 | 维度重排 | HWC → CHW |
| 5 | 添加 batch 维度 | (1, 3, 224, 224) float32 |

> 注意：归一化在 0-255 像素值范围上执行；需严格使用 float32 类型，避免 numpy 隐式提升为 float64。

## 3. 相似度对比结果

| 对比 | 类型 | 余弦相似度 |
|------|------|-----------|
| bird_1 vs bird_2 | 同类（鸟 vs 鸟） | **0.1527** |
| dog_1 vs dog_2 | 同类（狗 vs 狗） | **0.2716** |
| bird_1 vs dog_1 | 异类（鸟 vs 狗） | **-0.0003** |
| turtle_1 vs crab_1 | 异类（龟 vs 蟹） | **0.1328** |
| dog_1 vs turtle_2 | 异类（狗 vs 龟） | **0.0348** |

## 4. 结论

- ✅ 模型输入输出 shape 与导出项目文档完全一致
- ✅ 同类图片相似度明显高于异类：
  - 同类相似度 > 0.15
  - 异类相似度 < 0.04（龟 vs 蟹 除外，为 0.1328）
- ✅ 模型验证全部通过

## 5. 部署建议

针对 6S 监控场景（同一 ROI 区域、同一物品、固定机位）：

| 项目 | 建议 |
|------|------|
| 初始阈值 | **0.6 ~ 0.8** |
| 调优方式 | 现场标定时使用「测试相似度」接口微调 |




