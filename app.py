# -*- coding: utf-8 -*-
"""ELECTRA-small-discriminator 判别器 WebUI 演示（不加载真实模型权重）。"""
from __future__ import annotations

import gradio as gr


def fake_load_model():
    """模拟加载模型，实际不下载权重，仅用于界面演示。"""
    return "模型状态：ELECTRA-small-discriminator 已就绪（演示模式，未加载真实权重）"


def fake_discriminate(text: str) -> str:
    """模拟对输入文本进行替换词判别并返回可视化描述。"""
    if not (text or "").strip():
        return "请输入待判别的英文句子。"
    words = text.strip().split()
    n = len(words)
    lines = [
        "[演示] 已对输入进行替换词判别（未加载真实模型）。",
        f"输入共 {n} 个 token，判别结果说明（占位）：",
        "",
    ]
    for i, w in enumerate(words[: min(15, n)]):
        lines.append(f'  token_{i+1}: "{w}" → 原始概率: 0.9x (演示)')
    if n > 15:
        lines.append(f"  ... 其余 {n - 15} 个 token 省略")
    lines.append("")
    lines.append("加载真实 ELECTRA discriminator 后，将在此显示各位置为「原始词」的置信度，用于检测被替换的 token。")
    return "\n".join(lines)


def build_ui():
    with gr.Blocks(title="ELECTRA-small-discriminator WebUI") as demo:
        gr.Markdown("## ELECTRA-small-discriminator 判别器 · WebUI 演示")
        gr.Markdown(
            "本界面以交互方式展示 ELECTRA 判别器（Discriminator）的典型使用流程，"
            "包括模型加载状态与输入句子的替换词判别（Replaced Token Detection）结果展示。"
        )

        with gr.Row():
            load_btn = gr.Button("加载模型（演示）", variant="primary")
            status_box = gr.Textbox(label="模型状态", value="尚未加载", interactive=False)
        load_btn.click(fn=fake_load_model, outputs=status_box)

        with gr.Tabs():
            with gr.Tab("替换词判别"):
                gr.Markdown(
                    "在下方输入英文句子，判别器将逐 token 预测该位置是「原始词」还是「被替换词」，"
                    "可用于预训练质量评估或下游任务的表示抽取。"
                )
                inp = gr.Textbox(
                    label="输入句子",
                    placeholder="例如：The quick brown fox jumps over the lazy dog.",
                    lines=4,
                )
                out = gr.Textbox(label="判别结果说明", lines=12, interactive=False)
                run_btn = gr.Button("执行判别（演示）")
                run_btn.click(fn=fake_discriminate, inputs=inp, outputs=out)

        gr.Markdown(
            "---\n*说明：当前为轻量级演示界面，未实际下载与加载 ELECTRA-small-discriminator 模型参数。*"
        )

    return demo


def main():
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=8760, share=False)


if __name__ == "__main__":
    main()
