#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from typing import List
from PIL import Image
from vllm import LLM, SamplingParams
from flask import Flask, request, jsonify
import os
from transformers import AutoTokenizer

VISION_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
PLACEHOLDER = "<image>"

def replace_placeholders(prompt: str) -> str:
    return prompt.replace(PLACEHOLDER, VISION_TOKEN)

def load_images(image_paths: List[str]) -> List[Image.Image]:
    images = []
    for p in image_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image File Not found: {p}")
        with Image.open(p) as im:
            images.append(im.convert("RGB"))
    return images


def main():
    parser = argparse.ArgumentParser(description="infer server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--trust-remote-code", action='store_true')
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    print(f"[+] Loading Model from: {args.model}")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        limit_mm_per_prompt={"image": 5},
    )
    print("[+] Model successfully loaded.")

    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=0.7,
        top_p=0.9,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    app = Flask(__name__)

    @app.route('/generate_multimodal', methods=['POST'])
    def generate_multimodal():
        """
        ideal JSON input format:
        {
            "prompts": ["describe this image <image>", ...],
            "image_paths": [["/path/to/image1.jpg"], ...],
            "n": 1
        }
        """
        try:
            payload = request.get_json()
            if not payload:
                return jsonify({"error": "invalid JSON input"}), 400

            input_prompts = payload.get("prompts")
            input_images_list = payload.get("image_paths")
            n = payload.get("n", 1)

            if not all([isinstance(input_prompts, list), isinstance(input_images_list, list), len(input_prompts) == len(input_images_list)]):
                return jsonify({"error": "'prompts' and 'image_paths' must be lists with same length"}), 400
            

            vllm_inputs = []
            original_indices = []

            for i, (prompt_raw, image_paths) in enumerate(zip(input_prompts, input_images_list)):
                if prompt_raw.count(PLACEHOLDER) != len(image_paths):
                    return jsonify({"error": f"For prompt {i}, num of '<image>' placeholders isn't equal to the images"}), 400

                pil_images = load_images(image_paths) if image_paths else []
                prompt_for_model = replace_placeholders(prompt_raw)

                prompt_for_model = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_for_model}],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                for _ in range(n):
                    data = {"prompt": prompt_for_model}
                    if pil_images:
                        data["multi_modal_data"] = {"image": pil_images}
                    vllm_inputs.append(data)
                    original_indices.append(i)
            
            results = llm.generate(vllm_inputs, sampling_params)

            structured_outputs = [[] for _ in range(len(input_prompts))]
            for i, res in enumerate(results):
                original_prompt_index = original_indices[i]
                output_text = res.outputs[0].text if (res and res.outputs) else ""
                structured_outputs[original_prompt_index].append(output_text)

            return jsonify({"predictions": structured_outputs})

        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            print(f"[!] Server error: {repr(e)}")
            return jsonify({"error": f"Server error: {repr(e)}"}), 500

    print(f"[+] Flask Server running on http://0.0.0.0:{args.port}")
    app.run(host='0.0.0.0', port=args.port)

if __name__ == "__main__":
    main()