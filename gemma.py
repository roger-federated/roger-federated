"""
Demonstration of how to deploy a custom model. All it requires is a standardized convenience wrapper, \
implementing `__init__`, `__call__`, and `get_params`.
"""

import kagglehub, onnxruntime as ort, os, numpy as np, onnx
from transformers import AutoConfig, AutoProcessor, GenerationConfig
from typing import Generator

class ConvenienceWrapper():
    def __init__(self, *args, **kwargs):
        """
        Initialize the onnx model's runtime. For portability, use absolute paths.
        """
        super().__init__(*args, **kwargs)
        ## Load config and processor
        self.model_path = kagglehub.model_download("google/gemma-4/onnx/gemma-4-e2b-it-onnx")
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        config = AutoConfig.from_pretrained(self.model_path)
        generation_config = GenerationConfig.from_pretrained(self.model_path)

        ## Load sessions
        self.vision_session = ort.InferenceSession(os.path.join(self.model_path, "onnx/vision_encoder.onnx"))
        self.audio_session = ort.InferenceSession(os.path.join(self.model_path, "onnx/audio_encoder.onnx"))
        self.embed_session = ort.InferenceSession(os.path.join(self.model_path, "onnx/embed_tokens_q4.onnx"))
        self.decoder_session = ort.InferenceSession(os.path.join(self.model_path, "onnx/decoder_model_merged_q4.onnx"))

        ## Set config values
        self.eos_token_id = generation_config.eos_token_id
        self.image_token_id = config.image_token_id
        self.audio_token_id = config.audio_token_id

    def __call__(self, messages) -> Generator[str, None, None]:
        """
        Return a generator that yields the model output. The input messages should be formatted according to the processor's expected JSON template.
        """
        ## Tokenize inputs
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=True
        )
        ## Extract some of the inputs
        attention_mask = inputs["attention_mask"].numpy()
        position_ids = np.cumsum(attention_mask, axis=-1) - 1
        input_tokens = inputs["input_ids"].numpy()
        ## Prepare decoder inputs
        batch_size = inputs["input_ids"].shape[0]
        num_logits_to_keep = np.array(1, dtype=np.int64)
        past_key_values = {
            inp.name: np.zeros(
                [batch_size, inp.shape[1], 0, inp.shape[3]],
                dtype=np.float32 if inp.type == "tensor(float)" else np.float16,
            )
            for inp in self.decoder_session.get_inputs()
            if inp.name.startswith("past_key_values")
        }

        ## Generation loop
        max_new_tokens = 1024
        generated_tokens = np.zeros((batch_size, 0), dtype=np.int64)
        image_features = None
        audio_features = None
        thinking = False
        for _ in range(max_new_tokens):
            ## Embed the input tokens
            inputs_embeds, per_layer_inputs = self.embed_session.run(None, {"input_ids": input_tokens})
            ## Insert image features into embedding
            if image_features is None and "pixel_values" in inputs:
                image_features = self.vision_session.run(["image_features"], {
                    "pixel_values": inputs["pixel_values"].numpy(),
                    "pixel_position_ids": inputs["image_position_ids"].numpy()
                })[0]
                mask = (input_tokens == self.image_token_id).reshape(-1)
                flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
                flat_embeds[mask] = image_features
                inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)
            ## Insert audio features into embedding
            if audio_features is None and "input_features" in inputs:
                audio_features = self.audio_session.run(["audio_features"], {
                    "input_features": inputs["input_features"].numpy().astype(np.float32), 
                    "input_features_mask": inputs["input_features_mask"].numpy()
                })[0]
                mask = (input_tokens == self.audio_token_id).reshape(-1)
                flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
                flat_embeds[mask] = audio_features
                inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)
            ## Decode embedded input
            logits, *present_key_values = self.decoder_session.run(None, dict(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                per_layer_inputs=per_layer_inputs,
                position_ids=position_ids,
                num_logits_to_keep=num_logits_to_keep,
                **past_key_values,
            ))
            ## Update values for next generation loop
            input_tokens = logits[:, -1].argmax(-1, keepdims=True)
            attention_mask = np.concatenate([attention_mask, np.ones_like(input_tokens)], axis=-1)
            position_ids = position_ids[:, -1:] + 1
            for j, key in enumerate(past_key_values):
                past_key_values[key] = present_key_values[j]
            ## Check whether eos is reached
            generated_tokens = np.concatenate([generated_tokens, input_tokens], axis=-1)
            if np.isin(input_tokens, self.eos_token_id).any():
                break
            ## Stream output
            out = self.processor.decode(input_tokens[0])
            if out=="<|channel>":
                thinking = True
            if not thinking:
                yield out
            if out=="<channel|>":
                thinking = False

    def get_params(self):
        model = onnx.load(os.path.join(self.model_path, "onnx/decoder_model_merged_q4.onnx"), load_external_data=True)
        params = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}
        return params