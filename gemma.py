"""
Demonstration of how to deploy a custom model. All it requires is a standardized convenience wrapper, implementing \
`__init__`, `__call__`, and `get_params`.
"""

import kagglehub, onnxruntime as ort, os, numpy as np
from transformers import AutoConfig, AutoProcessor, GenerationConfig

class ConvenienceWrapper():
    def __init__(self, *args, **kwargs):
        """
        Initialize the onnx model's runtime. For portability, use absolute paths.
        """
        super().__init__(**args, **kwargs)
        ## Load config and processor
        MODEL_PATH = kagglehub.model_download("google/gemma-4/onnx/gemma-4-e2b-it-onnx")
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH)
        config = AutoConfig.from_pretrained(MODEL_PATH)
        generation_config = GenerationConfig.from_pretrained(MODEL_PATH)

        ## Load sessions
        self.vision_session = ort.InferenceSession(os.path.join(MODEL_PATH, "onnx/vision_encoder.onnx"))
        self.audio_session = ort.InferenceSession(os.path.join(MODEL_PATH, "onnx/audio_encoder.onnx"))
        self.embed_session = ort.InferenceSession(os.path.join(MODEL_PATH, "onnx/embed_tokens_q4.onnx"))
        self.decoder_session = ort.InferenceSession(os.path.join(MODEL_PATH, "onnx/decoder_model_merged_q4.onnx"))

        ## Set config values
        self.eos_token_id = generation_config.eos_token_id
        self.image_token_id = config.image_token_id
        self.audio_token_id = config.audio_token_id

    def __call__(self, messages):
        """
        Generate output based on input messages. The input messages should be formatted according to the processor's expected JSON template.
        """
        ## Tokenize inputs
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        )
        ## Extract some of the inputs
        attention_mask = inputs["attention_mask"].numpy()
        position_ids = np.cumsum(attention_mask, axis=-1) - 1
        input_ids = inputs["input_ids"].numpy()
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
        for i in range(max_new_tokens):
            ## Embed the input tokens
            inputs_embeds, per_layer_inputs = self.embed_session.run(None, {"input_ids": input_ids})
            ## Insert image features into embedding
            if image_features is None and "pixel_values" in inputs:
                image_features = self.vision_session.run(["image_features"], {
                    "pixel_values": inputs["pixel_values"].numpy(),
                    "pixel_position_ids": inputs["image_position_ids"].numpy()
                })[0]
                mask = (input_ids == self.image_token_id).reshape(-1)
                flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
                flat_embeds[mask] = image_features
                inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)
            ## Insert audio features into embedding
            if audio_features is None and "input_features" in inputs:
                audio_features = self.audio_session.run(["audio_features"], {
                    "input_features": inputs["input_features"].numpy().astype(np.float32), 
                    "input_features_mask": inputs["input_features_mask"].numpy()
                })[0]
                mask = (input_ids == self.audio_token_id).reshape(-1)
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
            input_ids = logits[:, -1].argmax(-1, keepdims=True)
            attention_mask = np.concatenate([attention_mask, np.ones_like(input_ids)], axis=-1)
            position_ids = position_ids[:, -1:] + 1
            for j, key in enumerate(past_key_values):
                past_key_values[key] = present_key_values[j]
            ## Check whether eos is reached
            generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
            if np.isin(input_ids, self.eos_token_id).any():
                break
            ## Stream output
            print(self.processor.decode(input_ids[0]), end="", flush=True)

    def get_params(self):
        pass