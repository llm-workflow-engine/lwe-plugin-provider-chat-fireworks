import os
import requests

from langchain_fireworks import ChatFireworks

from lwe.core.provider import Provider, PresetValue

FIREWORKS_API_BASE = "https://api.fireworks.ai/inference/v1"


class CustomChatFireworks(ChatFireworks):

    @property
    def _llm_type(self):
        """Return type of llm."""
        return "chat_fireworks"


class ProviderChatFireworks(Provider):
    """
    Access to Fireworks chat models.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.models = self.config.get('plugins.provider_chat_fireworks.models') or self.fetch_models()

    def fetch_models(self):
        models_url = f"{FIREWORKS_API_BASE}/models"
        try:
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}",
            }
            response = requests.get(models_url, headers=headers)
            response.raise_for_status()
            models_data = response.json()
            models_list = models_data.get('data')
            if not models_list:
                raise ValueError('Could not retrieve models')
            models = {model['id']: {'max_tokens': model['context_length']} for model in models_list if 'context_length' in model and model['context_length'] and 'supports_chat' in model and model['supports_chat']}
            return models
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Could not retrieve models: {e}")

    @property
    def default_model(self):
        pass

    @property
    def capabilities(self):
        return {
            "chat": True,
            'validate_models': False,
            "models": self.models,
        }

    def prepare_messages_method(self):
        return self.prepare_messages_for_llm_chat

    def llm_factory(self):
        return CustomChatFireworks

    # TODO: Remove this when the 'required' validation issue is resolved
    def transform_tool(self, tool):
        import copy
        tool = copy.deepcopy(tool)
        del tool['required']
        return tool

    def customization_config(self):
        return {
            "verbose": PresetValue(bool),
            "model_name": PresetValue(str, options=self.available_models),
            "temperature": PresetValue(float, min_value=0.0, max_value=2.0),
            "fireworks_api_base": PresetValue(str, include_none=True),
            "fireworks_api_key": PresetValue(str, include_none=True, private=True),
            "request_timeout": PresetValue(int),
            "n": PresetValue(int, 1, 10),
            "max_tokens": PresetValue(int, include_none=True),
            "tools": None,
            "tool_choice": None,
            "model_kwargs": {
                "top_p": PresetValue(float, min_value=0.0, max_value=1.0),
                'top_k': PresetValue(int, min_value=1, max_value=40),
                "presence_penalty": PresetValue(float, min_value=-2.0, max_value=2.0),
                "frequency_penalty": PresetValue(float, min_value=-2.0, max_value=2.0),
                "logit_bias": dict,
                "logprobs": PresetValue(bool, include_none=True),
                "top_logprobs": PresetValue(int, min_value=0, max_value=20, include_none=True),
                "response_format": dict,
                "stop": PresetValue(str, include_none=True),
                "user": PresetValue(str),
            },
        }
