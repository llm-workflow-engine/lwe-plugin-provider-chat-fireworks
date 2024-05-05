# LLM Workflow Engine (LWE) Chat Fireworks Provider plugin

Chat Fireworks Provider plugin for [LLM Workflow Engine](https://github.com/llm-workflow-engine/llm-workflow-engine)

Access to [Fireworks](https://fireworks.ai/models) chat models.

## Installation

### Export API key

Grab a Fireworks API key from [https://fireworks.ai/api-keys](https://fireworks.ai/api-keys)

Export the key into your local environment:

```bash
export FIREWORKS_API_KEY=<API_KEY>
```

### From packages

Install the latest version of this software directly from github with pip:

```bash
pip install git+https://github.com/llm-workflow-engine/lwe-plugin-provider-chat-fireworks
```

### From source (recommended for development)

Install the latest version of this software directly from git:

```bash
git clone https://github.com/llm-workflow-engine/lwe-plugin-provider-chat-fireworks.git
```

Install the development package:

```bash
cd lwe-plugin-provider-chat-fireworks
pip install -e .
```

## Configuration

Add the following to `config.yaml` in your profile:

```yaml
plugins:
  enabled:
    - provider_chat_fireworks
    # Any other plugins you want enabled...
  # THIS IS OPTIONAL -- By default the plugin loads all model data via an API
  # call on startup. This does make startup time longer.
  # You can instead provide a 'models' object here with the relevant data, and
  # It will be used instead of an API call.
  provider_chat_fireworks:
    models:
      # 'id' parameter of the model as it appears in the API.
      "accounts/fireworks/models/llama-v3-8b-instruct":
        # The only parameter, and it's required.
        max_tokens: 8192
```

## Usage

From a running LWE shell:

```
/provider fireworks
/model model_name accounts/fireworks/models/llama-v3-8b-instruct
```
