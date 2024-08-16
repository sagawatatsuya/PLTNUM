from torch import sigmoid
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel, AutoTokenizer


class PLTNUM_PreTrainedModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super(PLTNUM_PreTrainedModel, self).__init__(config)
        self.model = AutoModel.from_pretrained(self.config._name_or_path)

        self.fc_dropout1 = nn.Dropout(0.8)
        self.fc_dropout2 = nn.Dropout(0.4)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                nn.init.constant_(module.weight[module.padding_idx], 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state[:, 0]
        output = (
            self.fc(self.fc_dropout1(last_hidden_state))
            + self.fc(self.fc_dropout2(last_hidden_state))
        ) / 2
        return output

    def create_embedding(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state[:, 0]
        return last_hidden_state


model = PLTNUM_PreTrainedModel.from_pretrained("/home2/sagawa/PLTNUM/classification_ESM2_mouse_converted")
tokenizer = AutoTokenizer.from_pretrained("/home2/sagawa/PLTNUM/classification_ESM2_mouse_converted")
seq = "MSGRGKQGGKARAKAKTRSSRAGLQFPVGRVHRLLRKGNYSERVGAGAPVYLAAVLEYLTAEILELAGNAARDNKKTRIIPRHLQLAIRNDEELNKLLGRVTIAQGGVLPNIQAVLLPKKTESHHKPKGK"
input = tokenizer(
    [seq],
    add_special_tokens=True,
    max_length=512,
    padding="max_length",
    truncation=True,
    return_offsets_mapping=False,
    return_attention_mask=True,
    return_tensors="pt",
)
print(sigmoid(model(input)))  # tensor([[0.9798]], grad_fn=<SigmoidBackward0>)
