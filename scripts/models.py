import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel


class PLTNUM(nn.Module):
    def __init__(self, cfg):
        super(PLTNUM, self).__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(cfg.model, config=self.config)

        self.fc_dropout1 = nn.Dropout(0.8)
        self.fc_dropout2 = nn.Dropout(0.4 if cfg.task == "classification" else 0.8)
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


class PLTNUM_PreTrainedModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config, cfg):
        super(PLTNUM_PreTrainedModel, self).__init__(config)
        self.cfg = cfg
        self.model = AutoModel.from_pretrained(self.config._name_or_path)

        self.fc_dropout1 = nn.Dropout(0.8)
        self.fc_dropout2 = nn.Dropout(0.4 if cfg.task == "classification" else 0.8)
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


class LSTMModel(nn.Module):
    def __init__(self, cfg):
        super(LSTMModel, self).__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=21,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.fc_dropout = nn.Dropout(0.8)
        self.fc = nn.Linear(256 * 2, 1)
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
        outputs, _ = self.lstm(inputs)
        last_hidden_state = outputs[:, -1, :]
        output = self.fc(self.fc_dropout(last_hidden_state))
        return output
