from transformers import AutoModel, AutoConfig
import torch.nn as nn

class PLTNUM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(
            cfg.model, output_hidden_states=True
        )
        self.model = AutoModel.from_pretrained(cfg.model, config=self.config)

        self.fc_dropout1 = nn.Dropout(0.8)
        self.fc_dropout2 = nn.Dropout(0.4 if cfg.task == "classification" else 0.8)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0][:, 0]
        output = (
            self.fc(self.fc_dropout1(last_hidden_states))
            + self.fc(self.fc_dropout2(last_hidden_states))
        ) / 2
        return output

    def create_embedding(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0][:, 0]
        return last_hidden_states
    

class LSTMModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
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
            module.weight.data.normal_(mean=0.0, std=0.05)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.05)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        outputs, _ = self.lstm(inputs)
        last_hidden_states = outputs[:, -1, :]
        output = self.fc(self.fc_dropout(last_hidden_states))
        return output