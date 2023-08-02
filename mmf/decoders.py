import torch
from mmf.registry import registry
from torch import nn
from torch.nn.utils.weight_norm import weight_norm

class VisDiaDiscriminator(nn.Module):
    def __init__(self, config, embedding):
        super().__init__()
        self.config = config
        self.embedding = embedding

        self.emb_out_dim = embedding.text_out_dim
        self.hidden_dim = self.config.hidden_dim

        self.projection_layer = nn.Linear(self.emb_out_dim, self.hidden_dim)

    def forward(self, encoder_output, batch):
        answer_options_len = batch['answer_options_len']

        # BATCH_SIZE X DIALOGUES X 100 X SEQ_LEN
        answer_options = batch["answer_options"]

        max_seq_len = answer_options.size(-1) #맨 끝 dimension의 크기

        batch_size, ndialogues, noptions, seq_len = answer_options.size()

        # (B X D X 100) X SEQ_LEN
        answer_options = answer_options.view(-1, max_seq_len) #tensor.view(2, 3) 하면 (2, 3)으로 자료형 사이즈가 변경됨, tensor.view(-1, x)하면 (?, x)로 변경됨, 즉, 첫번째 차원은 내가 잘 모르겠는데 두번재 차원의 길이는 x로
        answer_options_len = answer_options_len.view(-1) #-1로 하면 다른 차원으로부터 해당 값을 유추함

        # (B X D X 100) X EMB_OUT_DIM
        answer_options = self.embedding(answer_options)

        # (B X D X 100) X HIDDEN_DIM
        answer_options = self.projection_layer(answer_options)

        # (B X D) X 100 X HIDDEN_DIM
        answer_options = answer_options.view(
            batch_size*ndialogues, noptions, self.hidden_dim
        )

        # (B X D) X HIDDEN_DIM => (B X D) X 100 X HIDDEN_DIM
        encoder_output = encoder_output.unsqueeze(1).expand(-1, noptions, -1)

        # (B X D) X 100 X HIDDEN_DIM * (B X D) X 100 X HIDDEN_DIM = SAME THING
        # SUM => (B X D) X 100
        scores = torch.sum(answer_options*encoder_output, dim=2)

        return scores
    

class LanguageDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs): #kwargs는 keyword argument의 줄임말로 키워드를 제공함. 딕셔너리 형태로 {'키워드': '특정값'} 이렇게 전달받음.
        super().__init__()

        self.language_lstm = nn.lstm_cell(
            in_dim + kwargs["hidden_dim"], kwargs["hidden_dim"], bias=True
        )
        self.fc = weight_norm(nn.Linear(kwargs["hidden_dim"], out_dim))
        self.dropout = nn.dropout(p=kwargs["dropout"])
        self.init_weights(kwargs["fc_bias_init"])

    def init_weights(self, fc_bias_init):
        self.fc.bias.data.fill_(fc_bias_init)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, weighted_attn):
        # Get LSTM state
        state = registry.get(f"{weighted_attn.device}_lstm_state")
        h1, c1 = state["td_hidden"]
        h2, c2 = state["lm_hidden"]

        # Language LSTM
        h2, c2 = self.language_lstm(torch.cat([weighted_attn, h1], dim=1), (h2, c2))
        predictions = self.fc(self.dropout(h2))

        # Update hidden state for t+1
        state["lm_hidden"] = (h2, c2)

        return predictions