import torch.nn as nn


def get_layer_lrs(lr, decay_rate, original_lr_layer_decays, num_hidden_layers):
    lrs = [lr * (decay_rate ** depth) for depth in range(num_hidden_layers + 2)]
    if original_lr_layer_decays:
        for i in range(1, len(lrs)):
            lrs[i] *= decay_rate
    return list(reversed(lrs))


def list_parameters(model, submod_name):
    return list(eval(f"model.{submod_name}").parameters())


class SentencePredictor(nn.Module):
    def __init__(self, model, hidden_size, num_class, xavier_reinited_outlayer):
        super().__init__()
        self.base_model = model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_class)
        if xavier_reinited_outlayer:
            nn.init.xavier_uniform_(self.classifier.weight.data)
            self.classifier.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids):

        print(input_ids.shape, attention_mask.shape, attention_mask.shape, token_type_ids.shape )

        x = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]
        return (
            self.classifier(self.dropout(x[:, 0, :])).squeeze(-1).float()
        )  # if regression task, squeeze to (B), else (B,#class)


def tokenize_sents_max_len(example, hf_tokenizer, cols, max_len, swap=False):
    # Follow BERT and ELECTRA, truncate the examples longer than max length
    tokens_a = hf_tokenizer.tokenize(example[cols[0]])
    tokens_b = hf_tokenizer.tokenize(example[cols[1]]) if len(cols) == 2 else []
    _max_length = max_len - 1 - len(cols)  # preserved for cls and sep tokens
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= _max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
    if swap:
        tokens_a, tokens_b = tokens_b, tokens_a
    tokens = [hf_tokenizer.cls_token, *tokens_a, hf_tokenizer.sep_token]
    token_type = [0] * len(tokens)
    if tokens_b:
        tokens += [*tokens_b, hf_tokenizer.sep_token]
        token_type += [1] * (len(tokens_b) + 1)
    example["inp_ids"] = hf_tokenizer.convert_tokens_to_ids(tokens)
    example["attn_mask"] = [1] * len(tokens)
    example["token_type_ids"] = token_type

    return example
