from model import CustomModel
from transformers import AutoConfig, AutoTokenizer
import torch

def test_model_initialization():
    
    model_name = "microsoft/deberta-v3-base"
    
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 13

    # create the model 
    custom_model = CustomModel(config)

    print(f'Model Type: {type(custom_model).__name__}')
    print(f'The number of labels: {custom_model.num_labels}')
    print(f'Hidden Size:  {custom_model.config.hidden_size}')
    print(f'BiLstm enabled: {custom_model.is_bilstm_layer}')
    print(f'Total parameters: {sum([p.numel() for p in custom_model.parameters()])}')
    print(f'Trainable parameters:{sum([p.numel() for p in custom_model.parameters() if p.requires_grad])}')

    return custom_model, config


def test_forward_without_labels():

    model_name = "microsoft/deberta-v3-base"

    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 13

    model = CustomModel(config)

    batch_size = 2 
    seq_len = 32

    # define the input_ids, attention_mask 
    input_ids = torch.randint(0,model.config.hidden_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    
    model.eval()
    with torch.no_grad():
        output = model(
            input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True, output_attentions=True
        )
   

    print(f'Logits shape {output.logits.shape}')
    print(f'hidden states {output.hidden_states}')
    print(f'Attentions {output.attentions}')


def test_forward_with_labels():

    model_name = "microsoft/deberta-v3-base"
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 13

    batch_size = 2
    seq_len = 32

    model = CustomModel(config)


    input_ids = torch.randint(0, config.hidden_size,(batch_size, seq_len))
    labels = torch.randint(0, config.num_labels, (batch_size * seq_len,))
    attention_mask = torch.ones(batch_size, seq_len)


    output = model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True, output_attentions=True, return_dict=True)

    print(f'loss {output.loss}')



if __name__ == '__main__':
    test_forward_with_labels()





