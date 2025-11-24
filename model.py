from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Model, DebertaV2PreTrainedModel
)
import torch.nn as nn
import torch

class CustomModel(DebertaV2PreTrainedModel):
    
    def __init__(self, config):
        
        super().__init__(config)

        self.num_labels = config.num_labels

        self.deberta = DebertaV2Model(config)
       
        self.is_bilstm_layer = True

        self.bilstm = nn.LSTM(
            config.hidden_size, # Input
            config.hidden_size // 2, # Hidden
            num_layers=1, #single LSTM layer 
            dropout = config.hidden_dropout_prob, 
            batch_first= True,
            bidirectional=True
        )

        # initialize the bilstm layer 
        self.initialize_bilstm(self.bilstm)
        
        # initialize the dropout layer 
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # define the classifier layer 
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # loss function
        self.loss_fct = torch.nn.CrossEntropyLoss()

        self.post_init()

    
    def initialize_bilstm(self, lstm_layer):
        """
            Strategy 
                1. xavier uniform for input to hidden weights
                2. orthogonal for hidden to hidden weights 
                3. zero for biases
        """
        for name, param in lstm_layer.named_parameters():
            if 'weight_ih' in name: # input to hidden weights 
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0) # zero bias initialization 


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


    def forward(self, input_ids, attention_mask = None,token_type_ids=None, position_ids=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        
        return_dict =  return_dict if return_dict is not None else self.config.use_return_dict 

        outputs = self.deberta(
            input_ids, # Tokenized input [batch_size, seq_len]
            token_type_ids=token_type_ids,
            attention_mask=attention_mask, # which tokens are real vs padding 
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        output_backbone = outputs[0] # [batch_size, seq_len, 1024]
        
        # We need to model sequential dependencies just the way of LSTM 
        # BiLSTM captures sequential relationships 
        
        if self.is_bilstm_layer:
            # Apply dropout for regularization 
            output_backbone = self.dropout(output_backbone)

            # pass it to enhancement layer 
            self.bilstm.flatten_parameters()

            # Input will be [batch_size, seq_len, 1024]
            # output will be [batch_size, seq_len, 1024]
            output, hc = self.bilstm(output_backbone)

        else:
            output = output_backbone


        # Add the classifier head 
        logits = self.classifier(output)

        loss = None 
        if labels is not None:
            # Reshape for loss calculation 
            # Moving from [batch_size, seq_len, num_labels] to [batch_size * seq_len, num_labels]

            loss = self.loss_fct(
                logits.view(-1, self.num_labels), # (batch_size * seq_len, num_labels)
                labels.view(-1) #(batch_size * seq_len)
            )
            
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output)  if loss is not None else output
        
        from transformers.modeling_outputs import TokenClassifierOutput
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )






